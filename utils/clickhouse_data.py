import pandas as pd
import modin.pandas as mpd
from clickhouse_driver import Client
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import talib

def load_ticker_data(ticker_path):
    return pd.read_csv(ticker_path)

def get_ticker_list(ticker_df):
    return ticker_df['Ticker'].tolist()

def fetch_financialinstrumentid_values(ticker_df):
    ticker_list = get_ticker_list(ticker_df)
    client = Client('localhost', user='user', password='password', database='stock_data')
    query = f"SELECT financialinstrumentid, tickersymbol, financialinstrumentname FROM stock_data.tickers WHERE tickersymbol IN {tuple(ticker_list)}"
    result = client.execute(query)
    financialinstrumentid_df = pd.DataFrame(result, columns=['financialinstrumentid', 'tickersymbol', 'financialinstrumentname'])
    return financialinstrumentid_df.drop_duplicates()

def clickhouse_data(ticker_path, start_date, end_date, size=10000):
    ticker_df = load_ticker_data(ticker_path)
    financialinstrumentid_list = fetch_financialinstrumentid_values(ticker_df)["financialinstrumentid"].tolist()
    
    # Connect to the ClickHouse instance
    clickhouse_client = Client('localhost', user='user', password='password', database='stock_data')
    
    # Define the base query with filtering by financialinstrumentid_list and date range
    base_query = f"""
    SELECT 
        financialinstrumentid,
        date,
        open,
        high,
        low,
        close,
        lastprice,
        previouscloseprice,
        volume,
        totaltradingvolume,
        totaltradevalue,
        totalnumberoftradesexecuted,
        tickersymbol,
        securityseries,
        settlementprice,
        financialinstrumentname
    FROM stock_data.tickers
    WHERE financialinstrumentid IN {tuple(financialinstrumentid_list[:size])} AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY financialinstrumentid, date ASC
    """

    print(base_query)
    # Execute the query and fetch the result
    result = clickhouse_client.execute(base_query)
    
    # Convert the result to a Modin DataFrame
    df = mpd.DataFrame(result, columns=[
        'financialinstrumentid', 'date', 'open', 'high', 'low', 'close', 'lastprice', 'previouscloseprice', 'volume', 
        'totaltradingvolume', 'totaltradevalue', 'totalnumberoftradesexecuted', 'tickersymbol', 
        'securityseries', 'settlementprice', 'financialinstrumentname'
    ])
    
    return df

def clickhouse_largecap(start_date, end_date, size):
    largecap_ticker_path = "/Users/akash/personal/finance/technical-analysis/data/metadata/largecap.csv"
    return clickhouse_data(largecap_ticker_path, start_date, end_date, size)

def clickhouse_midcap(start_date, end_date, size):
    midcap_ticker_path = "/Users/akash/personal/finance/technical-analysis/data/metadata/midcap.csv"
    return clickhouse_data(midcap_ticker_path, start_date, end_date, size)

def clickhouse_smallcap(start_date, end_date, size):
    smallcap_ticker_path = "/Users/akash/personal/finance/technical-analysis/data/metadata/smallcap.csv"
    return clickhouse_data(smallcap_ticker_path, start_date, end_date, size)

def fill_missing_dates_modin_optimized(df: mpd.DataFrame) -> mpd.DataFrame:
    """
    Fill missing dates in the given Modin DataFrame for each 'financialinstrumentid'.
    Assumes 'financialinstrumentid' and 'date' columns exist, and 'date' is a datetime column.
    
    Args:
    - df (modin.pandas.DataFrame): The input DataFrame with 'financialinstrumentid' and 'date' columns.
    
    Returns:
    - modin.pandas.DataFrame: The DataFrame with missing dates filled.
    """
    # Ensure 'date' is a datetime column
    df['date'] = pd.to_datetime(df['date'])

    # Create a date range covering all possible dates from min to max 'date'
    full_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')

    # Prepare a DataFrame with full date range
    full_dates_df = pd.DataFrame({'date': full_dates})

    # Function to fill missing dates for each 'financialinstrumentid'
    def fill_for_financialinstrumentid(group: pd.DataFrame) -> pd.DataFrame:
        financialinstrumentid = group['financialinstrumentid'].iloc[0]  # Assume all rows in a group have the same 'financialinstrumentid'
        
        # Create a DataFrame with the full date range for this 'financialinstrumentid'
        full_group_dates = pd.DataFrame({'date': full_dates})
        full_group_dates['financialinstrumentid'] = financialinstrumentid
        
        # Merge with the existing group to fill missing dates
        merged_group = pd.merge(full_group_dates, group, on=['financialinstrumentid', 'date'], how='left')
        
        # Optional: Fill missing values (e.g., forward fill or other filling methods)
        merged_group = merged_group.ffill()
        
        # Convert numeric columns to the appropriate type (if needed)
        numeric_cols = merged_group.select_dtypes(include=['float64', 'int64']).columns
        merged_group[numeric_cols] = merged_group[numeric_cols].apply(pd.to_numeric, downcast='float')
        
        return merged_group

    # Apply the fill operation to each group (group by 'financialinstrumentid')
    filled_df = df.groupby('financialinstrumentid').apply(fill_for_financialinstrumentid).reset_index(drop=True)

    # Optional: Remove any duplicate rows if they exist
    filled_df = filled_df.drop_duplicates(subset=['financialinstrumentid', 'date'])

    return filled_df

def get_name_by_financialinstrumentid(financialinstrumentid, df):
    # Set 'financialinstrumentid' as the index
    df = df.set_index('financialinstrumentid')
    
    # Check if the financialinstrumentid exists in the index
    if financialinstrumentid not in df.index:
        # Handle the case where no match is found
        return None  # Or you can raise an error or return a default value
    
    # Extract the 'financialinstrumentname' for the given financialinstrumentid
    company_names = df.loc[financialinstrumentid, 'financialinstrumentname'].unique()
    
    # If there are multiple names, handle the case, for example, return the first one
    if len(company_names) > 1:
        return company_names  # Or you can return a list, or raise an error
    
    # Return the unique company name
    return company_names

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on stock data for clustering based on liquidity and market activity.
    
    Args:
    - df (pandas.DataFrame): The stock data frame with necessary columns.
    
    Returns:
    - pandas.DataFrame: DataFrame with new features for clustering.
    """
    # Ensure that 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Set the index for time-based operations (if not already set)
    df.set_index(['financialinstrumentid', 'date'], inplace=True)
    df['close'].ffill(inplace=True)
    
    # --- Feature 1: Log Returns
    df['log_returns'] = df.groupby('financialinstrumentid')['close'].transform(lambda x: np.log(x / x.shift(1)))
    
    # --- Feature 2: Percentage Returns
    df['pct_returns'] = df['close'].pct_change() * 100
    
    # --- Feature 3: Rolling Volatility (14-day and extended for multiple periods)
    periods = [1, 3, 5, 7, 14, 30, 60, 90]
    for period in periods:
        df[f'rolling_std_{period}'] = df.groupby('financialinstrumentid')['close'].transform(lambda x: x.rolling(period).std())
        df[f'realized_volatility_{period}'] = df.groupby('financialinstrumentid')['log_returns'].transform(
            lambda x: np.sqrt((x**2).rolling(period).sum())
        )
    
    # --- Feature 4: Cumulative Returns
    df['cumulative_returns'] = df.groupby('financialinstrumentid')['log_returns'].transform(lambda x: x.cumsum())
    df['cumulative_pct_returns'] = (1 + df['pct_returns'] / 100).groupby(df.index.get_level_values('financialinstrumentid')).cumprod() - 1
    
    # --- Feature 5: Skewness and Kurtosis (30-day rolling window)
    for period in [1, 3, 5, 7, 14, 30, 60, 90]:
        df[f'skewness_{period}'] = df.groupby('financialinstrumentid')['log_returns'].transform(lambda x: x.rolling(period).skew())
        df[f'kurtosis_{period}'] = df.groupby('financialinstrumentid')['log_returns'].transform(lambda x: x.rolling(period).kurt())
    
    # --- Feature 6: Z-Score Normalization
    df['z_score'] = (
        (df['log_returns'] - df.groupby('financialinstrumentid')['log_returns'].transform('mean')) / 
        df.groupby('financialinstrumentid')['log_returns'].transform('std')
    )
    
    # --- Feature 7: Autocorrelation (14-day rolling window)
    for period in periods:
        df[f'autocorr_{period}'] = df.groupby('financialinstrumentid')['log_returns'].transform(
            lambda x: x.rolling(period).apply(lambda y: y.autocorr(), raw=False)
        )
    
    # --- Additional Features for different dated periods
    extended_periods = [1, 3, 5, 7, 14, 30, 60, 90]
    for period in extended_periods:
        df[f'close_pct_change_{period}'] = df.groupby('financialinstrumentid')['close'].transform(lambda x: x.pct_change(period) * 100)
        df[f'volume_ma_{period}'] = df.groupby('financialinstrumentid')['volume'].transform(lambda x: x.rolling(period).mean())
        df[f'close_ma_{period}'] = df.groupby('financialinstrumentid')['close'].transform(lambda x: x.rolling(period).mean())
        df[f'rolling_std_{period}'] = df.groupby('financialinstrumentid')['close'].transform(lambda x: x.rolling(period).std())
        df[f'momentum_roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    
    # --- Liquidity Feature
    if 'marketcap' in df.columns:
        df['liquidity'] = df['volume'] / df['marketcap']
    
    # --- Volume-Price Correlation (applies to grouped data by financialinstrumentid)
    df['volume_price_corr'] = df.groupby('financialinstrumentid').apply(
        lambda x: x['volume'].corr(x['close'].pct_change())
    ).reset_index(drop=True)
    
    # --- Price Range
    df['price_range'] = df['close'] - df['open']
    
    # --- Moving Averages and Momentum
    df['close_ma_30'] = df.groupby('financialinstrumentid')['close'].transform(lambda x: x.rolling(30).mean())
    df['momentum_roc'] = (df['close'] - df['close'].shift(14)) / df['close'].shift(14) * 100

    # --- Sharpe Ratio (using log returns, risk-free rate assumed to be 0)
    for period in extended_periods:
        df[f'sharpe_ratio_{period}'] = df.groupby('financialinstrumentid')['log_returns'].transform(
            lambda x: (x.rolling(period).mean() / x.rolling(period).std())
        )

    # --- Sortino Ratio (using log returns, assuming a target return of 0)
    for period in extended_periods:
        df[f'sortino_ratio_{period}'] = df.groupby('financialinstrumentid')['log_returns'].transform(
            lambda x: (x.rolling(period).mean() / x[x < 0].rolling(period).std()) if x[x < 0].rolling(period).std().any() > 0 else np.nan
        )
    
    # --- Calmar Ratio (using 1-year return and max drawdown)
    for period in extended_periods:
        df['calmar_ratio'] = df.groupby('financialinstrumentid')['cumulative_returns'].transform(lambda x: x.rolling(period).mean() / abs(x.rolling(period).min()))
    
    return df

def decompose_time_series_multiindex(df: mpd.DataFrame, period=30) -> mpd.DataFrame:
    """
    Decompose time series data into trend, seasonal, and residual components for each financial instrument.

    Args:
    - df (pandas.DataFrame): Multi-index DataFrame with ['financialinstrumentid', 'date'] index and 'close' column.
    - period (int): The period for seasonal decomposition.

    Returns:
    - pandas.DataFrame: DataFrame with decomposed components for each financial instrument.
    """
    # Ensure index is set correctly for time-series decomposition
    if not isinstance(df.index, mpd.MultiIndex):
        raise ValueError("Input DataFrame must have a MultiIndex with ['financialinstrumentid', 'date'].")
    
    # Ensure the index is sorted for proper decomposition
    df = df.sort_index()

    # Define a helper function for decomposition
    def decompose_group(group):
        result = seasonal_decompose(group['close'], model='additive', period=period, extrapolate_trend='freq')
        group['trend'] = result.trend
        group['seasonal'] = result.seasonal
        group['residual'] = result.resid
        return group

    # Apply the decomposition to each financial instrument group
    df = df.groupby(level='financialinstrumentid', group_keys=False).apply(decompose_group)
    
    return df

def add_lagged_features_multiindex(df: mpd.DataFrame, target_column='close', lags=5) -> mpd.DataFrame:
    """
    Add lagged features to a multi-index DataFrame for each financial instrument.

    Args:
    - df (mpd.DataFrame): Multi-index DataFrame with ['financialinstrumentid', 'date'] as index levels.
    - target_column (str): The column for which lagged features are to be created.
    - lags (int): The number of lagged features to create.

    Returns:
    - mpd.DataFrame: DataFrame with new lagged features.
    """
    # Ensure the DataFrame has a MultiIndex
    if not isinstance(df.index, mpd.MultiIndex):
        raise ValueError("Input DataFrame must have a MultiIndex with ['financialinstrumentid', 'date'].")
    
    # Sort the index for consistency
    df = df.sort_index()

    # Define a helper function to add lagged features to each group
    def add_lags(group):
        for lag in range(1, lags + 1):
            group[f'{target_column}_Lag_{lag}'] = group[target_column].shift(lag)
        return group

    # Apply the lagged feature creation for each financial instrument
    df = df.groupby(level='financialinstrumentid', group_keys=False).apply(add_lags)
    
    return df


def add_stochastic(df, timeframes=[3, 5, 7, 14, 30, 60, 90]):
    for tf in timeframes:
        k, d = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=tf, slowk_period=3, slowd_period=3)
        df[f'stoch_%K_{tf}'] = k.fillna(0)
        df[f'stoch_%D_{tf}'] = d.fillna(0)

def add_donchian_channels(df, timeframes=[3, 5, 7, 14, 30, 60, 90]):
    for tf in timeframes:
        df[f'donchian_high_{tf}'] = df['high'].rolling(window=tf, min_periods=1).max()
        df[f'donchian_low_{tf}'] = df['low'].rolling(window=tf, min_periods=1).min()
        df[f'donchian_mid_{tf}'] = (df[f'donchian_high_{tf}'] + df[f'donchian_low_{tf}']) / 2

def add_bollinger_bands(df, timeframes=[3, 5, 7, 14, 30, 60, 90]):
    for tf in timeframes:
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=tf)
        df[f'bb_upper_{tf}'] = upper.fillna(method='bfill')
        df[f'bb_middle_{tf}'] = middle.fillna(method='bfill')
        df[f'bb_lower_{tf}'] = lower.fillna(method='bfill')

def add_macd(df):
    macd, macd_signal, macd_hist = talib.MACD(df['close'])
    df['macd'] = macd.fillna(0)
    df['macd_signal'] = macd_signal.fillna(0)
    df['macd_hist'] = macd_hist.fillna(0)

def add_rsi(df, timeframes=[3, 5, 7, 14, 30, 60, 90]):
    for tf in timeframes:
        df[f'rsi_{tf}'] = talib.RSI(df['close'], timeperiod=tf).fillna(0)

def add_atr(df, timeframes=[3, 5, 7, 14, 30, 60, 90]):
    for tf in timeframes:
        df[f'atr_{tf}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=tf).fillna(0)

def add_adx(df, timeframes=[3, 5, 7, 14, 30, 60, 90]):
    for tf in timeframes:
        df[f'adx_{tf}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=tf).fillna(0)

def add_candlestick_patterns(df):
    pattern_list = {
        'CDLDOJI': talib.CDLDOJI, 'CDLENGULFING': talib.CDLENGULFING,
        'CDLHARAMI': talib.CDLHARAMI, 'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR
    }
    for pattern, func in pattern_list.items():
        df[pattern] = func(df['open'], df['high'], df['low'], df['close'])

def add_anchored_vwap(df):
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def add_ichimoku(df):
    conversion_line = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    base_line = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    leading_span_a = ((conversion_line + base_line) / 2).shift(26)
    leading_span_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    lagging_span = df['close'].shift(-26)

    df['ichimoku_conversion'] = conversion_line.fillna(method='bfill')
    df['ichimoku_base'] = base_line.fillna(method='bfill')
    df['ichimoku_span_a'] = leading_span_a.fillna(method='bfill')
    df['ichimoku_span_b'] = leading_span_b.fillna(method='bfill')
    df['ichimoku_lagging'] = lagging_span.fillna(method='bfill')

def add_fibonacci(df, start_idx=0, end_idx=-1):
    """ Adds Fibonacci retracement levels based on a price swing. """
    if end_idx == -1:
        end_idx = len(df) - 1  # Use last index if unspecified

    high_price = df['high'][start_idx:end_idx].max()
    low_price = df['low'][start_idx:end_idx].min()
    diff = high_price - low_price

    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    for level in fib_levels:
        df[f'fib_{level}'] = high_price - (diff * level)

def add_composite_signals(df):
    df['buy_signal'] = (
        (df['macd'] > df['macd_signal']) &
        (df['rsi_14'] > 50) &
        (df['stoch_%K_14'] > df['stoch_%D_14']) &
        (df['adx_14'] > 20)
    ).astype(int)

    df['sell_signal'] = (
        (df['macd'] < df['macd_signal']) &
        (df['rsi_14'] < 50) &
        (df['stoch_%K_14'] < df['stoch_%D_14']) &
        (df['adx_14'] < 20)
    ).astype(int)

def calculate_technical_indicators(df):
    add_stochastic(df)
    add_donchian_channels(df)
    add_bollinger_bands(df)
    add_macd(df)
    add_rsi(df)
    add_atr(df)
    add_adx(df)
    add_candlestick_patterns(df)
    add_anchored_vwap(df)
    add_ichimoku(df)
    add_fibonacci(df)
    add_composite_signals(df)

    return df

def impute_missing_values(df, window=7):
    df = df.ffill()  # Forward fill first
    df = df.fillna(df.rolling(window=window, min_periods=1).median())  # Rolling median (past only)
    df = df.fillna(0)  # Fill remaining NaNs with 0 (if any)
    return df

def compute_trend_features(df):
    """
    Computes trend-based features from OHLCV data.
    
    Features:
    - Log Returns
    - Momentum (10, 20, 50-day)
    - RSI (14-day)
    - SMA (20, 50, 200-day)
    - EMA (12, 26-day)
    - MACD (Signal Line)
    - Hurst Exponent (Trend Persistence)
    """
    df = df.copy()

    # Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Momentum
    for window in [10, 20, 50]:
        df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)

    # Relative Strength Index (RSI) - Fixing TA-Lib Input
    df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)  # Convert Series to NumPy array

    # Simple Moving Averages (SMA)
    for window in [20, 50, 200]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()

    # Exponential Moving Averages (EMA) - Fixing TA-Lib Input
    df['ema_12'] = talib.EMA(df['close'].values, timeperiod=12)  # Convert Series to NumPy array
    df['ema_26'] = talib.EMA(df['close'].values, timeperiod=26)  # Convert Series to NumPy array

    # MACD - Fixing TA-Lib Input
    macd, macd_signal, _ = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal

    # Hurst Exponent (Trend Strength)
    def compute_hurst(series):
        """Computes the Hurst Exponent to measure trend strength."""
        if len(series) < 20:
            return np.nan
        lags = range(2, 20)
        tau = [np.std(series.diff(lag).dropna()) for lag in lags]
        hurst_exp = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return hurst_exp

    df['hurst_50'] = df['close'].rolling(50).apply(compute_hurst, raw=False)

    return df



def compute_volatility_features(df):
    """
    Computes volatility-based features.
    
    Features:
    - Realized Volatility (10, 20, 50-day)
    - Garman-Klass Volatility
    - ATR Ratio
    - Bollinger Band Width
    """
    df = df.copy()

    # Realized Volatility
    for window in [10, 20, 50]:
        df[f'realized_vol_{window}'] = df['log_return'].rolling(window).std()

    # Garman-Klass Volatility
    log_hl = np.log(df['high'] / df['low']) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2
    df['gk_vol_20'] = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean()

    # ATR Ratio (Average True Range / Close Price)
    df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    df['atr_ratio'] = df['atr_14'].values / df['close'].values

    # Bollinger Band Width
    upper_bb, middle_bb, lower_bb = talib.BBANDS(df['close'].values, timeperiod=20)
    df['bollinger_band_width'] = (upper_bb - lower_bb) / middle_bb

    return df

def compute_liquidity_features(df):
    """
    Computes liquidity-based features.
    
    Features:
    - Volume Z-Score
    - Price Impact Ratio
    - Volume-to-Range Ratio
    """
    df = df.copy()

    # Volume Z-Score
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

    # Price Impact Ratio (Captures liquidity shocks)
    df['price_impact'] = abs(df['log_return']) / df['volume']

    # Volume-to-Range Ratio (Liquidity depth indicator)
    df['volume_range_ratio'] = df['volume'] / (df['high'] - df['low'])

    return df

def compute_microstructure_features(df):
    """
    Computes market microstructure-based features.
    
    Features:
    - VWAP
    - Order Flow Imbalance (if LOB data is available)
    - Bid-Ask Spread (if available)
    """
    df = df.copy()

    # Volume-Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Order Flow Imbalance & Bid-Ask Spread would require LOB (Level 2) data.
    return df
