import numpy as np
import pandas as pd
import talib
from prophet import Prophet
from utils.clickhouse_data import compute_trend_features, compute_volatility_features

def preprocess_data(df):
    """Preprocesses data by handling missing values and normalizing"""
    df = extract_all_features(df)
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.fillna(0, inplace=True)  # Fill any remaining NaNs
    return df

def extract_all_features(df):
    """
    Extracts features while optimizing memory usage and ensuring TA-Lib compatibility.
    """

    # Select only necessary columns
    raw_cols = [
        'date', 'open', 'high', 'low', 'close', 'lastprice', 'previouscloseprice',
        'volume', 'totaltradingvolume', 'totaltradevalue', 'totalnumberoftradesexecuted'
    ]
    df = df[raw_cols].copy()

    # Convert date column and set index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # === Compute Derived Features ===
    df = compute_trend_features(df)
    df = compute_volatility_features(df)

    # Convert numerical columns to float64 to avoid TA-Lib errors
    df['close'] = df['close'].astype(np.float64)
    df['high'] = df['high'].astype(np.float64)
    df['low'] = df['low'].astype(np.float64)
    df['open'] = df['open'].astype(np.float64)
    df['volume'] = df['volume'].astype(np.float64)

    # Liquidity & Microstructure Features
    df['prev_close_return'] = np.log(df['close'] / df['previouscloseprice'].astype(np.float64))
    df['trading_intensity'] = df['totalnumberoftradesexecuted'].astype(np.float64) / df['totaltradingvolume'].astype(np.float64)
    df['turnover_ratio'] = df['totaltradevalue'].astype(np.float64) / df['totaltradingvolume'].astype(np.float64)

    # Compute VWAP-based features only if VWAP is available
    if 'vwap' in df.columns:
        df['vwap'] = df['vwap'].astype(np.float64)
        df['vwap_ratio'] = df['close'] / df['vwap']

    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']

    # === Advanced Technical Indicators (Ensure float64) ===
    high, low, close, volume = (
        df['high'].to_numpy(dtype=np.float64, copy=False), 
        df['low'].to_numpy(dtype=np.float64, copy=False), 
        df['close'].to_numpy(dtype=np.float64, copy=False), 
        df['volume'].to_numpy(dtype=np.float64, copy=False)
    )

    df['cci_20'] = talib.CCI(high, low, close, timeperiod=20)
    df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

    # Compute stochastic indicators in a single call to save memory
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    df['chande_momentum'] = talib.CMO(close, timeperiod=14)
    df['ulcer_index'] = np.sqrt((pd.Series(close).rolling(14).max().to_numpy(copy=False) - close) ** 2 / 14)
    df['obv'] = talib.OBV(close, volume)

    # Compute Chaikin Money Flow only if volume is nonzero
    if np.any(volume > 0):
        df['chaikin_money_flow'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

    # === Extract Seasonality Features Using Prophet (Optimized) ===
    df_prophet = df[['close']].reset_index()
    df_prophet.columns = ['ds', 'y']

    prophet = Prophet(
        yearly_seasonality=1,  # Reduce memory load by limiting seasonalities
        weekly_seasonality=1, 
        daily_seasonality=False, 
        changepoint_prior_scale=0.05  # Reduce complexity to save memory
    )
    prophet.fit(df_prophet)

    future = prophet.make_future_dataframe(periods=0)  # No extra periods needed
    forecast = prophet.predict(future)

    # Use only necessary seasonal components to save memory
    df['seasonal_weekly'] = forecast['weekly'].to_numpy(dtype=np.float64, copy=False)
    df['seasonal_yearly'] = forecast['yearly'].to_numpy(dtype=np.float64, copy=False)

    return df
