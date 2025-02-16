# market-regime


Hybrid Approach for Market Regime Labeling (Large, Mid, Small Caps)

The goal is to build a robust market regime labeling model that is both generalized across stocks and specialized for each category (large-cap, mid-cap, small-cap). This is critical because each stock category behaves differently in terms of volatility, liquidity, and momentum.

## Feature Engineering and Extraction

1. Momentum 
2. Volatility
3. Liquidity
4. Bollinger Bands 
5. William %R
6. MACD
7. Marubozu
8. Exponenial weighted moving moments. 
9. z-score, kurtosis, skewness of prices
10. Polynomial trendline on 7 days. (might not use this one)


### Feature Extraction and selection

Rule based is not working, below are the only option for transfer learning. 

1. Autoencoders
2. Contanstive Learning (Self Supervised)
3. HMM or GMM (Tested this on single stock, its not that great). Maybe the feature are not that great.