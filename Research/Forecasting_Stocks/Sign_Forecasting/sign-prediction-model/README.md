# Sign Prediction Model

This project is designed to predict the sign of stock price movements using historical data. The initial code downloads stock data, calculates log returns, and generates a sign indicator.

Features used by [Stanford Paper](http://cs230.stanford.edu/projects_fall_2019/reports/26254244.pdf)

## TODOs
- Forecast different horizons using LSTM
- Standardize only the training set (if any standardization at all)
- Loss must be standardized, otherwise minimizing over both components would be a bit useless as the one with bigger scales dominates the total loss

Potentially have two neurons at the end, one for vola, the other for sign. For "true" vola, just take GARCH(1,1)
