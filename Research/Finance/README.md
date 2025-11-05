# Stock return prediction Literature
- in the 80's and 90's, several papers showed correlations (and therefore predictability) of variables like short and long-term treasury, corporate bonds, valuation ratios, interest rates, [relative valuation of high beta stocks to low-beta stocks](https://personal.lse.ac.uk/polk/research/ptv.pdf) with the stock returns (e.g. Fama and French 1989).
- Goyal and Welch (GW, 2008) showed that many of these predictors are not significant for stock return forecasting over long time horizons than the original study. There is critiscm on the interpretation of the OOS R^2 by GW, since even with ngative OOS R^2 positive Sharpe ratios are obtainable (see [Kelly et al.](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13298) ).


## Forecasting Stocks with high-frequency data

### On the forecasting of high-frequency financial time seriesbased on ARIMA model improved by deep learning
https://onlinelibrary.wiley.com/doi/full/10.1002/for.2677

- Use ARIMA on high frequency data, expand ARIMA with LSTM
- Deep Learning is used as error correction which can compensate for nonlinear features in high-frequency settings:

![alt text](images/ARIMA_LSTM.png)

- ARIMA models makes predictions and LSTM forecasts the error. These two forecasts are then added together. 
- Basically, LSTM is just predicting the sequence of residuals based on the last observed residuals
- Diebold-Mariano Test for testing forecasting accuracy of different models (gives a answer to the question which model did better forecasting)

#### Further Research Ideas
- Apply same approach to optimize forecasting models such as GARCH for the volatility of high-frequency time series

### Volatility forecasting for stock market incorporatingmacroeconomic variables based on GARCH-MIDAS and deep learning models
https://onlinelibrary.wiley.com/doi/epdf/10.1002/for.2899

- incorporate macroeconomoic variables to forecast short-term volatility within a GARCH-MIDAS model
- Use short-term volatility as input indicator to ML/DL models to forecast the realized volatility
- DL model used: gated recurrent unit (GRU) which is based on LSTM

![alt text](images/GRU_LSTM.png)