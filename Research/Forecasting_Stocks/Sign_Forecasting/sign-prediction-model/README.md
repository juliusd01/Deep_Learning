# Sign Prediction Model

This project is designed to predict the sign of stock price movements using historical data. The initial code downloads stock data, calculates log returns, and generates a sign indicator based on those returns. The project will include a model that utilizes this data to make predictions.

Features used by [Stanford Paper](http://cs230.stanford.edu/projects_fall_2019/reports/26254244.pdf):

Our features consist of several transformations of the given
Open, High, Low, Close, Volume data. We use Log Return, Log
Volume Change, Log Trading Range (high vs. low for a given
trading day), Previous 30-day Volatility, Previous 10-Day
Volatility, and GARCH forward-looking 10-day volatility
prediction as our features. 