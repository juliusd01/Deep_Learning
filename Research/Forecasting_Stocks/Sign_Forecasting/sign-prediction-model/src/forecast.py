import yfinance as yf
import numpy as np
from model import SignPredictionLogit, SignPredictionLSTM, VolaPredictionLSTM, prepare_sign_data, prepare_volatility_data

ticker = "SPY"
data = yf.download(ticker, start="2000-01-01", end="2023-01-01")

data["LogRet"] = np.log(data["Close"]).diff()
data["Sign"] = (data["LogRet"] > 0).astype(int)


### Logit Model
# X_train, X_test, y_train, y_test = prepare_sign_data(data)
# model = SignPredictionLogit()
# model.train(X_train, y_train)
# accuracy, report = model.evaluate(X_test, y_test)
# print(f"Accuracy: {accuracy}")
# print(f"Classification Report:\n{report}")

### LSTM Sign Model
X_train, X_test, y_train, y_test = prepare_sign_data(data, no_lags=10)
model = SignPredictionLSTM(no_lags=10)
model.train(X_train, y_train, epochs=10, batch_size=32)
accuracy, report = model.evaluate(X_test, y_test)
print(f"LSTM Accuracy: {accuracy}")
print(f"LSTM Classification Report:\n{report}")
model.plot_sign_predictions(y_true=y_test, y_pred=model.predict(X_test), title="LSTM Sign Prediction (Test)")

### LSTM Volatility Model
# X_train, X_test, y_train, y_test = prepare_volatility_data(data, no_lags=10)
# model = VolaPredictionLSTM(no_lags=10)
# model.train(X_train, y_train, epochs=10, batch_size=32)
# mse = model.evaluate(X_test, y_test)
# print(f"LSTM Volatility MSE: {mse}")
# model.plot_volatility_predictions(X_test, y_test, title="Volatility Prediction (Test)")
