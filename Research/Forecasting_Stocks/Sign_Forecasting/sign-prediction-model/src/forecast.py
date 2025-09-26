import yfinance as yf
import numpy as np
from model import SignPredictionModel, prepare_data

ticker = "SPY"
data = yf.download(ticker, start="2000-01-01", end="2023-01-01")

data["LogRet"] = np.log(data["Close"]).diff()
data["Sign"] = (data["LogRet"] > 0).astype(int)
data.to_csv("SPY_data.csv")

X_train, X_test, y_train, y_test = prepare_data(data)
model = SignPredictionModel()
model.train(X_train, y_train)
accuracy, report = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")