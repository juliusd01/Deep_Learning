import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical


class SignPredictionLogit:
    """Simple Logistic Regression model for sign prediction."""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report
    

class SignPredictionLSTM:
    """LSTM model for sign prediction."""
    def __init__(self, no_lags=5):
        self.no_lags = no_lags
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(no_lags, 2)))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['precision'])

    def train(self, X, y, epochs=10, batch_size=32):
        y_cat = to_categorical(y)
        X_lstm = X.values.reshape((X.shape[0], self.no_lags, 2))
        self.model.fit(X_lstm, y_cat, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        X_lstm = X.values.reshape((X.shape[0], self.no_lags, 2))
        predictions = self.model.predict(X_lstm)
        return np.argmax(predictions, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        print("-----\nPred: ",predictions)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report
    

class VolaPredictionLSTM:
    """LSTM model for volatility prediction."""
    def __init__(self, no_lags=5):
        self.no_lags = no_lags
        self.model = Sequential()
        self.model.add(LSTM(50, dropout=0.2, input_shape=(no_lags, 2)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=10, batch_size=32):
        X_lstm = X.values.reshape((X.shape[0], self.no_lags, 2))
        self.model.fit(X_lstm, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        X_lstm = X.values.reshape((X.shape[0], self.no_lags, 2))
        return self.model.predict(X_lstm).flatten()

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        return mse
    
    def plot_volatility_predictions(model, X, y, title="Volatility Prediction (Train)"):
        """Plot actual vs predicted volatility."""
        y_pred = model.predict(X)
        plt.figure(figsize=(12, 5))
        plt.plot(y.values, label="Actual Volatility")
        plt.plot(y_pred, label="Predicted Volatility")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Volatility")
        plt.legend()
        plt.tight_layout()
        plt.show()



def prepare_sign_data(data, no_lags=5):
    # Create lag features for the past 5 days for LogRet and Sign
    data = data.iloc[1:]
    data = data.copy()
    for lag in range(1, no_lags + 1):
        data[f'LogRet_lag{lag}'] = data['LogRet'].shift(lag)
        data[f'Sign_lag{lag}'] = data['Sign'].shift(lag)

    # Target is tomorrow's sign to predict next-day movement
    data['Sign_tomorrow'] = data['Sign'].shift(-1)

    # Drop rows with any NA due to shifting
    data = data.dropna().copy()
    
    feature_cols = []
    for lag in range(1, no_lags + 1):
        feature_cols.append(f'LogRet_lag{lag}')
        feature_cols.append(f'Sign_lag{lag}')

    X = data[feature_cols]
    y = data['Sign_tomorrow']

    # Ensure target has both classes present
    if y.nunique() < 2:
        raise ValueError("Target variable has only one class after preprocessing. Check data range and feature shifts.")

    # Use chronological split (no shuffling) to avoid leakage in time series
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


def prepare_volatility_data(data, no_lags=5, rolling_window=5):
    data = data.copy()
    data["volatility"] = data["LogRet"].rolling(window=rolling_window).std()
    data = data.iloc[rolling_window:]
    
    # Target variable: Volatility tomorrow
    data['volatility_tomorrow'] = data['volatility'].shift(-1)

    # Create lag features for the past `no_lags` days for LogRet and volatility
    for lag in range(1, no_lags + 1):
        data[f'LogRet_lag{lag}'] = data['LogRet'].shift(lag)
        data[f'volatility_lag{lag}'] = data['volatility'].shift(lag)
    data = data.dropna().copy()
    
    feature_cols = []
    for lag in range(1, no_lags + 1):
        feature_cols.append(f'LogRet_lag{lag}')
        feature_cols.append(f'volatility_lag{lag}')
    X = data[feature_cols]
    y = data['volatility_tomorrow']

    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)