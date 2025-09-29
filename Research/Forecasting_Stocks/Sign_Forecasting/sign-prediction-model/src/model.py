import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class SignPredictionModel:
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

def prepare_data(data):
    # Create lag features for the past 5 days for LogRet and Sign
    data = data.iloc[1:]
    data = data.copy()
    for lag in range(1, 6):
        data[f'LogRet_lag{lag}'] = data['LogRet'].shift(lag)
        data[f'Sign_lag{lag}'] = data['Sign'].shift(lag)

    # Target is tomorrow's sign to predict next-day movement
    data['Sign_tomorrow'] = data['Sign'].shift(-1)

    # Drop rows with any NA due to shifting
    data = data.dropna().copy()

    feature_cols = [
        'LogRet_lag1', 'LogRet_lag2', 'LogRet_lag3', 'LogRet_lag4', 'LogRet_lag5',
        'Sign_lag1', 'Sign_lag2', 'Sign_lag3', 'Sign_lag4', 'Sign_lag5'
    ]

    X = data[feature_cols]
    y = data['Sign_tomorrow']

    print(data)
    print(X)

    # Ensure target has both classes present
    if y.nunique() < 2:
        raise ValueError("Target variable has only one class after preprocessing. Check data range and feature shifts.")

    # Use chronological split (no shuffling) to avoid leakage in time series
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)