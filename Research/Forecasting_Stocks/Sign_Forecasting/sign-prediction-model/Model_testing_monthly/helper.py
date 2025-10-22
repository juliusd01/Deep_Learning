import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.metrics import accuracy_score, classification_report

from models import SignLSTM, SignGRU

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def create_sequences(X, y_sign, window):
    """Convert the data into sequences for model input.

    :param X: Feature data
    :type X: np.ndarray
    :param y_sign: Target sign data
    :type y_sign: np.ndarray
    :param window: Number of time steps (lags)
    :type window: int
    :return: Tuple of (X sequences with shape: (num_samples, window, num_features), y signs)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window : i, :])
        ys.append(y_sign[i])
    return np.array(Xs), np.array(ys)


def run_sign_model(X:np.ndarray, y:np.ndarray, number_of_lags:list[int], train_ratio:float,
            hidden_dim:list[int], num_layers:list[int], dropout:list[float], lr:list[float], 
            batch_size:list[int], epochs:list[int], class_threshold:list[float], model_type:str) -> pd.DataFrame:
    """Run grid search for LSTM or GRU sign prediction model.

    :param X: Feature data as numpy array
    :type X: np.ndarray
    :param y: Target sign data as numpy array
    :type y: np.ndarray
    :param number_of_lags: List of lag values to try
    :type number_of_lags: list[int]
    :param train_ratio: Ratio of data to use for training
    :type train_ratio: float
    :param hidden_dim: List of hidden dimension sizes to try
    :type hidden_dim: list[int]
    :param num_layers: List of number of layers to try
    :type num_layers: list[int]
    :param dropout: List of dropout rates to try
    :type dropout: list[float]
    :param lr: List of learning rates to try
    :type lr: list[float]
    :param batch_size: List of batch sizes to try
    :type batch_size: list[int]
    :param epochs: List of epoch counts to try
    :type epochs: list[int]
    :param class_threshold: List of classification thresholds to try
    :type class_threshold: list[float]
    :param model_type: Type of model to use ('LSTM' or 'GRU')
    :type model_type: str
    :return: DataFrame with results of all model configurations
    :rtype: pd.DataFrame
    """
    
    results = []
    no_iterations = (len(number_of_lags) * len(hidden_dim) * len(num_layers) * len(dropout) *
                     len(lr) * len(batch_size) * len(epochs) * len(class_threshold))
    i = 0
    # Grid Search
    for num_lags in number_of_lags:
        X_seq, y_seq = create_sequences(X, y, num_lags)
        train_size = int(len(X_seq) * train_ratio)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        ys_train, ys_test = y_seq[:train_size], y_seq[train_size:]
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        ys_train_tensor = torch.tensor(ys_train, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, ys_train_tensor)
        for batch_size in batch_size:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for hidden_dimension in hidden_dim:
                for number_of_layers in num_layers:
                    for drop_out in dropout:
                        for learning_rate in lr:
                            for epochs_i in epochs:
                                if model_type == 'LSTM':
                                    model = SignLSTM(input_dim=X_train.shape[2], hidden_dim=hidden_dimension, num_layers=number_of_layers, dropout=drop_out)
                                elif model_type == 'GRU':
                                    model = SignGRU(input_dim=X_train.shape[2], hidden_dim=hidden_dimension, num_layers=number_of_layers, dropout=drop_out)
                                else:
                                    raise ValueError("Model type not recognized. Use 'LSTM' or 'GRU'.")
                                model = SignLSTM(input_dim=X_train.shape[2], hidden_dim=hidden_dimension, num_layers=number_of_layers, dropout=drop_out)
                                criterion = nn.BCELoss()
                                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                model.train()
                                for _ in range(epochs_i):
                                    for X_batch, y_batch in train_loader:
                                        optimizer.zero_grad()
                                        y_pred = model(X_batch)
                                        loss = criterion(y_pred, y_batch)
                                        loss.backward()
                                        optimizer.step()
                                        # slightly wrong loss, as last batch might be smaller
                                model.eval()
                                with torch.no_grad():
                                    y_test_pred = model(X_test_tensor).numpy()
                                for class_thresh in class_threshold:
                                    y_pred_sign = (y_test_pred >= class_thresh).astype(int)
                                    accuracy = accuracy_score(ys_test, y_pred_sign)
                                    report = classification_report(ys_test, y_pred_sign, output_dict=True)
                                    results.append({
                                        'num_lags': num_lags,
                                        'batch_size': batch_size,
                                        'hidden_dim': hidden_dimension,
                                        'num_layers': number_of_layers,
                                        'dropout': drop_out,
                                        'lr': learning_rate,
                                        'epochs': epochs_i,
                                        'class_threshold': class_thresh,
                                        'accuracy': accuracy,
                                        'precision_0': report['0']['precision'],
                                        'recall_0': report['0']['recall'],
                                        'f1_0': report['0']['f1-score'],
                                        'precision_1': report['1']['precision'],
                                        'recall_1': report['1']['recall'],
                                        'f1_1': report['1']['f1-score'],
                                    })
                                i += 1
                                if i % 10 == 0 or i == no_iterations:
                                    print(f"Completed {i}/{no_iterations} configurations.")
                                del model
                                torch.cuda.empty_cache()
    results_df = pd.DataFrame(results)
    results_df.to_csv('Results/model_performance_results.csv', index=False)
    return results_df