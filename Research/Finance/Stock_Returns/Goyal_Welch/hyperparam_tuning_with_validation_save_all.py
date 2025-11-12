import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from models import ReturnLSTM, ReturnGRU


def __create_lagged_matrix(X, window):
    out = []
    for i in range(window, len(X)):
        out.append(X[i - window:i, :])
    return np.array(out)


def run_forecast_with_validation(data: pd.DataFrame, y: pd.Series, num_lags: int, num_layers: int, hidden_dim: int,
                                 epochs: int, batch_size: int, lr: float, dropout: float, minimum_train_months: int,
                                 retrain_frequency: int, rnn_model: str, val_months: int = 60):
    """
    Run expanding-window forecasts. For each training block, split the tail of the training block of length
    `val_months` as a validation set (time-aware), train on the inner-train, compute validation MSE, then
    generate predictions on the test block. Returns (results_df, mean_val_mse)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    features = data.to_numpy()
    X_lagged = __create_lagged_matrix(features, num_lags)
    y_target = y.values[num_lags:]
    dates_lagged = data.index[num_lags:]

    min_train_months = minimum_train_months
    test_block_months = retrain_frequency
    total_months = len(dates_lagged)
    window_results = []
    val_mse_list = []

    for test_start_pos in tqdm(range(min_train_months, total_months, test_block_months), desc="monthly expanding window"):
        test_end_pos = min(test_start_pos + test_block_months, total_months)
        train_idx = np.arange(0, test_start_pos)
        test_idx = np.arange(test_start_pos, test_end_pos)

        X_tr = X_lagged[train_idx]
        y_tr = y_target[train_idx]
        X_te = X_lagged[test_idx]
        y_te = y_target[test_idx]
        input_dim = X_tr.shape[2]

        # validation split from tail of training block
        if val_months is not None and val_months > 0 and X_tr.shape[0] > val_months:
            n_inner = X_tr.shape[0] - val_months
            X_tr_inner = X_tr[:n_inner]
            y_tr_inner = y_tr[:n_inner]
            X_val = X_tr[n_inner:]
            y_val = y_tr[n_inner:]
        else:
            X_tr_inner = X_tr
            y_tr_inner = y_tr
            X_val = None

        # tensors
        X_tr_t = torch.tensor(X_tr_inner, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_inner.reshape(-1, 1), dtype=torch.float32)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)

        # model
        if rnn_model == 'lstm':
            model_window = ReturnLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        elif rnn_model == 'gru':
            model_window = ReturnGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError("Model type not recognized. Use 'lstm' or 'gru'.")

        optimizer = optim.Adam(model_window.parameters(), lr=lr)
        criterion = nn.MSELoss()

        ds = TensorDataset(X_tr_t, y_tr_t)
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # train
        model_window.train()
        for epoch in range(epochs):
            for X_batch, y_batch in data_loader:
                optimizer.zero_grad()
                pred = model_window(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

        # evaluate
        model_window.eval()
        with torch.no_grad():
            y_pred = model_window(X_te_t).numpy().flatten()

            if X_val is not None:
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                val_pred = model_window(X_val_t).numpy().flatten()
                val_mse_block = float(mean_squared_error(y_val, val_pred))
                val_mse_list.append(val_mse_block)

        window_df = pd.DataFrame({
            'Actual_Value': y_te,
            'Predicted_Value': y_pred,
        }, index=dates_lagged[test_idx])

        window_results.append(window_df)

    if window_results:
        results = pd.concat(window_results).sort_index()
    else:
        results = pd.DataFrame()

    mean_val_mse = float(np.mean(val_mse_list)) if len(val_mse_list) > 0 else float('nan')

    return results, mean_val_mse


if __name__ == '__main__':
    # example grid-search script that saves results and the best hyperparameter by validation MSE
    NUM_LAYERS_list = [1]#, 5, 10]
    NUMBER_OF_LAGS = 13
    HIDDEN_DIM_list = [32]
    EPOCHS_list = [50]
    BATCH_SIZE = 64
    LR_list = [0.001, 0.0001]
    DROPOUT = 0.0
    MIN_TRAIN_MONTHS = 240
    RETRAIN_FREQUENCY = 48
    VAL_MONTHS = 60

    data_monthly = pd.read_csv('Research/Finance/Stock_Returns/Goyal_Welch/data_monthly_DL.csv')
    features = data_monthly.drop(columns=['Index', 'D12', 'E12', 'CRSP_SPvw', 'CRSP_SPvwx', 'Rfree', 'e10p', 'log_equity_premium', 'equity_premium', 'csp'])
    features.set_index('date', inplace=True)
    y = data_monthly['equity_premium']

    outdir = 'Research/Finance/Stock_Returns/Goyal_Welch/output'
    os.makedirs(outdir, exist_ok=True)

    model_results = {}
    best_val = float('inf')
    best_params = None

    ts = int(time.time())

    for NUM_LAYERS in NUM_LAYERS_list:
        for EPOCHS in EPOCHS_list:
            for LR in LR_list:
                for HIDDEN_DIM in HIDDEN_DIM_list:
                    res, val_mse = run_forecast_with_validation(
                        data=features,
                        y=y,
                        num_lags=NUMBER_OF_LAGS,
                        num_layers=NUM_LAYERS,
                        hidden_dim=HIDDEN_DIM,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        lr=LR,
                        dropout=DROPOUT,
                        minimum_train_months=MIN_TRAIN_MONTHS,
                        retrain_frequency=RETRAIN_FREQUENCY,
                        rnn_model='gru',
                        val_months=VAL_MONTHS
                    )

                    params_key = (NUM_LAYERS, EPOCHS, LR, HIDDEN_DIM)

                    if res.empty:
                        print(f"No results for combo: layers={NUM_LAYERS}, epochs={EPOCHS}, lr={LR}, hid={HIDDEN_DIM}")
                        model_results[params_key] = {
                            'MSE_DL': float('nan'),
                            'MSE_M': float('nan'),
                            'R2': float('nan'),
                            'val_MSE': val_mse
                        }
                        continue

                    hist_mean = data_monthly['equity_premium'][0:MIN_TRAIN_MONTHS].mean()
                    res['Historical_Mean'] = hist_mean

                    MSE_DL = float(mean_squared_error(res['Actual_Value'], res['Predicted_Value']))
                    MSE_M = float(mean_squared_error(res['Actual_Value'], res['Historical_Mean']))
                    R2 = 1 - MSE_DL / MSE_M

                    model_results[params_key] = {
                        'MSE_DL': MSE_DL,
                        'MSE_M': MSE_M,
                        'R2': R2,
                        'val_MSE': val_mse
                    }

                    # save per-combination OOS predictions
                    safe_lr = str(LR).replace('.', 'p')
                    combo_name = f"layers{NUM_LAYERS}_epochs{EPOCHS}_lr{safe_lr}_hid{HIDDEN_DIM}_{ts}"
                    res.to_csv(os.path.join(outdir, f"full_predictions/oos_predictions_{combo_name}.csv"))

                    if not np.isnan(val_mse) and val_mse < best_val:
                        best_val = val_mse
                        best_params = {
                            'NUM_LAYERS': NUM_LAYERS,
                            'EPOCHS': EPOCHS,
                            'LR': LR,
                            'HIDDEN_DIM': HIDDEN_DIM,
                            'val_MSE': val_mse,
                            'MSE_DL': MSE_DL,
                            'R2': R2,
                            'oos_predictions_file': f"oos_predictions_{combo_name}.csv"
                        }

    results_df = pd.DataFrame.from_dict(model_results, orient='index')
    results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['NUM_LAYERS', 'EPOCHS', 'LR', 'HIDDEN_DIM'])
    results_df.to_csv(os.path.join(outdir, f'hyperparam_tuning_results_{ts}.csv'))