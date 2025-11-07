import pandas as pd
from sklearn.metrics import mean_squared_error
from time import time

from helper import get_return_Forecast_DL

NUM_LAYERS_list = [1, 5, 10]
NUMBER_OF_LAGS = 13
HIDDEN_DIM_list = [32]
EPOCHS_list = [100]
BATCH_SIZE = 64
LR_list = [0.001, 0.0001]
DROPOUT = 0.0
MIN_TRAIN_MONTHS = 240
RETRAIN_FREQUENCY = 12

data_monthly = pd.read_csv('Research/Finance/Stock_Returns/Goyal_Welch/data_monthly_DL.csv')
features = data_monthly.drop(columns=['Index', 'D12', 'E12', 'CRSP_SPvw', 'CRSP_SPvwx', 'Rfree', 'e10p', 'log_equity_premium', 'equity_premium', 'csp'])
features.set_index('date', inplace=True)
y = data_monthly['equity_premium']

model_results = {}

for NUM_LAYERS in NUM_LAYERS_list:
    for EPOCHS in EPOCHS_list:
        for LR in LR_list:
            for HIDDEN_DIM in HIDDEN_DIM_list:
                res = get_return_Forecast_DL(
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
                    rnn_model='gru'
                )

# historical mean before each prediction
res['Historical_Mean'] = data_monthly['equity_premium'][0:MIN_TRAIN_MONTHS].mean()

MSE_DL = mean_squared_error(res['Actual_Value'], res['Predicted_Value'])
MSE_M = mean_squared_error(res['Actual_Value'], res['Historical_Mean'])
R2 = 1 - MSE_DL / MSE_M

model_results[(LR, HIDDEN_DIM)] = {
    'MSE_DL': MSE_DL,
    'MSE_M': MSE_M,
    'R2': R2
}

# save results in folder
results_df = pd.DataFrame.from_dict(model_results, orient='index')
results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['LR', 'HIDDEN_DIM'])
results_df.to_csv(f'Research/Finance/Stock_Returns/Goyal_Welch/output/hyperparam_tuning_{time()}.csv')