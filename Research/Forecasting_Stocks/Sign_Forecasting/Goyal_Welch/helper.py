import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt

def get_statistics(ts_df, indep, dep, h=1, start=1930, end=2005, est_periods_OOS=20):
    # Subset the data for in-sample period
    ts_in_sample = ts_df.loc[start:end]

    # 1. Historical mean model
    avg = ts_in_sample[dep].mean()
    IS_error_N = ts_in_sample[dep] - avg

    # 2. OLS model with lagged independent variable
    # Create lagged independent variable
    lagged_indep = ts_in_sample[indep].shift(1)
    # Drop NA values due to lag
    valid_idx = lagged_indep.dropna().index
    y = ts_in_sample.loc[valid_idx, dep]
    X = lagged_indep.loc[valid_idx]
    X_const = add_constant(X)
    reg = OLS(y, X_const).fit()
    IS_error_A = reg.resid

    # OOS ANALYSIS
    OOS_error_N = []
    OOS_error_A = []

    for i in range(start + est_periods_OOS, end):
        # Actual ERP at time i+1
        actual_ERP = ts_df.loc[i + 1, dep]
        # Historical mean model forecast using data up to i
        avg_i = ts_df.loc[start:i, dep].mean()
        OOS_error_N.append(actual_ERP - avg_i)

        # OLS model forecast using data up to i
        ts_train = ts_df.loc[start:i]
        lagged_indep_train = ts_train[indep].shift(1)
        valid_idx_train = lagged_indep_train.dropna().index
        y_train = ts_train.loc[valid_idx_train, dep]
        X_train = lagged_indep_train.loc[valid_idx_train]
        X_train_const = add_constant(X_train)
        reg_OOS = OLS(y_train, X_train_const).fit()

        # Prepare new data for prediction (lagged independent variable at time i)
        x_new = ts_df.loc[i, indep]
        x_new_df = pd.DataFrame({indep: [x_new]})
        x_new_const = add_constant(x_new_df, has_constant='add')
        pred_ERP = reg_OOS.predict(x_new_const)[0]

        OOS_error_A.append(pred_ERP - actual_ERP)

    OOS_error_N = np.array(OOS_error_N)
    OOS_error_A = np.array(OOS_error_A)

    # Compute statistics
    MSE_N = np.mean(OOS_error_N ** 2)
    MSE_A = np.mean(OOS_error_A ** 2)
    T = ts_df[dep].notna().sum()
    OOS_R2 = 1 - MSE_A / MSE_N
    OOS_oR2 = OOS_R2 - (1 - OOS_R2) * (reg.df_resid) / (T - 1)
    dRMSE = np.sqrt(MSE_N) - np.sqrt(MSE_A)

    # CREATE PLOT DATA
    IS_sq_diff = np.cumsum(IS_error_N.iloc[1:] ** 2) - np.cumsum(IS_error_A ** 2)
    OOS_sq_diff = np.cumsum(OOS_error_N ** 2) - np.cumsum(OOS_error_A ** 2)

    # Align IS series to OOS start
    IS_plot = IS_sq_diff[(1 + est_periods_OOS):]
    years = np.arange(start + 1 + est_periods_OOS, end + 1)

    # Shift IS errors vertically so IS line begins at zero at first OOS prediction
    IS_plot = IS_plot - IS_plot.iloc[0]

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(years[1:], IS_plot, label='IS')
    plt.plot(years, OOS_sq_diff, label='OOS')
    plt.axvspan(1973, 1975, color='red', alpha=0.1)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('Year')
    plt.ylabel('Cumulative SSE Difference')
    plt.legend()
    plt.tight_layout()
    plotGG = plt

    return {
        'IS_error_N': IS_error_N,
        'IS_error_A': IS_error_A,
        'OOS_error_N': OOS_error_N,
        'OOS_error_A': OOS_error_A,
        'IS_R2': reg.rsquared,
        'IS_aR2': reg.rsquared_adj,
        'OOS_R2': OOS_R2,
        'OOS_oR2': OOS_oR2,
        'dRMSE': dRMSE,
        'plotGG': plotGG
    }