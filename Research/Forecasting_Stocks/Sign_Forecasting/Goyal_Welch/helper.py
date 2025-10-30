import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt


def __get_IS_R2(ts_df, indep, dep, start=1930, end=2005):
    # Subset the data for in-sample period
    ts_in_sample = ts_df.loc[start:end]

    # OLS model with lagged independent variable
    lagged_indep = ts_in_sample[indep].shift(1)
    valid_idx = lagged_indep.dropna().index
    y = ts_in_sample.loc[valid_idx, dep]
    X = lagged_indep.loc[valid_idx]
    X_const = add_constant(X)
    reg = OLS(y, X_const).fit()
    
    IS_R2 = reg.rsquared
    IS_R2_head = IS_R2 - (1 - IS_R2)*(len(y)-int(reg.df_resid))/(len(y)-1)

    return IS_R2_head, reg.resid

def get_statistics(ts_df, indep, dep, h=1, start=1927, end=2005, est_periods_OOS=20, plot='no'):
    # Subset the data for in-sample period
    ts_in_sample = ts_df.loc[start:end]

    # 1. Historical mean model
    avg = ts_in_sample[dep].mean()
    IS_error_N = ts_in_sample[dep] - avg

    # 2. OLS model with lagged independent variable
    IS_R2_head, IS_error_A = __get_IS_R2(ts_df, indep, dep, start, end)

    # 3. OLS model for IS statistics for OOS period
    IS_R2_head_OOS, _ = __get_IS_R2(ts_df, indep, dep, start + est_periods_OOS, end)

    # 4. OLS model for IS statistics for period beginning in 1965
    IS_R2_head_1965, _ = __get_IS_R2(ts_df, indep, dep, start=1965, end=end)

    # OOS ANALYSIS
    OOS_error_N = []
    OOS_error_A = []

    for i in range(start + est_periods_OOS, end):
        # Actual Equity Risk Premium (ERP) at time i+1
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
    k = T - 1
    OOS_R2 = 1 - MSE_A / MSE_N
    OOS_oR2 = OOS_R2 - (1 - OOS_R2) * (T-k) / (T - 1)
    dRMSE = np.sqrt(MSE_N) - np.sqrt(MSE_A)
    MSEf = (T-h+1)*(MSE_N - MSE_A)/MSE_A

    if plot == 'yes':
        # CREATE PLOT DATA
        IS_sq_diff = np.cumsum(IS_error_N.iloc[1:] ** 2) - np.cumsum(IS_error_A ** 2)
        OOS_sq_diff = np.cumsum(OOS_error_N ** 2) - np.cumsum(OOS_error_A ** 2)

        # Align IS series to OOS start
        IS_plot = IS_sq_diff[(1 + est_periods_OOS):]
        years = np.arange(start + 1 + est_periods_OOS, end + 1)

        # Shift IS errors vertically so IS line begins at zero at first OOS prediction
        IS_plot = IS_plot - IS_plot.iloc[0]

        # Plotting
        plt.figure(figsize=(10, 3))
        plt.plot(years[1:], IS_plot, label='IS')
        plt.plot(years, OOS_sq_diff, label='OOS')
        plt.axvspan(1973, 1975, color='red', alpha=0.1)
        plt.xlabel('Year')
        plt.ylabel('Cumulative SSE Difference')
        plt.legend()
        plt.tight_layout()

    return {
        'IS_R2_head_1927': round(float(IS_R2_head)*100, 2),
        'IS_R2_head_OOS': round(float(IS_R2_head_OOS)*100, 2),
        'IS_R2_head_1965': round(float(IS_R2_head_1965)*100, 2),
        'OOS_oR2': round(float(OOS_oR2)*100, 2),
        'dRMSE': round(float(dRMSE)*100, 2),
        'MSEf': round(float(MSEf), 2)
    }