import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt


def __get_IS_R2(ts_df, indep, dep, start=1930, end=2005):
    # Subset the data for in-sample period
    ts_in_sample = ts_df.loc[start:end]

    # OLS model with lagged independent variable
    if indep == 'infl' and type(start) != int:
        # special treatment for 'infl' as we wait 1 month to get inflation data (but only for monthly regression)
        lagged_indep = ts_in_sample[indep].shift(2)
    else:
        lagged_indep = ts_in_sample[indep].shift(1)
    valid_idx = lagged_indep.dropna().index
    y = ts_in_sample.loc[valid_idx, dep]
    X = lagged_indep.loc[valid_idx]
    X_const = add_constant(X)
    reg = OLS(y, X_const).fit()
    # get adjusted r2
    IS_R2 = reg.rsquared
    IS_R2_head = IS_R2 - (1 - IS_R2)*(len(y)-int(reg.df_resid)-1)/(len(y)-1)

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


def OOS_analysis_monthly(ts_df, indep, dep, start, end, est_periods_OOS=240, expected_sign=None):
    idx = ts_df.index
    start_pos = idx.searchsorted(start, side='left')
    end_pos = idx.searchsorted(end, side='right') - 1

    OOS_error_M = []
    OOS_error_C = []
    OOS_error_TU = []
    num_truncated = 0
    num_slope_zero = 0

    for pos in range(start_pos + est_periods_OOS, end_pos):
        # Actual Equity Risk Premium (ERP) at time pos+1
        actual_ERP = ts_df.iloc[pos + 1][dep]

        # Historical mean forecast using data up to pos (positional slice via iloc)
        avg_i = ts_df.iloc[start_pos:pos + 1][dep].mean()
        OOS_error_M.append(actual_ERP - avg_i)

        # OLS model forecast using data up to pos (use iloc for positional slicing)
        ts_train = ts_df.iloc[start_pos:pos + 1]
        if indep == 'infl':
            # special treatment for 'infl' as we wait 1 month to get inflation data
            lagged_indep_train = ts_train[indep].shift(2)
        else:
            lagged_indep_train = ts_train[indep].shift(1)
        valid_idx_train = lagged_indep_train.dropna().index
        y_train = ts_train.loc[valid_idx_train, dep]
        X_train = lagged_indep_train.loc[valid_idx_train]
        X_train_const = add_constant(X_train)
        reg_OOS = OLS(y_train, X_train_const).fit()

        # Prepare new data for prediction (lagged independent variable at time pos)
        x_new = ts_df.iloc[pos][indep]
        x_new_df = pd.DataFrame({indep: [x_new]})
        x_new_const = add_constant(x_new_df, has_constant='add')

        # raw prediction
        pred_ERP = reg_OOS.predict(x_new_const)[0]
        OOS_error_C.append(pred_ERP - actual_ERP)

        if pred_ERP < 0:
            num_truncated += 1

        # Set coefficient to 0, if sign does not match expected sign
        if (reg_OOS.params[indep] * expected_sign) < 0:
            reg_OOS.params[indep] = 0.0
            num_slope_zero += 1

            # make prediction with slope set to zero
            pred_ERP_slope_zero = reg_OOS.predict(x_new_const)[0]

            # truncated prediction to not allow negative predicted equity premium
            pred_ERP_trunc = max(pred_ERP_slope_zero, 0)
            OOS_error_TU.append(pred_ERP_trunc - actual_ERP)

        else:
            if pred_ERP < 0:
                pred_ERP_trunc = max(pred_ERP, 0)
                OOS_error_TU.append(pred_ERP_trunc - actual_ERP)
            else:
                OOS_error_TU.append(pred_ERP - actual_ERP)
    
    OOS_error_TU = np.array(OOS_error_TU)
    OOS_error_M = np.array(OOS_error_M)

    share_truncated = num_truncated/len(OOS_error_M)
    share_slope_zero = num_slope_zero/len(OOS_error_M)

    MSE_M = np.mean(np.array(OOS_error_M)**2)
    MSE_C = np.mean(np.array(OOS_error_C)**2)
    MSE_TU = np.mean(np.array(OOS_error_TU)**2)

    OOS_R2 = 1 - (MSE_C / MSE_M)
    T = len(valid_idx_train)
    k = T - 1
    OOS_R2_head = OOS_R2 - (1-OOS_R2)*(T-k)/(T-1)

    OOS_R2_T = 1 - (MSE_TU / MSE_M)
    OOS_R2_head_T = OOS_R2_T - (1-OOS_R2_T)*(T-k)/(T-1)

    dRMSE = np.sqrt(MSE_M) - np.sqrt(MSE_TU)

    return OOS_R2_head, share_truncated, share_slope_zero, OOS_R2_head_T, dRMSE, OOS_error_M, OOS_error_TU


def get_monthly_statistics(ts_df, indep, start, end, est_periods_OOS=240, plot='no'):
    dep = 'equity_premium'
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    ts_in_sample = ts_df.loc[start_ts:end_ts].copy()

    # 1. Historical mean model
    avg = ts_in_sample[dep].mean()
    IS_error_M = ts_in_sample[dep] - avg

    # 2. IS OLS model with lagged independent variable, log equity premium
    IS_R2_head_log, IS_error_cond_log = __get_IS_R2(ts_df, indep, 'log_equity_premium', start_ts, end_ts)

    # 3. IS OLS model with lagged independent variable, equity premium
    if indep == 'infl':
        # special treatment for 'infl' as we wait 1 month to get inflation data
        lagged_indep = ts_in_sample[indep].shift(2)
    else:
        lagged_indep = ts_in_sample[indep].shift(1)
    # Drop NA values due to lag
    valid_idx = lagged_indep.dropna().index
    y = ts_in_sample.loc[valid_idx, dep]
    X = lagged_indep.loc[valid_idx]
    X_const = add_constant(X)
    reg = OLS(y, X_const).fit()
    EXPECTED_SIGN = np.where(float(reg.params[indep])>0, 1, -1)
    IS_error_cond = reg.resid
    IS_R2 = reg.rsquared
    T = len(IS_error_cond)
    k = int(reg.df_resid) + 1
    IS_R2_head = IS_R2 - (1 - IS_R2)*(T-k)/(T-1)

    # 4. IS Truncated model
    # raw in-sample predictions
    preds_insample = reg.predict(X_const)
    preds_insample_trunc = np.maximum(preds_insample, 0.0)
    y_insample = y
    # SSE for truncated model and null (full-sample mean) model
    SSE_trunc = np.sum((y_insample - preds_insample_trunc) ** 2)
    SSE_null = np.sum((y_insample - y_insample.mean()) ** 2)

    # in-sample R^2 (truncated forecasts)
    IS_R2_trunc = 1.0 - SSE_trunc / SSE_null

    # small-sample bias adjustment
    T = len(y_insample)
    k = int(reg.df_resid) + 1
    IS_R2_head_trunc = IS_R2_trunc - (1 - IS_R2_trunc) * (T - k) / (T - 1)

    # 5. OOS ANALYSIS
    OOS_R2_head, share_truncated, share_slope_zero, OOS_R2_head_T, dRMSE, OOS_error_M, OOS_error_TU = OOS_analysis_monthly(
        ts_df, indep, dep, start_ts, end_ts, est_periods_OOS, expected_sign=EXPECTED_SIGN
    )

    if plot == 'yes':
        # CREATE PLOT DATA with datetime indices
        IS_sq_diff = pd.Series(
            np.cumsum(IS_error_M.iloc[1:].values ** 2) - np.cumsum(IS_error_cond.values ** 2),
            index=IS_error_M.iloc[1:].index
        )
        idx = ts_df.index
        start_pos = idx.searchsorted(start, side='left')
        end_pos = idx.searchsorted(end, side='right') - 1

        # OOS_sq_diff: index from the OOS prediction dates
        oos_dates = ts_df.index[start_pos + est_periods_OOS + 1 : end_pos + 1]
        OOS_sq_diff = pd.Series(
            np.cumsum(OOS_error_M ** 2) - np.cumsum(OOS_error_TU ** 2),
            index=oos_dates
        )

        # IS_plot: full IS series, no vertical shift
        first_oos_date = oos_dates[0]
        IS_plot = IS_sq_diff - IS_sq_diff.loc[first_oos_date]

        # Plotting
        plt.figure(figsize=(10, 3))
        plt.plot(IS_plot.index, IS_plot.values, label='IS')
        plt.plot(OOS_sq_diff.index, OOS_sq_diff.values, label='OOS')
        plt.axvspan(pd.to_datetime('1973-01-01'), pd.to_datetime('1975-12-31'), color='red', alpha=0.1)
        plt.axvline(x=first_oos_date, color='black', linestyle='--', label='Start OOS')
        plt.xlabel('Date')
        plt.ylabel('Cumulative SSE Difference')
        plt.legend()
        plt.tight_layout()

    return {
        'IS_R2_head_log': round(float(IS_R2_head_log)*100, 2),
        'IS_R2_head': round(float(IS_R2_head)*100, 2),
        'IS_R2_head_trunc': round(float(IS_R2_head_trunc)*100, 2),
        'OOS_R2_head': round(float(OOS_R2_head)*100, 2),
        'share_T': round(float(share_truncated)*100, 2),
        'share_U': round(float(share_slope_zero)*100, 2),
        'OOS_R2_head_trunc': round(float(OOS_R2_head_T)*100, 2),
        'dRMSE': round(float(dRMSE)*100, 4)
    }