import pandas as pd

oil = pd.read_csv('Research/Finance/confidential_data/Bloomberg/oil_prices.csv', index_col='date', parse_dates=['date'])
oil = oil['PX_LAST']
oil_returns = oil.pct_change(fill_method=None).rename('oil_return')
oil_returns = oil_returns.dropna()

vix = pd.read_csv('Research/Finance/confidential_data/Cboe/vix.csv', index_col='date', parse_dates=['date'])
vix = vix['vix']

gold = pd.read_csv('Research/Finance/confidential_data/LSEG/gold_futures_GCc1.xls', index_col='Date', parse_dates=['Date'])
gold = gold['SETTLE']
gold_returns = gold.pct_change(fill_method=None).rename('gold_return')
gold_returns = gold_returns.dropna()

interest_rates = pd.read_csv('Research/Finance/confidential_data/FederalReserveBoardReports/Interest_Rate_Daily.csv', index_col='date', parse_dates=['date'])
interest_rates.sort_index(inplace=True)
t103m = interest_rates['t10y3m'].dropna()
dgs2 = interest_rates['dgs10'].dropna().diff().round(2)


def __merge_covariates(df, covariate, cov_name, year, seq_length=60, standardize=True, dates=None):
    cov_period = covariate.loc[f'{year-4}-11-01':f'{year}-12-31']
    if cov_period.isna().any():
        print(f"WARNING: {cov_name} has {cov_period.isna().sum()} missing values in the period {year-4}-12-20 to {year}-12-31. These will be interpolated.")
        cov_period = cov_period.interpolate()
    # Check if required dates exist in covariate data
    if dates is not None:
        missing_dates = [d for d in dates if d not in cov_period.index]
        if missing_dates:
            print(f"WARNING: {cov_name} missing {len(missing_dates)} dates which will be interpolated.")
            # Fill missing dates with interpolation
            cov_period = cov_period.reindex(cov_period.index.union(pd.DatetimeIndex(missing_dates)))
            cov_period = cov_period.sort_index().interpolate()
    
    # Standardize based on training data
    train_cov = covariate.loc[f'{year-3}-01-01':f'{year-1}-12-31']
    if standardize:
        mean_cov = train_cov.mean()
        std_dev_cov = train_cov.std()
        cov_period = (cov_period - mean_cov) / std_dev_cov

    # Create 60 sequences of past cov returns
    cov_return_sequences = []
    sequence_length = seq_length
    for i in range(sequence_length, len(cov_period)):
        cov_return_sequences.append(cov_period.iloc[i-sequence_length:i].values)
    cov_return_sequences = pd.DataFrame({'date': cov_period.index[sequence_length:], f'{cov_name}_sequence': cov_return_sequences})
    df = df.merge(cov_return_sequences, on='date', how='left')

    return df

for year in range(2017,2025):
    return_data = pd.read_parquet(f'Research/Finance/Stock_Returns/Krauss/data/returns_per_period/lag_60/returns_{year}.parquet')
    # get dates from return data, which should also exist in the covariates data
    dates = return_data['date'].unique().tolist()
    data = return_data.rename(columns={'sequence': 'stock_return_sequence'})
    data = __merge_covariates(data, oil_returns, 'oil_return', year, standardize=True, dates=dates)
    data = __merge_covariates(data, vix, 'vix', year, standardize=True, dates=dates)
    data = __merge_covariates(data, t103m, 't103m', year, standardize=True, dates=dates)
    data = __merge_covariates(data, dgs2, 'dgs2', year, standardize=True, dates=dates)
    data = __merge_covariates(data, gold_returns, 'gold_return', year, standardize=True, dates=dates)
    # show dates with nan values
    nan_dates = data[data.isnull().any(axis=1)]['date'].unique()
    print(f'Year {year} - Dates with NaN values after merging covariates: {len(nan_dates)}. \n', 50*'-')
    data.to_parquet(f'Research/Finance/Stock_Returns/Krauss/data/returns_per_period/lag_60_cov/returns_{year}.parquet')


