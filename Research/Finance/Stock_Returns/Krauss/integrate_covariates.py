import pandas as pd
import yfinance as yf

# Download gold price data (GC=F is gold futures)
gold = yf.download('GLD', start='1990-01-01')['Close']
gold_returns = gold.pct_change()
gold_returns = gold_returns.dropna()
print(gold_returns)
exit()

oil = pd.read_csv('Research/Finance/confidential_data/Bloomberg/oil_prices.csv', index_col='date', parse_dates=['date'])
oil = oil['PX_LAST']
oil_returns = oil.pct_change(fill_method=None).rename('oil_return')
oil_returns = oil_returns.dropna()

vix = pd.read_csv('Research/Finance/confidential_data/Cboe/vix.csv', index_col='Date', parse_dates=['Date'])
vix = vix['vix']

interest_rates = pd.read_csv('Research/Finance/confidential_data/FederalReserveBoardReports/Interest_Rate_Daily.csv', index_col='date', parse_dates=['date'])
interest_rates.sort_index(inplace=True)
t103m = interest_rates['t10y3m'].dropna()
dgs2 = interest_rates['dgs10'].dropna().diff().round(2)




def __merge_covariates(df, covariate, cov_name, year, seq_length=60, standardize=True):
    cov_period = covariate.loc[f'{year-4}-12-20':f'{year}-12-31']
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

for year in range(1998,2025):
    return_data = pd.read_parquet(f'Research/Finance/Stock_Returns/Krauss/data/returns_per_period/lag_60/returns_{year}.parquet')
    data = return_data.rename(columns={'sequence': 'stock_return_sequence'})
    data = __merge_covariates(data, oil_returns, 'oil_return', year)
    data = __merge_covariates(data, vix, 'vix', year)
    data = __merge_covariates(data, t103m, 't103m', year, standardize=False)
    data = __merge_covariates(data, dgs2, 'dgs2', year, standardize=False)
    # show dates with nan values
    nan_dates = data[data.isnull().any(axis=1)]['date'].unique()
    print(f'Year {year} - Dates with NaN values after merging covariates: {nan_dates}')
    exit()


