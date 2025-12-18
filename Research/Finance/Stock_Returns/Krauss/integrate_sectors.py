import pandas as pd


sectors = pd.read_csv('Research/Finance/confidential_data/LSEG/Company_Sector.csv', usecols=['Instrument', 'Economic_Sector'])
print(sectors['Economic_Sector'].value_counts())
# stock to industry sector mapping
stoi = {row['Instrument']: row['Economic_Sector'] for _, row in sectors.iterrows()}

for year in range(1998,2025):
    return_data = pd.read_parquet(f'Research/Finance/Stock_Returns/Krauss/data/returns_per_period/lag_60/returns_{year}.parquet')
    # add sector information
    return_data['sector'] = return_data['stock'].map(stoi)
    # save data
    return_data.to_parquet(f'Research/Finance/Stock_Returns/Krauss/data/returns_per_period/lag_60_sector/returns_{year}.parquet')
