import pandas as pd
from pathlib import Path
import glob

# Get all CSV files from the output folder
output_folder = Path('Research/Finance/Stock_Returns/Krauss/data/predictions/pytorch_2nd')
csv_files = sorted(glob.glob(str(output_folder / 'predictions_*.csv')))
K = 150

def get_top_flop_k(k: int, data: pd.DataFrame, unique_dates):

    
    top_stocks_by_date = {}

    for date in unique_dates:
        date_data = data[data['date'] == date]
        
        # Top 10 for Class 0 (below median)
        top_class0 = date_data.nlargest(k, 'pred_Class0')[['stock', 'pred_Class0', 'pred_Class1', 'return', 'predicted_class', 'actual_class']]
        
        # Top 10 for Class 1 (above median)
        top_class1 = date_data.nlargest(k, 'pred_Class1')[['stock', 'pred_Class1', 'pred_Class0', 'return', 'predicted_class', 'actual_class']]
        
        top_stocks_by_date[date] = {
            'top_class0': top_class0,
            'top_class1': top_class1
        }

    return top_stocks_by_date

# Store all yearly metrics
all_metrics = []

# Iterate through each CSV file (one per year)
for csv_file in csv_files:
    # Extract year from filename (e.g., predictions_2005.csv -> 2005)
    year = Path(csv_file).stem.split('_')[-1]
    print(f"Processing year: {year}")
    
    # Read the CSV file
    results_df = pd.read_csv(csv_file)
    
    unique_dates = results_df['date'].unique()
    top_stocks_by_date = get_top_flop_k(K, results_df, unique_dates)

    # Calculate metrics for each date
    metrics_data = []
    for date in unique_dates:
        flop_k = top_stocks_by_date[date]['top_class0']
        avg_return_flop_k = - flop_k['return'].mean()
        accuracy_flop_k = 1 - flop_k['actual_class'].sum()/K
        flop_stocks = flop_k['stock'].tolist()

        top_k = top_stocks_by_date[date]['top_class1']
        avg_return_top_k = top_k['return'].mean()
        accuracy_top_k = top_k['actual_class'].sum()/K
        top_stocks = top_k['stock'].tolist()

        overall_return = (avg_return_flop_k + avg_return_top_k)
        
        metrics_data.append({
            'year': year,
            'date': date,
            'avg_return_flop_k': avg_return_flop_k,
            'accuracy_flop_k': accuracy_flop_k,
            'avg_return_top_k': avg_return_top_k,
            'accuracy_top_k': accuracy_top_k,
            'overall_return': overall_return,
            'flop_stocks': flop_stocks,
            'top_stocks': top_stocks
        })

    metrics_df = pd.DataFrame(metrics_data)
    all_metrics.append(metrics_df)

# Combine all yearly metrics into one DataFrame
global_metrics_df = pd.concat(all_metrics, ignore_index=True)

# Save to CSV
global_metrics_df.to_csv(output_folder / f'global_metrics_k{K}.csv', index=False)
print(f"\nSaved global metrics for {len(csv_files)} years to global_metrics_k{K}.csv")
print(f"Total rows: {len(global_metrics_df)}")
print("\nSummary by year:")
print(global_metrics_df.groupby('year')[['avg_return_flop_k', 'avg_return_top_k', 'overall_return']].mean())