import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import re


def get_top_flop_k(k: int, data: pd.DataFrame, unique_dates):
    
    top_stocks_by_date = {}

    for i, date in enumerate(unique_dates):
        date_data = data[data['date'] == date]
        
        # Top k for Class 0 (below median) - shuffle to randomize ties
        top_class0 = date_data.nlargest(k, 'pred_Class0', keep='all')
        if len(top_class0) > k:
            top_class0 = top_class0.sample(n=k, random_state=43+i)
        top_class0 = top_class0[['stock', 'pred_Class0', 'pred_Class1', 'return', 'predicted_class', 'actual_class']]
        
        # Top k for Class 1 (above median) - shuffle to randomize ties
        top_class1 = date_data.nlargest(k, 'pred_Class1', keep='all')
        if len(top_class1) > k:
            top_class1 = top_class1.sample(n=k, random_state=42+i)
        top_class1 = top_class1[['stock', 'pred_Class1', 'pred_Class0', 'return', 'predicted_class', 'actual_class']]
        
        top_stocks_by_date[date] = {
            'top_class0': top_class0,
            'top_class1': top_class1
        }

    return top_stocks_by_date

def create_metric_df(results_folder, K_List):
    results_folder = Path(results_folder)
    print(results_folder)
    Parquet_files = sorted(glob.glob(str(results_folder / 'predictions_*.parquet')))
    print(len(Parquet_files), "parquet files found.")
    
    for K in K_List:
        # Store all yearly metrics
        all_metrics = []
        # Iterate through each parquet file (one per year)
        for parquet_file in Parquet_files:
            # Extract year from filename (e.g., predictions_2005.parquet -> 2005)
            year = Path(parquet_file).stem.split('_')[-1]
            print(f"Processing year: {year}")
            
            # Read the parquet file
            results_df = pd.read_parquet(parquet_file)
            unique_dates = sorted(results_df['date'].unique())
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
                    'accuracy': round((accuracy_flop_k + accuracy_top_k) / 2, 2),
                    'flop_stocks': flop_stocks,
                    'top_stocks': top_stocks
                })

            metrics_df = pd.DataFrame(metrics_data)
            all_metrics.append(metrics_df)

        # Combine all yearly metrics into one DataFrame
        metrics_df = pd.concat(all_metrics, ignore_index=True)

        # Save to CSV
        metrics_df.to_csv(results_folder / f'metrics_k{K}.csv', index=False)
        print(f"\nSaved global metrics for {len(Parquet_files)} years to metrics_k{K}.csv")
        print(f"Total rows: {len(metrics_df)}")
        print("\nSummary by year:")
        print(metrics_df.groupby('year')[['avg_return_flop_k', 'avg_return_top_k', 'overall_return', 'accuracy']].mean())
    

def summarize_global_metrics(folder_path: str, k: int):
    folder = Path(folder_path)
    subfolders = [f for f in folder.glob('**/') if f.is_dir()]
    global_metric_df = pd.DataFrame()
    summary_statistics = pd.DataFrame()
    for subfolder in subfolders:
        csv_files = sorted(glob.glob(str(subfolder / f'metrics_k{k}.csv'))) # later to switched to metrics_*.csv
        for csv_file in csv_files:
            temp_df = pd.read_csv(csv_file)
            yearly_stats = temp_df.groupby('year').agg({
                'year': 'first',
                'avg_return_flop_k': 'mean',
                'avg_return_top_k': 'mean',
                'overall_return': ['mean', 'std'],
                'accuracy': 'mean'
            })
            yearly_stats.columns = ['year', 'avg_return_flop_k', 'avg_return_top_k', 'avg_overall_return', 'std_overall_return', 'avg_accuracy']
            yearly_stats = yearly_stats.reset_index(drop=True)
            yearly_stats['model_config'] = subfolder.name
            summary_stats = yearly_stats.groupby('model_config').agg({
                'avg_overall_return': 'mean',
                'std_overall_return': 'mean',
                'avg_accuracy': 'mean'
            }).reset_index()
            global_metric_df = pd.concat([global_metric_df, yearly_stats], ignore_index=True)
            summary_statistics = pd.concat([summary_statistics, summary_stats], ignore_index=True)
    return global_metric_df, summary_statistics

def accuracy_per_year(metric_df: pd.DataFrame, img_name: str):
    for config_name, group_df in metric_df.groupby('model_config'):
        #print(f'Accuracy for {config_name}: {group_df['accuracy'].to_list()}')
        plt.plot(group_df['year'], group_df['avg_accuracy'], marker='o', label=config_name)
    plt.xlabel('Year')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy per Year')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Research/Finance/Stock_Returns/Fischer/paper_material/img/{img_name}')
    plt.close()

def return_per_year(metric_df: pd.DataFrame, img_name: str):
    for config_name, group_df in metric_df.groupby('model_config'):
        #print(f'Accuracy for {config_name}: {group_df['accuracy'].to_list()}')
        plt.plot(group_df['year'], group_df['avg_overall_return'], marker='o', label=config_name)
    plt.xlabel('Year')
    plt.ylabel('Daily Return')
    plt.title('Daily Return by Year')
    plt.legend()
    plt.grid()
    plt.savefig(f'Research/Finance/Stock_Returns/Fischer/paper_material/img/{img_name}')
    plt.close()


def return_vs_layers(summary_stats: pd.DataFrame, img_name: str):
    # Extract number of layers from model_config
    summary_stats['num_layers'] = summary_stats['model_config'].apply(
        lambda x: int(re.search(r'l(\d+)', x).group(1))
    )
    
    # Sort by number of layers for better visualization
    summary_stats = summary_stats.sort_values('num_layers')
    
    plt.figure(figsize=(10, 6))
    plt.plot(summary_stats['num_layers'], summary_stats['avg_overall_return'], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Number of Layers')
    plt.ylabel('Average Overall Return')
    plt.title('Overall Return vs. Number of Layers')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'Research/Finance/Stock_Returns/Fischer/paper_material/img/{img_name}')
    plt.close()
    print(f"Saved plot to {img_name}")


folder = 'Research/Finance/Stock_Returns/Fischer/results/lstm/cov_sector'
create_df = False
if create_df:
    for sub_folder in Path(folder).glob('*'):
        # Get all parquet files from the output folder
        K_list = [10]#, 50, 100, 150, 200]
        create_metric_df(sub_folder, K_List=K_list)

#create_metric_df(f"{folder}/h_25_l_1_lr_0.001_B256_49383382", K_List=[10])
gmdf, summary_stats = summarize_global_metrics(folder, k=10)
print(summary_stats)
# accuracy_per_year(gmdf, img_name='accuracy_lstm_base.png')
# return_per_year(gmdf, img_name='return_lstm_base.png')
#return_vs_layers(summary_stats, img_name='return_vs_layers_sector.png')