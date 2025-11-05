import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.metrics import accuracy_score, classification_report

# seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def create_lagged_matrix_multifeature(X, window):
    """Create a matrix with dimension (samples, window, features). 
    That is, for every time t, we have a matrix of the previous #window
    observations for all features.
    :param X: 2D numpy array of shape (time_steps, features)
    :param window: number of lags
    """
    out = []
    for i in range(window, len(X)):
        out.append(X[i-window:i, :])
    return np.array(out)


def prepare_data(input_data: pd.DataFrame, number_of_lags: int):
    """Prepare lagged feature matrix and target vector.
    :param input_data: DataFrame with features and target
    :param number_of_lags: number of lags
    """
    # Feature matrix
    features = np.column_stack([
        input_data['Realized_Vol_5d'].values,
        input_data['Realized_Vol_20d'].values,
        input_data['Return_Lag1'].values,
        input_data['Volume_change'].values,
        input_data['Trading_range'].values,
        input_data['ten_y_tnotes'].values,
        input_data['three_months_tbills'].values,
        input_data['oil_price_return'].values,
        input_data['vix_index'].values,
        input_data['vix_return'].values
    ])
    X_lagged = create_lagged_matrix_multifeature(features, number_of_lags)
    y_target = input_data['Sign_1d'].values[number_of_lags:]

    return X_lagged, y_target


def get_valid_years(input_data: pd.DataFrame, number_of_lags: int):
    """Determine valid test years based on data availability.
    :param input_data: DataFrame with features and target
    :param number_of_lags: number of lags
    """
    dates_lagged = input_data.index[number_of_lags:]
    # Convert to DatetimeIndex if not already
    if not isinstance(dates_lagged, pd.DatetimeIndex):
        dates_lagged = pd.to_datetime(dates_lagged)
    years = dates_lagged.year
    unique_years = np.unique(years).astype(int)
    min_year = int(unique_years.min())
    max_year = int(unique_years.max())

    # require at least this many trading observations for a year to be considered "full"
    min_days_per_year = 200

    # find first test_year such that test_year and previous 3 years each have >= min_days_per_year samples
    first_valid_test_year = None
    for cand in range(min_year + 3, max_year + 1):
        train_years = [cand - 3, cand - 2, cand - 1]
        counts = {y: np.sum(years == y) for y in train_years + [cand]}
        if all(counts[y] >= min_days_per_year for y in train_years + [cand]):
            first_valid_test_year = cand
            print("First valid test year:", first_valid_test_year)
            break

    if first_valid_test_year is None:
        raise RuntimeError("No calendar-aligned test year found with sufficient data. Lower min_days_per_year or check your date range.")
    
    return first_valid_test_year, max_year, dates_lagged, years


def split_train_val_test(X_lagged, y_target, train_idx, test_idx, val_split=0.15):
    """Split data into train, validation, and test sets.
    :param X_lagged: Lagged feature matrix
    :param y_target: Target vector
    :param train_idx: Indices for training data
    :param test_idx: Indices for test data
    :param val_split: Proportion of training data to use for validation
    """
    X_tr = X_lagged[train_idx]
    y_tr = y_target[train_idx]
    X_te = X_lagged[test_idx]
    y_te = y_target[test_idx]
    
    val_size = int(len(X_tr) * val_split)
    X_train, X_val = X_tr[:-val_size], X_tr[-val_size:]
    y_train, y_val = y_tr[:-val_size], y_tr[-val_size:]
    
    return X_train, X_val, X_te, y_train, y_val, y_te


def create_tensors(X_train, X_val, X_te, y_train, y_val, device):
    """Convert numpy arrays to PyTorch tensors.
    :param X_train: Training features
    :param X_val: Validation features
    :param X_te: Test features
    :param y_train: Training targets
    :param y_val: Validation targets
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    return X_train_t, y_train_t, X_val_t, y_val_t, X_te_t


def train_single_model(model, X_train_t, y_train_t, X_val_t, y_val_t, 
                       lr, batch_size, epochs, patience=5, device=None):
    """Train a single model with early stopping.
    :param model: PyTorch model
    :param X_train_t: Training features tensor
    :param y_train_t: Training targets tensor
    :param X_val_t: Validation features tensor
    :param y_val_t: Validation targets tensor
    :param lr: Learning rate
    :param batch_size: Batch size
    :param epochs: Number of epochs
    :param patience: Patience for early stopping
    :param device: Device to run the model on
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ds_train = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
        model.train()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_model(model, X_te_t, y_te, class_threshold):
    """Evaluate model and return predictions and metrics.
    :param model: Trained PyTorch model
    :param X_te_t: Test features tensor
    :param y_te: Test targets numpy array
    :param class_threshold: Classification threshold for when to assign logits a 1
    """
    model.eval()
    with torch.no_grad():
        logits_test = model(X_te_t).squeeze()
        probs = torch.sigmoid(logits_test).cpu().numpy()
        preds = (probs >= class_threshold).astype(int)
    
    accuracy = accuracy_score(y_te, preds)
    report = classification_report(y_te, preds, output_dict=True, zero_division=0)
    
    return accuracy, report


def train_window(X_lagged, y_target, dates_lagged, years, test_year, 
                 model_class, input_dim, hidden_dim, num_layers, dropout,
                 lr, batch_size, epochs, class_threshold, val_split=0.15, device=None):
    """Train and evaluate model for a single rolling window."""
    train_years = [test_year - 3, test_year - 2, test_year - 1]
    train_idx = np.where(np.isin(years, train_years))[0]
    test_idx = np.where(years == test_year)[0]
    
    if len(train_idx) < 700 or len(test_idx) < 230:
        return None
    
    # Split data
    X_train, X_val, X_te, y_train, y_val, y_te = split_train_val_test(
        X_lagged, y_target, train_idx, test_idx, val_split
    )
    
    # Create tensors
    X_train_t, y_train_t, X_val_t, y_val_t, X_te_t = create_tensors(
        X_train, X_val, X_te, y_train, y_val, device
    )
    
    # Initialize model
    model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, 
                       num_layers=num_layers, dropout=dropout)
    
    # Train
    model = train_single_model(model, X_train_t, y_train_t, X_val_t, y_val_t,
                               lr, batch_size, epochs, device=device)
    
    # Evaluate
    accuracy, report = evaluate_model(model, X_te_t, y_te, class_threshold)
    
    # Package results
    result = {
        'test_year': test_year,
        'train_start': dates_lagged[train_idx[0]].date(),
        'train_end': dates_lagged[train_idx[-int(len(X_train)*val_split)-1]].date(),
        'test_start': dates_lagged[test_idx[0]].date(),
        'test_end': dates_lagged[test_idx[-1]].date(),
        'accuracy': accuracy,
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'f1_0': report['0']['f1-score'],
        'precision_1': report['1']['precision'],
        'recall_1': report['1']['recall'],
        'f1_1': report['1']['f1-score'],
    }
    
    del model
    torch.cuda.empty_cache()
    
    return result


def run_sign_model(input_data: pd.DataFrame, model_class, number_of_lags: list[int],
                   hidden_dim: list[int], num_layers: list[int], dropout: list[float],
                   lr: list[float], epochs: list[int], class_threshold: list[float],
                   batch_size: int=64, val_split: float = 0.15, device=None) -> pd.DataFrame:
    """Run grid search for sign prediction model with rolling windows.
    
    :param input_data: DataFrame with features and target
    :param model_class: Model class (e.g., SimpleLSTM or SimpleGRU)
    :param number_of_lags: List of lag values to try
    :param hidden_dim: List of hidden dimension sizes
    :param num_layers: List of number of layers
    :param dropout: List of dropout rates
    :param lr: List of learning rates
    :param epochs: List of epoch counts
    :param class_threshold: List of classification thresholds
    :param batch_size: Batch size for training
    :param val_split: Validation split ratio
    :param device: Device to run the model on (e.g., 'cuda' or 'cpu')
    :return: DataFrame with results
    """
    print(f"Using device: {device}")
    
    # Calculate total iterations
    total_configs = (len(number_of_lags) * len(hidden_dim) * len(num_layers) * 
                    len(dropout) * len(lr) * len(epochs) * len(class_threshold))
    
    config_no = 0
    all_configs = []

    # Grid search over hyperparameters
    for n_lags in number_of_lags:
        X_lagged, y_target = prepare_data(input_data, n_lags)
        first_valid_year, max_year, dates_lagged, years = get_valid_years(input_data, n_lags)
        input_dim = X_lagged.shape[2]
        
        for h_dim in hidden_dim:
            for n_layers in num_layers:
                for drop in dropout:
                    for learning_rate in lr:
                        for ep in epochs:
                            for thresh in class_threshold:
                                config_no += 1

                                # Store results for this configuration across all test years
                                config_results = []
                                
                                # Train for each rolling window
                                for test_year in range(first_valid_year, max_year + 1):
                                    result = train_window(
                                        X_lagged, y_target, dates_lagged, years, test_year,
                                        model_class, input_dim, h_dim, n_layers, drop,
                                        learning_rate, batch_size, ep, thresh, val_split, device=device
                                    )
                                    
                                    if result is not None:
                                        config_results.append(result)

                                # Aggregate metrics across all test years for this configuration
                                if config_results:
                                    avg_config = {
                                        'num_lags': n_lags,
                                        'hidden_dim': h_dim,
                                        'num_layers': n_layers,
                                        'dropout': drop,
                                        'lr': learning_rate,
                                        'epochs': ep,
                                        'class_threshold': thresh,
                                        'batch_size': batch_size,
                                        'num_test_years': len(config_results),
                                        'avg_accuracy': np.mean([r['accuracy'] for r in config_results]),
                                        'std_accuracy': np.std([r['accuracy'] for r in config_results]),
                                        'avg_precision_0': np.mean([r['precision_0'] for r in config_results]),
                                        'avg_recall_0': np.mean([r['recall_0'] for r in config_results]),
                                        'avg_f1_0': np.mean([r['f1_0'] for r in config_results]),
                                        'avg_precision_1': np.mean([r['precision_1'] for r in config_results]),
                                        'avg_recall_1': np.mean([r['recall_1'] for r in config_results]),
                                        'avg_f1_1': np.mean([r['f1_1'] for r in config_results]),
                                    }
                                    all_configs.append(avg_config)        

                                if config_no % 10 == 0 or config_no == total_configs:
                                    print(f"Completed {config_no}/{total_configs} configurations.")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_configs)
    results_df = results_df.sort_values('avg_accuracy', ascending=False)
    results_df.to_csv('Results/model_performance_results.csv', index=False)
    return results_df