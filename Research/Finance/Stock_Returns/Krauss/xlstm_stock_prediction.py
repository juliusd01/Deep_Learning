from datetime import datetime
import random
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from pathlib import Path

# Import xLSTM components
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

seed = 123

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# get returns for all stocks for a specific date (day)
def get_returns_for_date(returns_df, target_date):
    target_date = pd.to_datetime(target_date)
    if target_date not in returns_df.index:
        raise ValueError(f"No data available for the date: {target_date}")
    else:
        return returns_df.loc[target_date]

# get csv file
data = pd.read_csv('Research/Finance/Stock_Returns/Krauss/data/study_periods/test_1998.csv')
year = '1998'
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

missing_value_stocks = data.columns[data.isna().sum(axis=0) > 0]
for stock in missing_value_stocks:
    print(f"{stock}: {data[stock].isna().sum()} missing values")
    if data[stock].isna().sum() == len(data):
        data.drop(columns=stock, inplace=True)
        print('Deleted column:', stock)

returns_data = data.pct_change(fill_method=None)
returns_data = returns_data[1:]

start_date = returns_data.index.min().year
end_date = start_date + 2
start_date = f'{start_date}-01-01'
end_date = f'{end_date}-12-31'

returns_3years = returns_data[f'{start_date}':f'{end_date}']
mean_return = returns_3years.mean().mean()
mean_volatility = returns_3years.std().mean()
returns_data = (returns_data - mean_return) / mean_volatility

stacked_returns = returns_data.T.stack().reset_index()
stacked_returns.columns = ['stock', 'date', 'return']

window = 240

sequences = []
for stock, group in stacked_returns.groupby('stock'):
    vals = group['return'].values
    for i in range(window, len(vals)):
        seq = vals[i-window:i]
        sequences.append({
            'stock': stock,
            'date': group['date'].iloc[i],
            'return': vals[i],
            'sequence': seq
        })

sequences_df = pd.DataFrame(sequences)

end = (datetime.strptime(end_date, '%Y-%m-%d') + relativedelta(years=1)).strftime('%Y-%m-%d')

valid_days = returns_data.loc[
    (returns_data.index >= start_date) & 
    (returns_data.index <= end)
].index

sequences_df['Class0'] = 0
sequences_df['Class1'] = 0

for date in valid_days:
    current = date.strftime('%Y-%m-%d')
    returns_on_date = get_returns_for_date(returns_data, current)
    median_return = returns_on_date.median()

    below_median = returns_on_date[returns_on_date < median_return].index
    above_median = returns_on_date[returns_on_date >= median_return].index

    mask = sequences_df['date'] == current

    sequences_df.loc[
        mask & sequences_df['stock'].isin(below_median),
        'Class0'
    ] = 1

    sequences_df.loc[
        mask & sequences_df['stock'].isin(above_median),
        'Class1'
    ] = 1

# ===========================
# PREPARE TENSORS
# ===========================
X = np.array(sequences_df['sequence'].tolist())
# For xLSTM: shape should be (batch, seq_len, embedding_dim)
# We have (batch, seq_len), so we need to add a dimension
X = X.reshape(X.shape[0], 240, 1)
y = sequences_df[['Class0', 'Class1']].values

# Split based on year: specified year for test, earlier years for train/val
test_mask = sequences_df['date'].dt.year == int(year)
X_test = X[test_mask.values]
y_test = y[test_mask.values]
X_train_val = X[~test_mask.values]
y_train_val = y[~test_mask.values]

# Keep the metadata for test set (stock, date, return)
test_metadata = sequences_df[test_mask][['stock', 'date', 'return']].reset_index(drop=True)

# Split train_val into train and val (80% train, 20% val of the train_val set)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


# Build the xLSTM model
class xLSTMStockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=2, num_blocks=1):
        """
        xLSTM-based model for stock price prediction.
        
        Args:
            input_size: Input feature dimension (1 for single time series)
            hidden_size: Hidden dimension / embedding dimension for xLSTM
            num_classes: Number of output classes (2 for binary classification)
            num_blocks: Number of xLSTM blocks (alternating mLSTM and sLSTM)
        """
        super(xLSTMStockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Input projection to match embedding dimension
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Configure xLSTM
        # Use CPU backend if CUDA is not available
        backend = "cuda" if torch.cuda.is_available() else "vanilla"
        
        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=240,  # Our sequence length
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            slstm_at=[1, 3] if num_blocks >= 4 else [1],  # Alternate between mLSTM and sLSTM
        )
        
        # Create xLSTM block stack
        self.xlstm_stack = xLSTMBlockStack(xlstm_cfg)
        
        # Output layers
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Output probabilities of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to embedding dimension
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_size)
        
        # Pass through xLSTM stack
        x = self.xlstm_stack(x)  # (batch_size, seq_len, hidden_size)
        
        # Use the last time step for classification
        x = x[:, -1, :]  # (batch_size, hidden_size)
        
        # Classification head
        out = self.fc(x)
        return out


# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = xLSTMStockPredictor(input_size=1, hidden_size=32, num_classes=2, num_blocks=2).to(device)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTotal parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
g = torch.Generator()
g.manual_seed(seed)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, generator=g)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0
max_epochs = 100

# print('\n' + '='*50)
# print('Starting training...')
# print('='*50)

# for epoch in range(max_epochs):
#     # Training phase
#     model.train()
#     train_loss = 0
#     train_correct = 0
#     train_total = 0
    
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         _, labels = torch.max(batch_y, 1)
#         train_total += labels.size(0)
#         train_correct += (predicted == labels).sum().item()
    
#     # Validation phase
#     model.eval()
#     val_loss = 0
#     val_correct = 0
#     val_total = 0
    
#     with torch.no_grad():
#         for batch_X, batch_y in val_loader:
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
            
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             _, labels = torch.max(batch_y, 1)
#             val_total += labels.size(0)
#             val_correct += (predicted == labels).sum().item()
    
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
#     train_acc = train_correct / train_total
#     val_acc = val_correct / val_total
    
#     print(f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
#     # Early stopping
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         # Save best model
#         Path('Research/Finance/Stock_Returns/Krauss/models').mkdir(parents=True, exist_ok=True)
#         torch.save(model.state_dict(), f'Research/Finance/Stock_Returns/Krauss/models/best_xlstm_model_{year}.pth')
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print(f'Early stopping at epoch {epoch+1}')
#             break

# Load best model
model.load_state_dict(torch.load(f'Research/Finance/Stock_Returns/Krauss/models/best_xlstm_model_{year}.pth'))
print('\n' + '='*50)
print('Training complete!')
print('='*50)

# Evaluate on test set
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

print('\n----------------\nTest set shapes:')
print('X_test:', X_test_tensor.shape)
print('y_test:', y_test_tensor.shape)

model.eval()
predictions = []
with torch.no_grad():
    for i in range(0, len(X_test_tensor), 256):   # batch size
        batch = X_test_tensor[i:i+256]
        pred = model(batch)
        predictions.append(pred.cpu())
        
    predictions = torch.cat(predictions)
    # Calculate test metrics
    _, predicted_labels = torch.max(torch.FloatTensor(predictions), 1)
    _, actual_labels = torch.max(y_test_tensor, 1)
    test_accuracy = (predicted_labels == actual_labels.cpu()).sum().item() / len(actual_labels)
    
    test_loss = criterion(torch.FloatTensor(predictions).to(device), y_test_tensor).item()

print(f'\n{" Test Results ":#^50}')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('#'*50)

# Create a DataFrame with predictions and metadata
results_df = test_metadata.copy()
results_df['pred_Class0'] = predictions[:, 0].numpy()
results_df['pred_Class1'] = predictions[:, 1].numpy()
results_df['actual_Class0'] = y_test[:, 0]
results_df['actual_Class1'] = y_test[:, 1]
results_df['predicted_class'] = np.argmax(predictions.numpy(), axis=1)
results_df['actual_class'] = np.argmax(y_test, axis=1)

Path('Research/Finance/Stock_Returns/Krauss/output').mkdir(parents=True, exist_ok=True)
results_df.to_csv(f'Research/Finance/Stock_Returns/Krauss/output/xlstm_predictions_{year}.csv', index=False)
print(f'\nPredictions saved to: Research/Finance/Stock_Returns/Krauss/output/xlstm_predictions_{year}.csv')
print(f'Model saved to: Research/Finance/Stock_Returns/Krauss/models/best_xlstm_model_{year}.pth')
