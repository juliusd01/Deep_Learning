import random
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

seed = 123

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 32
EPOCHS = 5000
LEARNING_RATE = 0.001
NUM_LAYERS = 1
HIDDEN_SIZE = 25
NUM_LAGS = 60
PATIENCE = 1000
START_Y = 2013       
END_Y = 2013        

dir = 'Krauss'
model_type = 'lstm'
model_id = f'h_{HIDDEN_SIZE}_l_{NUM_LAYERS}_lr_{LEARNING_RATE}_lags_{NUM_LAGS}'


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_classes=2, num_layers=NUM_LAYERS):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM with multiple layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states for all layers
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Process sequence through LSTM layers
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Use final hidden state from last layer
        out = self.fc(h_n[-1])
        return out

sequences_df = pd.read_parquet(f'Research/Finance/Stock_Returns/Krauss/data/returns_per_period/lag_60/returns_2013.parquet')

# PREPARE TENSORS
X = np.array(sequences_df['sequence'].tolist())
X = X.reshape(X.shape[0], NUM_LAGS, 1)
y = sequences_df[['Class0', 'Class1']].values
y = np.argmax(y, axis=1)

# For overfitting test: use only 30 training samples and 20 test samples from the same data
X_train = X[:1000]
y_train = y[:1000]
X_test = X[1000:1020]
y_test = y[1000:1020]

print(f"\nOverfitting Check - Train size: {len(X_train)}, Test size: {len(X_test)}")

# Initialize model
model = LSTMModel()
print(model)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
g = torch.Generator()
g.manual_seed(seed)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop - no early stopping, no scheduler for overfitting test
max_epochs = EPOCHS

for epoch in range(max_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_X, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Test phase (evaluate on test set)
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, labels in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    train_acc = train_correct / train_total
    test_acc = test_correct / test_total
    
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # Stop if perfect training accuracy achieved
    if train_acc == 1.0:
        print(f'\n✓ Perfect training accuracy achieved at epoch {epoch+1}!')
        print(f'Final Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        break