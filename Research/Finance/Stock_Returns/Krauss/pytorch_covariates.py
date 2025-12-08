import random
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import mlflow

seed = 123

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_LAYERS = 1
HIDDEN_SIZE = 25
NUM_LAGS = 60
PATIENCE = 20
START_Y = 1998
END_Y = 1998
experiment_type = 'covariates'

dir = 'Research/Finance/Stock_Returns/Krauss'
model_type = 'lstm'
model_id = f'h_{HIDDEN_SIZE}_l_{NUM_LAYERS}_lr_{LEARNING_RATE}_lag_{NUM_LAGS}_cov'

# ---------------- MLflow settings ----------------
mlflow.set_experiment(experiment_type)
# mlflow.set_tracking_uri("<PATH_TO_MLFLOW_TRACKING>")  # optional
# -------------------------------------------------

# ---------------- PARENT RUN START ----------------------
with mlflow.start_run(run_name=f"{model_type}_{model_id}") as parent_run:

    mlflow.set_tag("hidden_size", HIDDEN_SIZE)
    mlflow.set_tag("num_layers", NUM_LAYERS)
    mlflow.set_tag("learning_rate", LEARNING_RATE)

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


    for year in range(START_Y, END_Y + 1):
        with mlflow.start_run(run_name=f"year_{year}", nested=True):
            sequences_df = pd.read_parquet(f'{dir}/data/returns_per_period/lag_{NUM_LAGS}_cov/returns_{year}.parquet')
            feature_names = [col for col in sequences_df.columns if col.endswith('sequence')]
            number_of_features = len(feature_names)
            
            sequences_df = sequences_df.sort_values('date')
            sequences_df = sequences_df.reset_index(drop=True)

            # Stack all feature sequences along the last dimension
            X = np.stack([np.array(sequences_df[col].tolist()) for col in feature_names], axis=-1)
            print("Shape X: ", X.shape)
            
            # Check for NaN or Inf values and replace
            if np.isnan(X).any() or np.isinf(X).any():
                print("WARNING: NaN or Inf values found in input features.")
            
            y = sequences_df[['Class0', 'Class1']].values
            y = np.argmax(y, axis=1)

            # Train-Test Split based on year
            test_mask = sequences_df['date'].dt.year == int(year)
            X_test = X[test_mask.values]
            y_test = y[test_mask.values]
            X_train_val = X[~test_mask.values]
            y_train_val = y[~test_mask.values]

            # Keep the metadata for test set (stock, date, return)
            test_metadata = sequences_df[test_mask][['stock', 'date', 'return']].reset_index(drop=True)

            # Split train_val into train and val (80% train, 20% val of the train_val set)
            X_train = X_train_val[:int(0.8 * len(X_train_val))]
            y_train = y_train_val[:int(0.8 * len(y_train_val))]
            X_val = X_train_val[int(0.8 * len(X_train_val)):]
            y_val = y_train_val[int(0.8 * len(y_train_val)):]

            # Initialize model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'Using device: {device}')
            model = LSTMModel(input_size=number_of_features).to(device)
            
            # Check model weights for NaN
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"WARNING: NaN in {name} after initialization!")
            
            print(model)

            # Log parameters
            mlflow.log_param("year", year)
            mlflow.log_param("seed", seed)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("hidden_size", HIDDEN_SIZE)
            mlflow.log_param("num_layers", NUM_LAYERS)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("num_lags", NUM_LAGS)
            mlflow.log_param("device", device)
            num_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("num_parameters", num_params)
            mlflow.log_param("covariates", feature_names)

            # Convert data to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.LongTensor(y_train).to(device)
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)

            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Learning rate scheduler - reduces LR when validation loss plateaus
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',           # minimize validation loss
                factor=0.5,           # reduce LR by half
                patience=5,           # wait 5 epochs before reducing
                min_lr=1e-6          # minimum learning rate
            )

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = PATIENCE
            patience_counter = 0
            max_epochs = EPOCHS
            os.makedirs(f"{dir}/models/{model_type}/{model_id}", exist_ok=True)
            best_model_path = f"{dir}/models/{model_type}/{model_id}/best_lstm_model_{year}.pth"

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
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"NaN loss detected at epoch {epoch+1}!")
                        print(f"Outputs: {outputs[:5]}")
                        break
                    
                    loss.backward()
                    # Clip gradients to prevent explosion
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, labels in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total

                # Log epoch metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

                # Step the scheduler
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        mlflow.log_metric("epochs", epoch+1)
                        break

        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        print('Training complete!')
        # Evaluate on test set
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)

        model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X_test_tensor), 256):
                batch = X_test_tensor[i:i+256]             
                pred = model(batch)                        
                pred = torch.softmax(pred, dim=1)          
                predictions.append(pred)                   

            # Concatenate all predictions
            predictions = torch.cat(predictions, dim=0)    

            # Calculate metrics
            _, predicted_labels = torch.max(predictions, 1)
            test_accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor)
            test_loss = criterion(predictions, y_test_tensor).item()


        # Log test metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_accuracy)

        # Convert predictions to CPU for Pandas/Numpy
        pred_np = predictions.cpu().numpy()

        # Create a DataFrame with predictions and metadata
        results_df = test_metadata.copy()
        results_df['pred_Class0'] = pred_np[:, 0]
        results_df['pred_Class1'] = pred_np[:, 1]
        results_df['predicted_class'] = np.argmax(pred_np, axis=1)
        results_df['actual_class'] = y_test

        output_parquet = f"{dir}/output/{model_type}/{model_id}/predictions_{year}.parquet"
        os.makedirs(f"{dir}/output/{model_type}/{model_id}", exist_ok=True)
        results_df.to_parquet(output_parquet, index=False)

        mlflow.log_artifact(output_parquet)
        mlflow.log_artifact(best_model_path)

        print(f'Year {year} complete! Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
        print('-' * 50)