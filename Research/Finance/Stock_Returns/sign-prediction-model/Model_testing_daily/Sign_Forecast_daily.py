
import pandas as pd
import torch
import logging

from helper import run_sign_model
from models import SignLSTM, SignGRU

logging.basicConfig(level=logging.INFO)

# Parameters
NUMBER_OF_LAGS = [60]
HIDDEN_DIM = [32]
NUM_LAYERS = [3]
DROPOUT = [0.0]#,0.1,0.5]
LR = [3e-4]
EPOCHS = [20, 100]
CLASS_THRESHOLD = [0.5]

batch_size = 64 # no hyperparameter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('input_data_daily.csv', index_col=0)

results = run_sign_model(input_data=df,
                         model_class=SignLSTM,
                         number_of_lags=NUMBER_OF_LAGS, 
                         hidden_dim=HIDDEN_DIM, 
                         num_layers=NUM_LAYERS, 
                         dropout=DROPOUT, 
                         lr=LR, 
                         epochs=EPOCHS, 
                         class_threshold=CLASS_THRESHOLD,
                         device=device)

# Find best model based on accuracy and if 0 was predicted at least once
logging.info("Model training completed. Summary of results:")
logging.info(results)