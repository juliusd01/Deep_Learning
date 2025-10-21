
import pandas as pd

from helper import run_sign_model


# Parameters
NUMBER_OF_LAGS = [5]#[2,3,4,5,6,7,8,9,10,15,20]
TRAIN_RATIO = 0.8
HIDDEN_DIM = [16,32,64,128]
NUM_LAYERS = [1,2,3,4,5]
DROPOUT = [0.0]#,0.1,0.2,0.3,0.4,0.5]
LR = [3e-4]#[0.01, 1e-3, 1e-4]
EPOCHS = [50]#[20,50,100]
CLASS_THRESHOLD = [0.5]#[0.3,0.4,0.5,0.6,0.7]


batch_size = 64 # no hyperparameter

results = []

df = pd.read_csv('monthly_FRED-MD_2024-12_processed.csv')
X_all = df.drop(columns=['sign', 'volatility']).values
y_sign = df['sign'].values

results = run_sign_model(X=X_all, 
                         y=y_sign,
                         number_of_lags=NUMBER_OF_LAGS, 
                         train_ratio=TRAIN_RATIO,
                         hidden_dim=HIDDEN_DIM, 
                         num_layers=NUM_LAYERS, 
                         dropout=DROPOUT, 
                         lr=LR, 
                         batch_size=batch_size, 
                         epochs=EPOCHS, 
                         class_threshold=CLASS_THRESHOLD,
                         model_type='GRU')

# Find best model based on accuracy and if 0 was predicted at least once
best_model_candidates = results[results['recall_0'] > 0]
best_model = best_model_candidates.loc[best_model_candidates['accuracy'].idxmax()]
print("\n---------------------------------------------\nBest Model Parameters:")
print(best_model)



