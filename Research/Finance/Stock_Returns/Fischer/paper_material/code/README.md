# LSTM Stock Returns Project

## File Structure

```md
Krauss_refactored/
├── train.py                          # Main training script
├── models.py                         # Model architectures
├── run_base_lstm.sh                 # Base LSTM (1 layer)
├── run_base_lstm_10layers.sh        # Base LSTM (10 layers, complexity study)
├── run_covariates_lstm.sh           # LSTM with covariates
├── run_heterogeneity_lstm.sh        # LSTM with sector embeddings
├── run_covariates_sector_lstm.sh    # LSTM with covariates + sectors
└── README.md                         # This file
```

## How It Works

### Unified Training Script

The `train.py` script accepts command-line arguments to configure any experiment:

```bash
python train.py \
    --model_architecture [base|covariates|heterogeneity] \
    --input_subdir [lag_60|lag_60_cov|lag_60_sector|etc] \
    --model_type [lstm|lstm_cov|lstm_sector] \
    --experiment_type [complexity|covariates|heterogeneity] \
    --num_layers 2 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 64 \
    [... other parameters]
```

### Model Architectures

Three model types are supported:

1. **base**: Standard LSTM for single feature (returns)
2. **covariates**: LSTM for multiple features
3. **heterogeneity**: LSTM with sector embeddings

### Shell Scripts

Each shell script is simple and just specifies the parameters for that experiment:

```bash
#!/bin/bash
#SBATCH ... [SLURM configuration]

python -u Krauss_refactored/train.py \
    --model_architecture base \
    --input_subdir lag_60 \
    --model_type lstm \
    --num_layers 1 \
    [... parameters]
```

## Available Arguments

### Required Arguments
- `--model_architecture`: Model type (`base`, `covariates`, `heterogeneity`)
- `--input_subdir`: Input data subdirectory
- `--model_type`: Model type for saving outputs
- `--experiment_type`: MLflow experiment name

### Model Hyperparameters
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Maximum epochs (default: 1000)
- `--learning_rate`: Learning rate (default: 0.001)
- `--num_layers`: Number of LSTM layers (default: 1)
- `--hidden_size`: LSTM hidden size (default: 25)
- `--num_lags`: Number of lags (default: 60)
- `--dropout`: Dropout rate (default: 0.0, auto 0.1 if num_layers > 1)
- `--embedding_dim`: Sector embedding dimension (default: 2, heterogeneity only)

### Training Settings
- `--patience`: Early stopping patience (default: 20)
- `--label_smoothing`: Label smoothing (default: 0.1)
- `--seed`: Random seed (default: 123)
- `--start_year`: Start year (default: 1998)
- `--end_year`: End year (default: 2024)
- `--sequential_sampler`: Use sequential sampler instead of shuffling
- `--base_dir`: Base directory (default: 'Krauss')

## Usage Examples

### Running Experiments

Submit a single experiment:

```bash
sbatch Krauss_refactored/run_base_lstm.sh
```

Submit multiple experiments:

```sh
sbatch Krauss_refactored/run_base_lstm.sh
sbatch Krauss_refactored/run_base_lstm_10layers.sh
sbatch Krauss_refactored/run_covariates_lstm.sh
sbatch Krauss_refactored/run_heterogeneity_lstm.sh
```

### Testing Locally

Test a configuration locally (useful before submitting to cluster):

```sh
python Krauss_refactored/train.py \
    --model_architecture base \
    --input_subdir lag_60 \
    --model_type lstm \
    --experiment_type test \
    --num_layers 1 \
    --epochs 10 \
    --start_year 2020 \
    --end_year 2020
```
