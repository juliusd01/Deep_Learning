#!/bin/bash
#SBATCH --job-name=cov_lstm                     # Name of the job
#SBATCH --output=Fischer/output/outs/cov_%j.out  # Output file
#SBATCH --time=23:59:00                         # Max runtime (hh:mm:ss)
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --mem-per-cpu=16G                       # Memory per CPU
#SBATCH --gres=gpu:a100:1                       # Request 1 GPU

module purge
module load python/3.11           # Load Python module (cluster-provided)

# Run LSTM with covariates
python -u Fischer/train_new.py \
    --model_architecture covariates \
    --input_subdir lag_60_cov \
    --model_type lstm_cov \
    --experiment_type covariates \
    --num_layers 2 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --num_lags 60 \
    --recurrent_dropout 0.2 \
    --input_dropout 0.2 \
    --job_id $SLURM_JOB_ID