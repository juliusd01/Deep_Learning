#!/bin/bash
#SBATCH --job-name=cov_sector_lstm              # Name of the job
#SBATCH --output=Fischer/output/outs/cov_sector_%j.out  # Output file
#SBATCH --time=23:59:00                         # Max runtime (hh:mm:ss)
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --mem-per-cpu=16G                       # Memory per CPU
##SBATCH --gres=gpu:nvd:1                       # Request 1 GPU (commented for CPU)

module purge
module load python/3.11           # Load Python module (cluster-provided)

# Run LSTM with both covariates and sector heterogeneity
python -u Fischer/train_new.py \
    --model_architecture heterogeneity \
    --input_subdir lag_60_cov_sector \
    --model_type lstm_cov_sector \
    --experiment_type heterogeneity_covariates \
    --num_layers 1 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --num_lags 60 \
    --embedding_dim 2 \
    --recurrent_dropout 0.3 \
    --input_dropout 0.3 \
    --job_id $SLURM_JOB_ID
