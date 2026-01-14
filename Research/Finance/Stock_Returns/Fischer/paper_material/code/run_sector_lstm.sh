#!/bin/bash
#SBATCH --job-name=sector_lstm                  # Name of the job
#SBATCH --output=Fischer/output/outs/sector_%j.out  # Output file
#SBATCH --time=23:59:00                         # Max runtime (hh:mm:ss)
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --mem-per-cpu=12G                       # Memory per CPU
#SBATCH --gres=gpu:a100:1                       # Request 1 GPU

module purge
module load python/3.11           # Load Python module (cluster-provided)

# Run LSTM with sector heterogeneity
python -u Fischer/train_new.py \
    --model_architecture heterogeneity \
    --input_subdir lag_60_sector \
    --model_type lstm_sector \
    --experiment_type heterogeneity \
    --num_layers 2 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --num_lags 60 \
    --embedding_dim 5 \
    --job_id $SLURM_JOB_ID \
    --recurrent_dropout 0.2 \
    --input_dropout 0.05