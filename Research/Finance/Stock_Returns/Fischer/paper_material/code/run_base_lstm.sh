#!/bin/bash
#SBATCH --job-name=base_lstm                    # Name of the job
#SBATCH --output=Fischer/output/outs/base_%j.out  # Output file
#SBATCH --time=23:59:00                         # Max runtime (hh:mm:ss)
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --mem-per-cpu=16G                       # Memory per CPU
##SBATCH --gres=gpu:nvd:1                        # Request 1 GPU

module purge
module load python/3.11           # Load Python module (cluster-provided)

# Run base LSTM with 1 layer
python -u Fischer/train_new.py \
    --model_architecture base \
    --input_subdir lag_60 \
    --model_type lstm \
    --experiment_type complexity \
    --num_layers 1 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --num_lags 60 \
    --job_id $SLURM_JOB_ID \
    --recurrent_dropout 0.2 \
    --input_dropout 0.2