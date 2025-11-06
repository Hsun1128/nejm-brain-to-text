#!/bin/bash

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the b2txt25 conda environment
conda activate b2txt25

# Run the evaluation script
python evaluate_model.py \
    --model_path trained_models/rnn_balanced_sampling/ \
    --data_dir ../data/t15_copyTask_neuralData/hdf5_data_final/ \
    --csv_path ../data/t15_copyTaskData_description.csv \
    --eval_type val \
    --gpu_number 0 \
    --compute_per \
    --quiet \
    --save_text val_phonemes.txt \
    --skip_lm

echo "Evaluation complete."