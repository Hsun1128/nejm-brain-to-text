#!/bin/bash

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate b2txt25_lm

python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --do_opt --nbest 100 \
    --acoustic_scale 0.325 \
    --blank_penalty 90 \
    --alpha 0.55 \
    --redis_ip localhost \
    --gpu_number 0