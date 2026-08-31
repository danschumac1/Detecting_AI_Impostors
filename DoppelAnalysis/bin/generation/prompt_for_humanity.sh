#!/bin/bash
# PID: ( 1472117 )
# chmod +x ./bin/generation/prompt_for_humanity.sh
# ./bin/generation/prompt_for_humanity.sh
# nohup ./bin/generation/prompt_for_humanity.sh > ./logs/prompt_for_humanity.log 2>&1 &
# tail -f ./logs/prompt_for_humanity.log

LOG_FILE="./logs/prompt_for_humanity.log"

###############################################################################
# CONFIGURATION
###############################################################################
# STATIC VARIABLES
CUDA_DEVICES="0"    # Change based on your free GPU
BATCH_SIZE=15

# ITERABLES
MODELS=(
    # "gpt"
    "llama"
    # "qwen"
)

SHOTS=(
    # "zs"
    "fs"
)

###############################################################################
# EXECUTION LOOP
###############################################################################
echo "--- Starting Humanity Labeling Script ---"
for model in "${MODELS[@]}"; do      
    for shot in "${SHOTS[@]}"; do
        LAST_CONTEXT="model=${model} | shot=${shot}"
        echo ">> Running: ${LAST_CONTEXT}"
        
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
            python ./src/prompt_for_humanity.py \
                --model_type "${model}" \
                --shots "${shot}" \
                --batch_size "${BATCH_SIZE}"
    done
done

echo "--- All tasks complete ---"