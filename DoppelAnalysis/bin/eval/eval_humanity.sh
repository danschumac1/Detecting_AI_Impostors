#!/bin/bash
# PID: ( insert_pid_here )
# chmod +x ./bin/eval/eval_humanity.sh
# ./bin/eval/eval_humanity.sh
# nohup ./bin/eval/eval_humanity.sh > ./logs/eval_humanity.log 2>&1 &
# tail -f ./logs/eval_humanity.log

###############################################################################
# CONFIGURATION
###############################################################################
# ITERABLES
MODELS=(
    # "gpt"
    # "llama"
    # "qwen"
    # "uniform_random"
    "always_1"
)

SHOTS=(
    "zs"
    # "fs"
)

###############################################################################
# EXECUTION LOOP
###############################################################################
echo "--- Starting Humanity Evaluation Script ---"

for model in "${MODELS[@]}"; do      
    for shot in "${SHOTS[@]}"; do
        LAST_CONTEXT="model=${model} | shot=${shot}"
        echo ">> Evaluating: ${LAST_CONTEXT}"
        
        python ./src/eval_humanity.py \
            --model "${model}" \
            --shots "${shot}"
    done
done

echo "--- All tasks complete ---"