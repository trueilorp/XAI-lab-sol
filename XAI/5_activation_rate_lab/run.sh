#!/bin/bash

SEED=42
BATCH_SIZE=10
ENV=rware:rware-img-3-tiny-2ag-v2
CHECKPOINTS_PATH=models
EXPLANATIONS=True
CURRENT_TIME="`date +%Y%m%d%H%M%S`"

for theory_index in 0 1 2 3 4 5 6 7 8 9
do
    for step in 1001000 2001000 3001000 4001000 5001000 6001000 7001000 20001000
    do
        echo "Running checkpoint at step $step (batch size $BATCH_SIZE) -- index $theory_index"
        sleep 1s
        uv run python src/main.py --config=mappo --env-config=gymma with env_args.time_limit=100 seed=$SEED current_time=$CURRENT_TIME theory_index=$theory_index explanations=$EXPLANATIONS batch_size_run=$BATCH_SIZE env_args.key=$ENV t_max=1000000 checkpoint_path=$CHECKPOINTS_PATH load_step=$step render=False common_reward=False
    done
done

