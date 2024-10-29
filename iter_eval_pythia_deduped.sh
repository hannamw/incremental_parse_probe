#!/bin/bash
for i in {0..6}; do
    EXPERIMENT_PATH=experiment_checkpoints/eval/pythia-70m-deduped/StackActionProbe/layer_$i/
    echo $EXPERIMENT_PATH
    python3 src/parse.py --experiment_path $EXPERIMENT_PATH
done