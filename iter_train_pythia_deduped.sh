#!/bin/bash
for filename in configs/eval/pythia-70m-deduped/StackActionProbe/*; do
    python src/train.py --config $filename
done