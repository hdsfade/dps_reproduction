#!/bin/bash

tasks=("inpainting" "gamma_blur" "gaussian_blur" "gray" "mosaic" "motion_blur" "super_resolution")
metrics=("fid" "lpips")
for task in "${tasks[@]}"; do
    for metric in "${metrics[@]}"; do
        echo "evaluate: $metric"
        python3 evaluation.py --root_dir results --task "$task" --method mcg --metric "$metric"
    done
done

echo "evaluation is done"