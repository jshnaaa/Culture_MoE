#!/bin/bash
# scripts/predict_classification.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

MODEL_PATH="./output/classification/checkpoint-best"
INPUT_FILE="data/test_samples.txt"
OUTPUT_FILE="./output/predictions.json"

python examples/predict_classification.py \
  --model_path ${MODEL_PATH} \
  --input_file ${INPUT_FILE} \
  --output_file ${OUTPUT_FILE} \
  --batch_size 16 \
  --device cuda

echo "Predictions saved to ${OUTPUT_FILE}"