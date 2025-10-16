#!/bin/bash
# test_classification.sh

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Running classification tests..."
python examples/test_classification.py --test all