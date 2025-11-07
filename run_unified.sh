#!/bin/bash

python prepare_unified_dataset.py \
    --culturellm /root/autodl-fs/cultureLLM_merge_gen.json \
    --normad /root/autodl-fs/normad_merge_gen.json \
    --culturalbench /root/autodl-fs/CulturalBench_merge_gen.json \
    --output /root/autodl-fs/unified_all_datasets.json \
    --shuffle