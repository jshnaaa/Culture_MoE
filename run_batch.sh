#!/bin/bash

LOG_DIR="/root/autodl-fs/data/batch/batch_moe3_experts34578_logs"
mkdir -p $LOG_DIR
BATCH_LOG="$LOG_DIR/batch_training_${TIMESTAMP}.log"

sh run_ft_culturemoe_gen.sh qwen 3 True 5
sh run_ft_culturemoe_gen.sh qwen 3 True 4
sh run_ft_culturemoe_gen.sh qwen 3 True 3
sh run_ft_culturemoe_gen.sh qwen 3 True 7
sh run_ft_culturemoe_gen.sh qwen 3 True 8

sh run_ft_culturemoe_gen.sh llama 3 True 5
sh run_ft_culturemoe_gen.sh llama 3 True 4
sh run_ft_culturemoe_gen.sh llama 3 True 3
sh run_ft_culturemoe_gen.sh llama 3 True 7
sh run_ft_culturemoe_gen.sh llama 3 True 8