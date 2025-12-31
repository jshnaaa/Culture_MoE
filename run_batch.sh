bash run_joint_lora_moe_training.sh llama 24

#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 0 true true true false
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 0 true true false
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 0 true false
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 0 false

bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 5 true false

#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 4 true true true csl
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 4 true true true false
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 4 true true false
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 4 true false
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 4 false

#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 3 true true true csl
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 3 true true true false
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 3 true true false
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 3 true false
#bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_gatetrue_20251230_212244 llama 3 false

shutdown