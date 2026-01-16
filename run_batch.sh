echo "！！！0.005"
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_masktrue_gatetrue_losscsl_0.01_0.005_20260116_224500 llama 0
echo "！！！0.01"
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_masktrue_gatetrue_losscsl_0.01_0.01_20260116_105251 llama 0
echo "！！！0.02"
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_masktrue_gatetrue_losscsl_0.01_0.02_20260116_212428 llama 0
echo "！！！0.05"
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_masktrue_gatetrue_losscsl_0.01_0.05_20260116_211442 llama 0
echo "！！！0.1"
bash run_eval_joint_culturemoe.sh /autodl-fs/data/joint_lora_moe/llama_CulturalBench_sharedtrue_masktrue_gatetrue_losscsl_0.01_0.1_20260116_210344 llama 0
shutdown