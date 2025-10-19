from transformers import AutoTokenizer
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct')
data = load_and_process_dual_classification_data(
    '/root/autodl-fs/normad_ed_merge.json',
    tokenizer,
    max_length=512,
    val_split=0.1
)
print('Train size:', len(data['train']))
print('Val size:', len(data['validation']))
print('Sample:', data['train'][0])
