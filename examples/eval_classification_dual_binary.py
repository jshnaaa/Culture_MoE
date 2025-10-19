# examples/eval_classification_dual_binary.py
#!/usr/bin/env python3
"""
双路输入分类任务评估脚本（三分类模型 → 二分类评估）
在测试集上评估训练好的模型（支持 instruction + instruction_mask）
"""

import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llamafactory.data.dual_classification_processor import load_and_process_dual_classification_data
from src.llamafactory.model.CultureMoE import LlamaSharedRouterExpertsModel
from src.llamafactory.model.moe_args import ModelArgs


def load_model(model_path: str, base_model_path: str = None, use_lora: bool = True):
    """加载训练好的模型（与原评估脚本相同）"""
    print(f"Loading model from {model_path}...")

    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Tokenizer loaded from checkpoint")
    except Exception as e:
        if base_model_path:
            print(f"⚠️  Could not load tokenizer from checkpoint: {e}")
            print(f"   Loading tokenizer from base model: {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        else:
            raise ValueError(f"Could not load tokenizer from {model_path} and no base_model_path provided")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model_path is None:
        raise ValueError("Please provide --base_model_path argument.")

    # 加载基础模型
    print(f"Loading base LLaMA model from {base_model_path}...")

    if torch.cuda.is_available():
        target_device = "cuda:0"
        print(f"Using device: {target_device}")
    else:
        target_device = "cpu"
        print("Using device: cpu")

    llama_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if target_device.startswith("cuda") else torch.float32,
        device_map=None,
        trust_remote_code=True
    )

    llama_model = llama_model.to(target_device)

    # 加载 LoRA 权重
    if use_lora:
        try:
            from peft import PeftModel
            print(f"Loading LoRA weights from {model_path}...")
            llama_model = PeftModel.from_pretrained(
                llama_model,
                model_path,
                is_trainable=False
            )
            print("Merging LoRA weights...")
            llama_model = llama_model.merge_and_unload()
            print("✅ LoRA weights loaded and merged")
        except Exception as e:
            print(f"⚠️  Warning: Could not load LoRA weights: {e}")
            print("   Using base model only")

    # 重建 MoE 模型
    moe_args = ModelArgs(
        num_experts=6,
        shared_hidden_dim=2048,
        router_hidden_dim=1024,
        experts_hidden_dim=2048,
        lora_rank=16,
        num_classes=3,  # 保持三分类
        classification_hidden_dim=512,
        dropout=0.1,
        num_heads=8
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    # 加载 MoE 权重
    try:
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            print(f"Loading MoE weights from {state_dict_path}...")
            model_device = next(model.parameters()).device
            state_dict = torch.load(state_dict_path, map_location=model_device)

            moe_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('llama_model.'):
                    moe_state_dict[key] = value

            model.load_state_dict(moe_state_dict, strict=False)
            print("✅ MoE weights loaded")
    except Exception as e:
        print(f"⚠️  Warning: Could not load MoE weights: {e}")

    # 确保所有模块在正确的设备和数据类型上
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    print(f"\nEnsuring all modules are on {model_device} with dtype {model_dtype}...")
    model = model.to(device=model_device, dtype=model_dtype)

    model.eval()

    print(f"✅ Model loaded on: {next(model.parameters()).device}")
    print(f"   Model dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def map_three_to_binary(three_class_pred, strategy="merge_neutral_to_no"):
    """
    将三分类预测映射到二分类

    Args:
        three_class_pred: 三分类预测 (0=no, 1=neutral, 2=yes)
        strategy: 映射策略
            - "merge_neutral_to_no": neutral → no (0)
            - "merge_neutral_to_yes": neutral → yes (1)
            - "threshold": 使用概率阈值

    Returns:
        二分类预测 (0=no, 1=yes)
    """
    if strategy == "merge_neutral_to_no":
        # neutral (1) → no (0)
        return 0 if three_class_pred in [0, 1] else 1
    elif strategy == "merge_neutral_to_yes":
        # neutral (1) → yes (1)
        return 0 if three_class_pred == 0 else 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_model_binary(
        model,
        tokenizer,
        test_file: str,
        batch_size: int = 8,
        max_length: int = 512,
        output_file: str = None,
        mapping_strategy: str = "merge_neutral_to_no"
):
    """
    评估模型（三分类 → 二分类）
    """
    print(f"\nLoading test data from {test_file}...")
    print(f"Mapping strategy: {mapping_strategy}")

    # 加载数据
    data = load_and_process_dual_classification_data(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=max_length,
        val_split=0,
        num_proc=4
    )

    test_dataset = data['train']
    print(f"Test dataset size: {len(test_dataset)}")

    device = next(model.parameters()).device
    all_predictions_3class = []
    all_predictions_binary = []
    all_labels_original = []
    all_labels_binary = []
    all_logits = []

    print(f"\nEvaluating on {len(test_dataset)} samples...")

    # 批量评估
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch_indices = list(range(i, min(i + batch_size, len(test_dataset))))
        batch = test_dataset.select(batch_indices)

        from torch.nn.utils.rnn import pad_sequence

        input_ids_list = [torch.tensor(item) for item in batch['input_ids']]
        attention_mask_list = [torch.tensor(item) for item in batch['attention_mask']]
        input_ids_mask_list = [torch.tensor(item) for item in batch['input_ids_mask']]
        attention_mask_mask_list = [torch.tensor(item) for item in batch['attention_mask_mask']]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)
        input_ids_mask = pad_sequence(input_ids_mask_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask_mask = pad_sequence(attention_mask_mask_list, batch_first=True, padding_value=0).to(device)

        labels_original = torch.tensor(batch['labels'])

        # 前向传播
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask
            )

        # 三分类预测
        preds_3class = torch.argmax(logits, dim=-1).cpu()

        # 映射到二分类
        preds_binary = [map_three_to_binary(p.item(), mapping_strategy) for p in preds_3class]

        # 原始标签也映射到二分类（假设测试集标签是 0/1）
        labels_binary = labels_original.tolist()

        all_predictions_3class.extend(preds_3class.tolist())
        all_predictions_binary.extend(preds_binary)
        all_labels_original.extend(labels_original.tolist())
        all_labels_binary.extend(labels_binary)
        all_logits.extend(logits.cpu().tolist())

    # 计算二分类指标
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

    predictions_binary = np.array(all_predictions_binary)
    labels_binary = np.array(all_labels_binary)

    print("\n" + "=" * 60)
    print("Evaluation Results (Binary Classification)")
    print("=" * 60)

    # 整体指标
    accuracy = accuracy_score(labels_binary, predictions_binary)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_binary, predictions_binary, average='binary', pos_label=1
    )

    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:   {accuracy:.4f}")
    print(f"   Precision:  {precision:.4f}")
    print(f"   Recall:     {recall:.4f}")
    print(f"   F1:         {f1:.4f}")

    # 分类报告
    print(f"\n📊 Classification Report:")
    print(classification_report(
        labels_binary,
        predictions_binary,
        target_names=["no (0)", "yes (1)"],
        digits=4
    ))

    # 混淆矩阵
    cm = confusion_matrix(labels_binary, predictions_binary)
    print(f"\n📊 Confusion Matrix:")
    print("           Predicted")
    print("           no   yes")
    print(f"Actual no  {cm[0][0]:3d}  {cm[0][1]:3d}")
    print(f"Actual yes {cm[1][0]:3d}  {cm[1][1]:3d}")

    # 保存结果
    if output_file:
        results = {
            "mapping_strategy": mapping_strategy,
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            },
            "predictions_3class": all_predictions_3class,
            "predictions_binary": all_predictions_binary,
            "labels_original": all_labels_original,
            "labels_binary": all_labels_binary,
            "logits": all_logits,
            "confusion_matrix": cm.tolist()
        }

        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ Results saved!")

    print("\n" + "=" * 60)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评估双路输入分类模型（三分类→二分类）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径（checkpoint 目录）")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="原始 LLaMA 模型路径")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件（二分类数据）")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="是否使用了 LoRA 微调")
    parser.add_argument("--mapping_strategy", type=str,
                        default="merge_neutral_to_no",
                        choices=["merge_neutral_to_no", "merge_neutral_to_yes"],
                        help="三分类到二分类的映射策略")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available!")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    # 加载模型
    model, tokenizer = load_model(
        args.model_path,
        base_model_path=args.base_model_path,
        use_lora=args.use_lora
    )

    # 评估模型
    metrics = evaluate_model_binary(
        model=model,
        tokenizer=tokenizer,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_file=args.output_file,
        mapping_strategy=args.mapping_strategy
    )

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

