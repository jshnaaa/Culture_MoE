# examples/eval_classification_dual.py
#!/usr/bin/env python3
"""
双路输入分类任务评估脚本
在测试集上评估训练好的模型（支持 instruction + instruction_mask）双路
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
from src.llamafactory.train.classification.metrics import compute_classification_metrics


def load_model(model_path: str, base_model_path: str = None, use_lora: bool = True):
    """
    加载训练好的模型

    Args:
        model_path: 训练输出的模型路径（包含 MoE 权重）
        base_model_path: 原始 LLaMA 模型路径（如果为 None，尝试从 model_path 读取）
        use_lora: 是否使用了 LoRA 微调
    """
    print(f"Loading model from {model_path}...")

    # ✅ 尝试从训练输出目录读取 tokenizer，如果失败则使用 base_model_path
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

    # ✅ 尝试从训练输出目录读取配置，确定 base_model_path
    if base_model_path is None:
        # 尝试从 trainer_state.json 读取原始模型路径
        trainer_state_path = os.path.join(model_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                    # 尝试从训练参数中获取原始模型路径
                    # 这个路径可能在不同的字段中
                    print("⚠️  Could not find base model path in trainer_state.json")
            except Exception as e:
                print(f"⚠️  Could not read trainer_state.json: {e}")

        # 如果还是找不到，要求用户提供
        if base_model_path is None:
            raise ValueError(
                f"Could not determine base model path from {model_path}. "
                "Please provide --base_model_path argument."
            )

    # ✅ 从原始路径加载基础模型
    print(f"Loading base LLaMA model from {base_model_path}...")

    # ✅ 确定使用的设备（单个 GPU）
    if torch.cuda.is_available():
        target_device = "cuda:0"  # 使用第一个 GPU
        print(f"Using device: {target_device}")
    else:
        target_device = "cpu"
        print("Using device: cpu")

    llama_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if target_device.startswith("cuda") else torch.float32,
        device_map=None,  # ✅ 不使用 device_map，手动管理设备
        trust_remote_code=True
    )

    # ✅ 手动将模型移到目标设备
    llama_model = llama_model.to(target_device)

    # ✅ 如果使用了 LoRA，加载 LoRA 权重
    if use_lora:
        try:
            from peft import PeftModel
            print(f"Loading LoRA weights from {model_path}...")
            llama_model = PeftModel.from_pretrained(
                llama_model,
                model_path,
                is_trainable=False
            )
            # 合并 LoRA 权重以加速推理
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
        num_classes=3,
        classification_hidden_dim=512,
        dropout=0.1,
        num_heads=8
    )

    model = LlamaSharedRouterExpertsModel(
        llama_model=llama_model,
        config=llama_model.config,
        args=moe_args
    )

    # ✅ 加载 MoE 组件的权重
    try:
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            print(f"Loading MoE weights from {state_dict_path}...")
            # ✅ 先获取模型所在的设备
            model_device = next(model.parameters()).device
            print(f"Model device: {model_device}")

            # ✅ 加载权重到正确的设备
            state_dict = torch.load(state_dict_path, map_location=model_device)

            # ✅ 只加载 MoE 组件的权重（不包括 llama_model）
            moe_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('llama_model.'):
                    moe_state_dict[key] = value

            # ✅ 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(moe_state_dict, strict=False)

            print("✅ MoE weights loaded")
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   Unexpected keys: {len(unexpected_keys)}")
        else:
            print(f"⚠️  Warning: {state_dict_path} not found")
            print("   Trying alternative weight file names...")

            # 尝试其他可能的文件名
            alternative_paths = [
                os.path.join(model_path, "model.safetensors"),
                os.path.join(model_path, "pytorch_model.safetensors"),
            ]

            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"   Found: {alt_path}")
                    if alt_path.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(alt_path, device=str(next(model.parameters()).device))
                    else:
                        state_dict = torch.load(alt_path, map_location=next(model.parameters()).device)

                    moe_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('llama_model.')}
                    model.load_state_dict(moe_state_dict, strict=False)
                    print("✅ MoE weights loaded from alternative file")
                    break
    except Exception as e:
        print(f"⚠️  Warning: Could not load MoE weights: {e}")
        import traceback
        traceback.print_exc()

    # ✅ 确保所有模块都在正确的设备上
    model_device = next(model.parameters()).device
    print(f"\nEnsuring all modules are on {model_device}...")
    model = model.to(model_device)

    model.eval()

    print(f"✅ Model loaded on: {next(model.parameters()).device}")

    # ✅ 验证各个组件的设备
    print("\nVerifying component devices:")
    print(f"  llama_model: {next(model.llama_model.parameters()).device}")
    print(f"  shared: {next(model.shared.parameters()).device}")
    print(f"  router: {next(model.router.parameters()).device}")
    print(f"  experts_layer: {next(model.experts_layer.parameters()).device}")
    print(f"  classifier: {next(model.classifier.parameters()).device}")

    return model, tokenizer


def evaluate_model(
        model,
        tokenizer,
        test_file: str,
        batch_size: int = 8,
        max_length: int = 512,
        output_file: str = None
):
    """
    评估模型（双路输入）

    Args:
        model: 训练好的模型
        tokenizer: tokenizer
        test_file: 测试数据文件
        batch_size: 批次大小
        max_length: 最大序列长度
        output_file: 输出文件路径（可选）
    """
    print(f"\nLoading test data from {test_file}...")

    # ✅ 加载双路输入测试数据
    data = load_and_process_dual_classification_data(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=max_length,
        val_split=0,  # 不分割
        num_proc=4
    )

    test_dataset = data['train']  # 因为 val_split=0，所有数据都在 train 中
    print(f"Test dataset size: {len(test_dataset)}")

    # 准备评估
    device = next(model.parameters()).device
    all_predictions = []
    all_labels = []
    all_logits = []

    print(f"\nEvaluating on {len(test_dataset)} samples...")

    # 批量评估
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        # ✅ 获取批次数据（使用 select 方法）
        batch_indices = list(range(i, min(i + batch_size, len(test_dataset))))
        batch = test_dataset.select(batch_indices)

        # ✅ 准备双路输入
        # 使用 pad_sequence 处理不同长度的序列
        from torch.nn.utils.rnn import pad_sequence

        # 第一组输入
        input_ids_list = [torch.tensor(item) for item in batch['input_ids']]
        attention_mask_list = [torch.tensor(item) for item in batch['attention_mask']]

        # 第二组输入
        input_ids_mask_list = [torch.tensor(item) for item in batch['input_ids_mask']]
        attention_mask_mask_list = [torch.tensor(item) for item in batch['attention_mask_mask']]

        # Padding
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)
        input_ids_mask = pad_sequence(input_ids_mask_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask_mask = pad_sequence(attention_mask_mask_list, batch_first=True, padding_value=0).to(device)

        labels = torch.tensor(batch['labels'])

        # 前向传播
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_mask=input_ids_mask,
                attention_mask_mask=attention_mask_mask
            )

        # 获取预测
        preds = torch.argmax(logits, dim=-1).cpu()

        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_logits.extend(logits.cpu().tolist())

    # 转换为 numpy 数组
    import numpy as np
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # 计算指标
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    metrics = compute_classification_metrics((predictions, labels))

    # 打印主要指标
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:        {metrics['accuracy']:.4f}")
    print(f"   F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"   F1 (weighted):   {metrics['f1_weighted']:.4f}")

    print(f"\n📊 Per-class Metrics:")
    for class_name in ["no", "neutral", "yes"]:
        print(f"\n   {class_name.upper()}:")
        print(f"      Precision: {metrics[f'precision_{class_name}']:.4f}")
        print(f"      Recall:    {metrics[f'recall_{class_name}']:.4f}")
        print(f"      F1:        {metrics[f'f1_{class_name}']:.4f}")

    # 打印混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    print(f"\n📊 Confusion Matrix:")
    print("                Predicted")
    print("              no  neutral  yes")
    print(f"Actual no      {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
    print(f"Actual neutral {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
    print(f"Actual yes     {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")

    # 保存结果
    if output_file:
        results = {
            "metrics": metrics,
            "predictions": all_predictions,
            "labels": all_labels,
            "logits": all_logits,
            "confusion_matrix": cm.tolist()
        }

        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ Results saved!")

    print("\n" + "=" * 60)

    return metrics


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评估双路输入分类模型")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径（checkpoint 目录）")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="原始 LLaMA 模型路径（如果不提供，会尝试自动检测）")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件（需包含 instruction, instruction_mask, input, output 字段）")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="是否使用了 LoRA 微调")

    args = parser.parse_args()

    # 检查 GPU
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
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_file=args.output_file
    )

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

