#!/usr/bin/env python3
"""
测试Only MoE DDP参数重复标记问题
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.model.only_moe_model import OnlyMoEModel, OnlyMoEConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_distributed():
    """简化的分布式设置"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def test_only_moe_ddp():
    """测试Only MoE DDP"""
    rank, world_size, local_rank = setup_distributed()

    print(f"Testing DDP on rank {rank}/{world_size}")

    # 使用小模型进行测试
    base_model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
    device = torch.device(f"cuda:{local_rank}")

    # 加载基础模型
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    base_model = base_model.to(device)

    # 创建OnlyMoE配置
    config = OnlyMoEConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        num_moe_experts=2,  # 使用较少的专家进行测试
        top_k=2,
        moe_layers=[26, 27],  # 最后两层
        aux_loss_coef=0.01,
        dropout=0.1
    )

    # 创建OnlyMoE模型
    print("Creating OnlyMoE model...")
    model = OnlyMoEModel(base_model, config)

    if rank == 0:
        model.print_trainable_parameters()

    # 清理内存
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # DDP包装
    if world_size > 1:
        print("Wrapping with DDP...")
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )
        model._set_static_graph()
        print("✅ DDP wrapped successfully")

    # 创建测试数据
    batch_size = 2
    seq_len = 64
    vocab_size = base_model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :seq_len//2] = -100  # mask前半部分

    # 测试前向传播
    print("Testing forward pass...")
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        loss = outputs.loss
        print(f"Forward pass successful, loss: {loss.item():.4f}")

        # 测试反向传播
        print("Testing backward pass...")
        loss.backward()
        print("✅ Backward pass successful")

        # 检查梯度
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        print(f"Parameters with gradients: {grad_count}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_only_moe_ddp()

    if dist.is_initialized():
        dist.destroy_process_group()

    exit(0 if success else 1)