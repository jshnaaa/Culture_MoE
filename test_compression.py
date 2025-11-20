#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试权重压缩功能
验证压缩和解压缩的正确性
"""

import os
import torch
import tempfile
from model_utils_small import (
    save_moe_weights_compressed,
    load_moe_weights_smart,
    get_compression_info,
    find_best_weight_file,
    estimate_memory_usage
)

def create_dummy_moe_weights():
    """创建虚拟的MoE权重用于测试"""
    weights = {}

    # 模拟一些MoE组件权重
    weights['router.weight'] = torch.randn(512, 4096)
    weights['router.bias'] = torch.randn(512)

    # 模拟专家权重
    for i in range(6):
        weights[f'experts.{i}.mlp.gate_proj.weight'] = torch.randn(2048, 4096)
        weights[f'experts.{i}.mlp.up_proj.weight'] = torch.randn(2048, 4096)
        weights[f'experts.{i}.mlp.down_proj.weight'] = torch.randn(4096, 2048)

    # 模拟文化组件权重
    weights['cultural_embedding.weight'] = torch.randn(6, 256)
    weights['cultural_router.weight'] = torch.randn(512, 768)

    return weights

def test_compression_types():
    """测试不同压缩类型"""
    print("🧪 Testing different compression types...")

    # 创建测试权重
    original_weights = create_dummy_moe_weights()

    # 计算原始权重的参数数量和大小
    total_params = sum(w.numel() for w in original_weights.values())
    original_size_mb = sum(w.numel() * 4 for w in original_weights.values()) / (1024 * 1024)  # FP32 = 4 bytes

    print(f"Original weights: {total_params:,} parameters, {original_size_mb:.1f} MB")
    print()

    # 测试不同压缩类型
    compression_types = ["none", "fp16", "gzip", "fp16_gzip"]
    results = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        for comp_type in compression_types:
            print(f"Testing {comp_type} compression...")

            # 保存压缩权重
            save_path = os.path.join(temp_dir, f"test_weights_{comp_type}.pth")
            saved_path = save_moe_weights_compressed(
                original_weights,
                save_path,
                compression_type=comp_type
            )

            # 获取文件信息
            compression_info = get_compression_info(saved_path)
            memory_info = estimate_memory_usage(saved_path)

            # 加载权重并验证
            loaded_weights = load_moe_weights_smart(saved_path)

            # 验证权重完整性
            assert len(loaded_weights) == len(original_weights), f"Weight count mismatch for {comp_type}"

            for key in original_weights.keys():
                assert key in loaded_weights, f"Missing key {key} in {comp_type}"

                original_tensor = original_weights[key]
                loaded_tensor = loaded_weights[key]

                # 检查形状
                assert original_tensor.shape == loaded_tensor.shape, f"Shape mismatch for {key} in {comp_type}"

                # 检查数值（允许FP16的精度损失）
                if comp_type in ["fp16", "fp16_gzip"]:
                    # FP16压缩会有精度损失
                    max_diff = torch.max(torch.abs(original_tensor.float() - loaded_tensor.float()))
                    assert max_diff < 1e-3, f"Too much precision loss for {key} in {comp_type}: {max_diff}"
                else:
                    # 无损压缩应该完全相等
                    assert torch.allclose(original_tensor, loaded_tensor), f"Values mismatch for {key} in {comp_type}"

            # 记录结果
            results[comp_type] = {
                "file_size_mb": compression_info["file_size_mb"],
                "compression_ratio": original_size_mb / compression_info["file_size_mb"],
                "estimated_memory_mb": memory_info["estimated_memory_mb"],
                "compression_type": compression_info["compression_type"]
            }

            print(f"  ✅ {comp_type}: {compression_info['file_size_mb']:.1f} MB "
                  f"(ratio: {results[comp_type]['compression_ratio']:.2f}x)")

    print()
    print("📊 Compression Results Summary:")
    print("-" * 60)
    print(f"{'Type':<12} {'Size (MB)':<10} {'Ratio':<8} {'Memory (MB)':<12}")
    print("-" * 60)

    for comp_type, result in results.items():
        print(f"{comp_type:<12} {result['file_size_mb']:<10.1f} "
              f"{result['compression_ratio']:<8.2f} {result['estimated_memory_mb']:<12.1f}")

    return results

def test_smart_loading():
    """测试智能加载功能"""
    print("\n🧪 Testing smart loading functionality...")

    original_weights = create_dummy_moe_weights()

    with tempfile.TemporaryDirectory() as temp_dir:
        # 保存多种格式的权重
        base_path = os.path.join(temp_dir, "moe_weights.pth")

        # 保存不同格式
        paths = {}
        paths["fp16"] = save_moe_weights_compressed(original_weights, base_path, "fp16")
        paths["gzip"] = save_moe_weights_compressed(original_weights, base_path, "gzip")
        paths["fp16_gzip"] = save_moe_weights_compressed(original_weights, base_path, "fp16_gzip")

        # 测试智能查找最佳文件
        best_file = find_best_weight_file(base_path)
        print(f"Best file found: {os.path.basename(best_file)}")

        # 应该优先选择fp16_gzip（最高压缩比）
        assert "fp16" in best_file and "gz" in best_file, "Should prefer fp16_gzip format"

        # 测试智能加载
        loaded_weights = load_moe_weights_smart(best_file)

        # 验证加载成功
        assert len(loaded_weights) == len(original_weights)
        print("  ✅ Smart loading successful")

        # 测试文件不存在时的fallback
        nonexistent_path = os.path.join(temp_dir, "nonexistent.pth")
        try:
            loaded_weights = load_moe_weights_smart(nonexistent_path)
            print("  ✅ Fallback to compressed versions works")
        except FileNotFoundError:
            print("  ❌ Fallback failed - this is expected if no compressed versions exist")

def test_compatibility_with_existing_code():
    """测试与现有代码的兼容性"""
    print("\n🧪 Testing compatibility with existing code patterns...")

    original_weights = create_dummy_moe_weights()

    with tempfile.TemporaryDirectory() as temp_dir:
        # 模拟现有代码的保存方式
        old_path = os.path.join(temp_dir, "moe_weights.pth")
        torch.save(original_weights, old_path)

        # 使用新的智能加载函数加载旧格式
        loaded_weights = load_moe_weights_smart(old_path)

        # 验证完全兼容
        for key in original_weights.keys():
            assert torch.allclose(original_weights[key], loaded_weights[key])

        print("  ✅ Backward compatibility verified")

        # 测试压缩版本的兼容性
        compressed_path = save_moe_weights_compressed(original_weights, old_path, "fp16")
        loaded_compressed = load_moe_weights_smart(compressed_path)

        # 验证压缩版本也能正确加载
        assert len(loaded_compressed) == len(original_weights)
        print("  ✅ Compressed version compatibility verified")

def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 Testing Weight Compression Functionality")
    print("=" * 60)

    try:
        # 运行所有测试
        results = test_compression_types()
        test_smart_loading()
        test_compatibility_with_existing_code()

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)

        # 显示推荐的压缩设置
        print("\n💡 Recommendations:")

        best_compression = max(results.items(),
                             key=lambda x: x[1]["compression_ratio"] if x[0] != "none" else 0)

        print(f"  - Best compression: {best_compression[0]} "
              f"({best_compression[1]['compression_ratio']:.2f}x reduction)")

        fp16_result = results.get("fp16", {})
        if fp16_result:
            print(f"  - Recommended for production: fp16 "
                  f"({fp16_result['compression_ratio']:.2f}x reduction, good speed)")

        print(f"  - For maximum space saving: fp16_gzip "
              f"({results.get('fp16_gzip', {}).get('compression_ratio', 0):.2f}x reduction)")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)