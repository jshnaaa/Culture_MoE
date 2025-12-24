#!/usr/bin/env python3
"""
验证8:1:1数据划分的正确性
确保训练集、验证集、测试集之间没有重叠
"""

import pickle
import sys
import os

def verify_data_split(split_file):
    """验证数据划分文件的正确性"""

    if not os.path.exists(split_file):
        print(f"❌ 数据划分文件不存在: {split_file}")
        return False

    # 加载划分信息
    with open(split_file, 'rb') as f:
        split_info = pickle.load(f)

    print("=" * 60)
    print("📊 数据划分验证")
    print("=" * 60)

    # 基本信息
    print(f"数据文件: {split_info.get('data_path', 'N/A')}")
    print(f"总样本数: {split_info['total_size']}")
    print(f"划分方法: {split_info.get('split_method', 'N/A')}")
    print(f"随机种子: {split_info.get('random_seed', 'N/A')}")
    print()

    # 各集合大小
    train_indices = set(split_info['train_indices'])
    val_indices = set(split_info['val_indices'])
    test_indices = set(split_info['test_indices'])

    train_size = len(train_indices)
    val_size = len(val_indices)
    test_size = len(test_indices)
    total_size = split_info['total_size']

    print("📈 各集合大小:")
    print(f"  训练集: {train_size:,} 样本 ({train_size/total_size*100:.1f}%)")
    print(f"  验证集: {val_size:,} 样本 ({val_size/total_size*100:.1f}%)")
    print(f"  测试集: {test_size:,} 样本 ({test_size/total_size*100:.1f}%)")
    print(f"  总计: {train_size + val_size + test_size:,} 样本")
    print()

    # 验证完整性
    all_indices = train_indices | val_indices | test_indices
    expected_indices = set(range(total_size))

    print("🔍 完整性检查:")
    if len(all_indices) == total_size:
        print("  ✅ 索引数量正确")
    else:
        print(f"  ❌ 索引数量错误: 期望{total_size}, 实际{len(all_indices)}")

    if all_indices == expected_indices:
        print("  ✅ 索引范围正确 (0 到 {})".format(total_size-1))
    else:
        print("  ❌ 索引范围错误")
        missing = expected_indices - all_indices
        extra = all_indices - expected_indices
        if missing:
            print(f"    缺失索引: {sorted(list(missing))[:10]}...")
        if extra:
            print(f"    多余索引: {sorted(list(extra))[:10]}...")
    print()

    # 验证无重叠
    print("🔍 重叠检查:")

    train_val_overlap = train_indices & val_indices
    train_test_overlap = train_indices & test_indices
    val_test_overlap = val_indices & test_indices

    if len(train_val_overlap) == 0:
        print("  ✅ 训练集与验证集无重叠")
    else:
        print(f"  ❌ 训练集与验证集有{len(train_val_overlap)}个重叠样本")
        print(f"    重叠索引: {sorted(list(train_val_overlap))[:10]}...")

    if len(train_test_overlap) == 0:
        print("  ✅ 训练集与测试集无重叠")
    else:
        print(f"  ❌ 训练集与测试集有{len(train_test_overlap)}个重叠样本")
        print(f"    重叠索引: {sorted(list(train_test_overlap))[:10]}...")

    if len(val_test_overlap) == 0:
        print("  ✅ 验证集与测试集无重叠")
    else:
        print(f"  ❌ 验证集与测试集有{len(val_test_overlap)}个重叠样本")
        print(f"    重叠索引: {sorted(list(val_test_overlap))[:10]}...")

    print()

    # 验证比例
    print("📊 比例检查:")
    train_ratio = train_size / total_size
    val_ratio = val_size / total_size
    test_ratio = test_size / total_size

    print(f"  训练集比例: {train_ratio:.3f} (期望: ~0.800)")
    print(f"  验证集比例: {val_ratio:.3f} (期望: ~0.100)")
    print(f"  测试集比例: {test_ratio:.3f} (期望: ~0.100)")

    # 检查比例是否合理
    ratio_ok = (0.79 <= train_ratio <= 0.81 and
                0.09 <= val_ratio <= 0.11 and
                0.09 <= test_ratio <= 0.11)

    if ratio_ok:
        print("  ✅ 比例分配合理")
    else:
        print("  ⚠️ 比例分配可能不理想")

    print()

    # 总结
    all_checks_passed = (
        len(all_indices) == total_size and
        all_indices == expected_indices and
        len(train_val_overlap) == 0 and
        len(train_test_overlap) == 0 and
        len(val_test_overlap) == 0 and
        ratio_ok
    )

    if all_checks_passed:
        print("🎉 所有检查通过！数据划分正确。")
        return True
    else:
        print("⚠️ 存在问题，请检查数据划分。")
        return False

def main():
    if len(sys.argv) != 2:
        print("用法: python verify_data_split.py <split_file>")
        print("示例: python verify_data_split.py /path/to/data_split_8_1_1.pkl")
        sys.exit(1)

    split_file = sys.argv[1]
    success = verify_data_split(split_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()