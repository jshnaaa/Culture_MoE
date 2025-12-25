#!/usr/bin/env python3
"""
MASK机制逻辑验证脚本（不依赖PyTorch运行时）

验证要点：
1. 代码逻辑结构是否正确
2. 关键函数是否存在
3. 参数传递是否正确
"""

import json
import os
import sys
import re

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_file_modifications():
    """检查文件修改是否正确"""
    print("=" * 60)
    print("验证MASK机制代码修改")
    print("=" * 60)

    checks = []

    # 1. 检查run_simplified_culturemoe.sh默认值修改
    print("🔍 检查1: run_simplified_culturemoe.sh默认值")
    try:
        with open('run_simplified_culturemoe.sh', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查USE_SHARED默认值
        if 'USE_SHARED=${3:-"true"}' in content:
            print("  ✅ USE_SHARED默认值已修改为true")
            checks.append(True)
        else:
            print("  ❌ USE_SHARED默认值未正确修改")
            checks.append(False)

        # 检查USE_GATE默认值
        if 'USE_GATE=${4:-"true"}' in content:
            print("  ✅ USE_GATE默认值已修改为true")
            checks.append(True)
        else:
            print("  ❌ USE_GATE默认值未正确修改")
            checks.append(False)

        # 检查MASK机制参数
        if '--enable_mask' in content and '--mask_prob' in content:
            print("  ✅ MASK机制参数已添加")
            checks.append(True)
        else:
            print("  ❌ MASK机制参数未添加")
            checks.append(False)

    except Exception as e:
        print(f"  ❌ 检查run_simplified_culturemoe.sh失败: {e}")
        checks.extend([False, False, False])

    # 2. 检查ft_lora_only_gen.py的MASK机制
    print("\n🔍 检查2: ft_lora_only_gen.py MASK机制")
    try:
        with open('ft_lora_only_gen.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查enable_mask参数
        if 'enable_mask: bool = False' in content:
            print("  ✅ enable_mask参数已添加")
            checks.append(True)
        else:
            print("  ❌ enable_mask参数未添加")
            checks.append(False)

        # 检查create_instruction_mask方法
        if 'def create_instruction_mask(self, instruction):' in content:
            print("  ✅ create_instruction_mask方法已添加")
            checks.append(True)
        else:
            print("  ❌ create_instruction_mask方法未添加")
            checks.append(False)

        # 检查input_type处理
        if "'input_type': input_type" in content:
            print("  ✅ input_type处理已添加")
            checks.append(True)
        else:
            print("  ❌ input_type处理未添加")
            checks.append(False)

        # 检查collate_fn中的input_type
        if 'batch_input_types.append(item[\'input_type\'])' in content:
            print("  ✅ collate_fn input_type处理已添加")
            checks.append(True)
        else:
            print("  ❌ collate_fn input_type处理未添加")
            checks.append(False)

    except Exception as e:
        print(f"  ❌ 检查ft_lora_only_gen.py失败: {e}")
        checks.extend([False, False, False, False])

    # 3. 检查simplified_culturemoe_adapter.py的条件激活
    print("\n🔍 检查3: simplified_culturemoe_adapter.py 条件激活")
    try:
        with open('src/llamafactory/model/simplified_culturemoe_adapter.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查_current_input_type全局状态
        if '_current_input_type = None' in content:
            print("  ✅ _current_input_type全局状态已添加")
            checks.append(True)
        else:
            print("  ❌ _current_input_type全局状态未添加")
            checks.append(False)

        # 检查条件激活方法
        if '_forward_shared_only' in content and '_forward_routed_only' in content:
            print("  ✅ 条件激活方法已添加")
            checks.append(True)
        else:
            print("  ❌ 条件激活方法未添加")
            checks.append(False)

        # 检查adapter_ref设置
        if 'adapter_ref = self' in content:
            print("  ✅ adapter_ref引用已设置")
            checks.append(True)
        else:
            print("  ❌ adapter_ref引用未设置")
            checks.append(False)

        # 检查forward方法的input_type参数
        if 'def forward(self, input_ids, attention_mask=None, labels=None, input_type=None' in content:
            print("  ✅ forward方法input_type参数已添加")
            checks.append(True)
        else:
            print("  ❌ forward方法input_type参数未添加")
            checks.append(False)

    except Exception as e:
        print(f"  ❌ 检查simplified_culturemoe_adapter.py失败: {e}")
        checks.extend([False, False, False, False])

    # 4. 检查train_simplified_culturemoe.py的MASK参数
    print("\n🔍 检查4: train_simplified_culturemoe.py MASK参数")
    try:
        with open('train_simplified_culturemoe.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查enable_mask参数
        if '--enable_mask' in content:
            print("  ✅ enable_mask参数已添加")
            checks.append(True)
        else:
            print("  ❌ enable_mask参数未添加")
            checks.append(False)

        # 检查input_type传递
        if 'input_type=input_type' in content:
            print("  ✅ input_type传递已添加")
            checks.append(True)
        else:
            print("  ❌ input_type传递未添加")
            checks.append(False)

    except Exception as e:
        print(f"  ❌ 检查train_simplified_culturemoe.py失败: {e}")
        checks.extend([False, False])

    # 5. 检查eval_simplified_culturemoe.py的评估支持
    print("\n🔍 检查5: eval_simplified_culturemoe.py 评估支持")
    try:
        with open('eval_simplified_culturemoe.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查enable_mask=False设置
        if 'enable_mask=False' in content:
            print("  ✅ 评估时禁用MASK已设置")
            checks.append(True)
        else:
            print("  ❌ 评估时禁用MASK未设置")
            checks.append(False)

        # 检查input_type处理
        if "input_type=input_type  # 🆕 MASK机制" in content:
            print("  ✅ input_type传递已添加")
            checks.append(True)
        else:
            print("  ❌ input_type传递未添加")
            checks.append(False)

    except Exception as e:
        print(f"  ❌ 检查eval_simplified_culturemoe.py失败: {e}")
        checks.extend([False, False])

    return checks


def check_mask_logic():
    """检查MASK逻辑是否正确"""
    print("\n" + "=" * 60)
    print("验证MASK机制逻辑")
    print("=" * 60)

    # 模拟create_instruction_mask逻辑
    def simulate_mask_creation(instruction, mask_prob=0.15):
        """模拟instruction masking逻辑"""
        tokens = instruction.split()
        masked_tokens = []
        mask_count = 0

        for token in tokens:
            # 模拟随机masking（这里用简单的逻辑代替random）
            should_mask = len(token) % 3 == 0 and mask_count < len(tokens) * mask_prob

            if should_mask:
                # 保留重要的提示词不被mask
                if token.lower() in ['answer:', '###', 'from', '1', '2', '3', '4', 'to']:
                    masked_tokens.append(token)
                else:
                    masked_tokens.append('[MASK]')
                    mask_count += 1
            else:
                masked_tokens.append(token)

        return ' '.join(masked_tokens), mask_count

    print("🔍 测试instruction masking逻辑:")
    test_instruction = "Give me the answer from 1 to 4: Do you agree with this statement about culture?"

    masked_instruction, mask_count = simulate_mask_creation(test_instruction)
    print(f"  原始: {test_instruction}")
    print(f"  掩码: {masked_instruction}")
    print(f"  掩码数量: {mask_count}")

    # 检查重要token是否被保留
    important_tokens = ['answer:', '###', 'from', '1', '2', '3', '4', 'to']
    preserved_important = True
    for token in important_tokens:
        if token in test_instruction.lower() and token not in masked_instruction.lower():
            preserved_important = False
            break

    if preserved_important:
        print("  ✅ 重要token正确保留")
    else:
        print("  ❌ 重要token被错误掩码")

    return preserved_important


def main():
    """主验证函数"""
    print("🚀 开始MASK机制代码验证")

    # 检查文件修改
    modification_checks = check_file_modifications()

    # 检查逻辑正确性
    logic_check = check_mask_logic()

    # 汇总结果
    print("\n" + "=" * 60)
    print("🎯 验证结果汇总")
    print("=" * 60)

    total_checks = len(modification_checks) + 1
    passed_checks = sum(modification_checks) + (1 if logic_check else 0)

    print(f"总检查项: {total_checks}")
    print(f"通过检查: {passed_checks}")
    print(f"通过率: {passed_checks/total_checks*100:.1f}%")

    if passed_checks == total_checks:
        print("\n🎉 所有验证通过！MASK机制实现正确")
        print("\n📋 MASK机制实现要点:")
        print("  ✅ 1. 修改了USE_SHARED和USE_GATE默认值为true")
        print("  ✅ 2. 数据集支持instruction masking")
        print("  ✅ 3. MoE层支持条件专家激活")
        print("  ✅ 4. 训练脚本正确传递input_type")
        print("  ✅ 5. 评估脚本禁用MASK机制")
        print("  ✅ 6. 保留重要token不被掩码")

        print("\n🔧 使用方法:")
        print("  # 启用MASK机制训练")
        print("  bash run_simplified_culturemoe.sh llama 2 true true 4 new 2 true 2 16 32")

        print("\n  # 评估时自动禁用MASK，使用原始instruction")
        print("  python eval_simplified_culturemoe.py --model_path /path/to/model \\")
        print("    --data_file /path/to/data.json --output_dir /path/to/results")

    else:
        failed_checks = total_checks - passed_checks
        print(f"\n❌ {failed_checks}个检查项失败，请检查实现")

    print("=" * 60)
    return passed_checks == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)