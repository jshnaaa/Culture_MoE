#!/usr/bin/env python3
"""
测试 SaveFullModelCallback 是否正确工作
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

def test_callback_import():
    """测试回调是否可以正确导入"""
    print("Testing callback import...")
    try:
        from src.llamafactory.train.classification.callbacks import SavePeftModelCallback, SaveFullModelCallback
        print("✅ Callbacks imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import callbacks: {e}")
        return False

def test_callback_structure():
    """测试回调类的结构"""
    print("\nTesting callback structure...")
    try:
        from src.llamafactory.train.classification.callbacks import SaveFullModelCallback
        from transformers import TrainerCallback

        # 检查是否继承自 TrainerCallback
        if not issubclass(SaveFullModelCallback, TrainerCallback):
            print("❌ SaveFullModelCallback does not inherit from TrainerCallback")
            return False

        # 检查是否有必要的方法
        callback = SaveFullModelCallback()
        if not hasattr(callback, 'on_save'):
            print("❌ SaveFullModelCallback does not have on_save method")
            return False

        print("✅ Callback structure is correct")
        return True
    except Exception as e:
        print(f"❌ Failed to test callback structure: {e}")
        return False

def test_workflow_import():
    """测试 workflow 是否正确导入回调"""
    print("\nTesting workflow import...")
    try:
        # 读取 workflow.py 文件
        workflow_path = "src/llamafactory/train/classification/workflow.py"
        with open(workflow_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否导入了回调
        if 'from .callbacks import' not in content:
            print("❌ workflow.py does not import callbacks")
            return False

        if 'SaveFullModelCallback' not in content:
            print("❌ workflow.py does not import SaveFullModelCallback")
            return False

        # 检查是否使用了回调
        if 'callbacks=' not in content:
            print("❌ workflow.py does not use callbacks in Trainer")
            return False

        print("✅ workflow.py correctly imports and uses callbacks")
        return True
    except Exception as e:
        print(f"❌ Failed to test workflow import: {e}")
        return False

def main():
    """运行所有测试"""
    print("="*60)
    print("Testing SaveFullModelCallback Implementation")
    print("="*60)

    tests = [
        test_callback_import,
        test_callback_structure,
        test_workflow_import,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ All tests passed! The callback is correctly implemented.")
        print("\nNext steps:")
        print("  1. Run training: bash run_train_ddp_lora_dual.sh")
        print("  2. Verify checkpoints: bash verify_checkpoint_save.sh <output_dir>")
        print("  3. Evaluate checkpoint: bash run_eval_dual_binary.sh")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the implementation.")

    print("="*60)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

