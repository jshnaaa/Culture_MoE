#!/bin/bash

# 测试 LoRA 合并工作流程

echo "============================================================"
echo "Testing LoRA Merge Workflow"
echo "============================================================"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试计数
PASSED=0
FAILED=0

# 测试函数
test_file_exists() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅ $2${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ $2${NC}"
        ((FAILED++))
        return 1
    fi
}

echo ""
echo "1. Checking required files..."
echo "------------------------------------------------------------"

test_file_exists "merge_lora_and_save.py" "merge_lora_and_save.py exists"
test_file_exists "run_merge_lora.sh" "run_merge_lora.sh exists"
test_file_exists "examples/eval_classification_dual_merged.py" "eval_classification_dual_merged.py exists"
test_file_exists "check_checkpoint.sh" "check_checkpoint.sh exists"

echo ""
echo "2. Checking Python imports..."
echo "------------------------------------------------------------"

python -c "from transformers import AutoModelForCausalLM, AutoTokenizer" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ transformers library available${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ transformers library not available${NC}"
    ((FAILED++))
fi

python -c "from peft import PeftModel, LoraConfig" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ peft library available${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ peft library not available${NC}"
    ((FAILED++))
fi

python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PyTorch available${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ PyTorch not available${NC}"
    ((FAILED++))
fi

echo ""
echo "3. Checking CUDA availability..."
echo "------------------------------------------------------------"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
if [ $? -eq 0 ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✅ CUDA available: $GPU_NAME${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  CUDA not available (CPU mode will be slow)${NC}"
fi

echo ""
echo "4. Checking workflow scripts..."
echo "------------------------------------------------------------"

# 检查脚本是否可执行
if [ -x "merge_lora_and_save.py" ]; then
    echo -e "${GREEN}✅ merge_lora_and_save.py is executable${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  merge_lora_and_save.py is not executable (run: chmod +x merge_lora_and_save.py)${NC}"
fi

if [ -x "run_merge_lora.sh" ]; then
    echo -e "${GREEN}✅ run_merge_lora.sh is executable${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️  run_merge_lora.sh is not executable (run: chmod +x run_merge_lora.sh)${NC}"
fi

echo ""
echo "5. Checking documentation..."
echo "------------------------------------------------------------"

test_file_exists "LORA_MERGE_WORKFLOW.md" "LORA_MERGE_WORKFLOW.md exists"
test_file_exists "FINAL_SOLUTION.md" "FINAL_SOLUTION.md exists"
test_file_exists "CHECKPOINT_LORA_GUIDE.md" "CHECKPOINT_LORA_GUIDE.md exists"

echo ""
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "You can now:"
    echo "  1. Train your model:"
    echo "     bash run_train_ddp_dual.sh"
    echo ""
    echo "  2. Merge LoRA weights:"
    echo "     bash run_merge_lora.sh"
    echo ""
    echo "  3. Evaluate merged model:"
    echo "     python examples/eval_classification_dual_merged.py ..."
    echo ""
    echo "See FINAL_SOLUTION.md for complete workflow."
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo ""
    echo "Please fix the issues above before proceeding."
    echo ""
    echo "Common fixes:"
    echo "  - Install missing libraries: pip install transformers peft torch"
    echo "  - Make scripts executable: chmod +x *.py *.sh"
fi

echo "============================================================"

exit $FAILED

