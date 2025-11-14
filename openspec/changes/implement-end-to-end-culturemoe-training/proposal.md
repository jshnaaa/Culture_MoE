# Implement End-to-End CultureMoE Training

## Summary
Convert the current two-stage training approach (LoRA pre-training + frozen MoE training) to a unified end-to-end training methodology that jointly optimizes both the base language model and MoE components simultaneously.

## Problem Statement
The current `ft_culturemoe_from_base_gen.py` implementation follows a suboptimal approach:
1. Load pre-trained base model + fine-tuned LoRA weights (merged and frozen)
2. Add MoE layers on top and only train the MoE components
3. Fine-tuned base model parameters remain frozen during MoE training

This approach has several limitations:
- **Limited Adaptation**: The fine-tuned base model cannot further adapt to the MoE routing decisions
- **Suboptimal Integration**: MoE layers are trained independently of the fine-tuned base model representations
- **Training Inefficiency**: The fine-tuned base model's knowledge is not jointly optimized with cultural specialization
- **Performance Gap**: Frozen fine-tuned models typically underperform compared to continued joint training

## Proposed Solution
Implement a unified end-to-end training approach that:
1. Starts from the fine-tuned model (loads and merges pre-trained LoRA weights)
2. Adds MoE layers to the fine-tuned model architecture
3. Jointly trains both fine-tuned model and MoE components with cultural objectives
4. Uses gradient accumulation and mixed precision to handle memory constraints
5. Applies different learning rates for fine-tuned model vs MoE components (layered learning rates)

## Benefits
- **Better Performance**: Joint optimization of fine-tuned model + MoE typically yields superior results
- **Leverages Existing Work**: Builds upon already fine-tuned LoRA weights rather than starting from scratch
- **Better Integration**: MoE routing can influence and adapt the fine-tuned model representations
- **Simplified Workflow**: Single training stage instead of frozen parameter training
- **More Flexible**: Easier to experiment with different architectural combinations while preserving fine-tuned knowledge

## Implementation Scope
This change affects:
- Training methodology in `ft_culturemoe_from_base_gen.py`
- Parameter initialization and optimization strategies
- Memory management and training efficiency
- Evaluation and monitoring workflows

## Success Criteria
- End-to-end training script successfully trains from base model
- Training converges with stable loss curves
- Cultural evaluation metrics show improvement over two-stage approach
- Memory usage remains manageable with gradient accumulation
- Training time is reasonable compared to current approach