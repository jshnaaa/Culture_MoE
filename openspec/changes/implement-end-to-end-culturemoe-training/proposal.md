# Implement End-to-End CultureMoE Training

## Summary
Convert the current two-stage training approach (LoRA pre-training + frozen MoE training) to a unified end-to-end training methodology that jointly optimizes both the base language model and MoE components simultaneously.

## Problem Statement
The current `ft_culturemoe_from_base_gen.py` implementation follows a suboptimal two-stage approach:
1. Load pre-trained base model + fine-tuned LoRA weights (merged and frozen)
2. Add MoE layers on top and only train the MoE components
3. Base model parameters remain frozen during MoE training

This approach has several limitations:
- **Limited Adaptation**: The base model cannot adapt to the MoE routing decisions
- **Suboptimal Integration**: MoE layers are trained independently of base model representations
- **Training Inefficiency**: The base model's knowledge is not jointly optimized with cultural specialization
- **Performance Gap**: Two-stage training typically underperforms compared to end-to-end approaches

## Proposed Solution
Implement a unified end-to-end training approach that:
1. Starts from the base model (without pre-trained LoRA weights)
2. Adds MoE layers to the base model architecture
3. Jointly trains both base model and MoE components with cultural objectives
4. Uses gradient accumulation and mixed precision to handle memory constraints
5. Applies different learning rates for base model vs MoE components (layered learning rates)

## Benefits
- **Better Performance**: Joint optimization typically yields superior results
- **Unified Training**: Single training script instead of multiple stages
- **Better Integration**: MoE routing can influence base model representations
- **Simplified Workflow**: Eliminates need for separate LoRA pre-training step
- **More Flexible**: Easier to experiment with different architectural combinations

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