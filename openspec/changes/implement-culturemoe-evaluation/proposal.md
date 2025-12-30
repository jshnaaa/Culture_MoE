# Implement CultureMoE Evaluation System

## Summary

This proposal implements a comprehensive evaluation system for trained CultureMoE models, enabling standardized testing on held-out test sets with proper model reconstruction from saved training parameters.

## Why

The CultureMoE training pipeline successfully trains models but lacks systematic evaluation capabilities. Without proper evaluation tools, researchers cannot:
- Validate model performance on held-out test sets
- Compare different model configurations objectively
- Reproduce evaluation results across different environments
- Ensure test set isolation and evaluation integrity

This proposal addresses these gaps by implementing a comprehensive evaluation system that maintains the same rigor as the training pipeline.

## Problem Statement

Currently, the CultureMoE training pipeline (`run_simplified_culturemoe.sh`) successfully trains models and saves the best performing checkpoints, but lacks a dedicated evaluation system to:

1. **Load trained models correctly**: Reconstruct complete CultureMoE models from saved LoRA weights, MoE weights, and configuration files
2. **Evaluate on test sets**: Use the 8:1:1 dataset splits to evaluate models on previously unseen test data
3. **Standardize evaluation metrics**: Provide consistent accuracy, precision, recall, and F1-score measurements
4. **Support reproducible research**: Enable systematic comparison of different model configurations

## Goals

### Primary Goals
- Create `run_eval_simplified_culturemoe.sh` script with parameters for model output directory and base model
- Implement `eval_simplified_culturemoe.py` for model loading and evaluation
- Support all trained model configurations (different backbones, MoE settings, LoRA configurations)
- Generate comprehensive evaluation reports with multiple metrics

### Secondary Goals
- Maintain compatibility with existing training pipeline outputs
- Follow project conventions for code style and architecture
- Provide clear error handling and user feedback

## Technical Approach

### Model Loading Strategy
1. **Base Model Loading**: Load the original base model (LLaMA/Qwen) from specified path
2. **Configuration Reconstruction**: Parse `simplified_culturemoe_config.json` to rebuild model architecture
3. **Weight Integration**:
   - Load LoRA weights from `lora_weights/` directory
   - Load MoE weights from `moe_weights.pt` file
   - Merge weights with base model using SimplifiedCultureMoEAdapter
4. **Model Validation**: Verify model architecture matches training configuration

### Dataset Handling
1. **Split File Loading**: Read test set indices from saved `.pkl` split files
2. **Data Preprocessing**: Apply same tokenization and formatting as training
3. **Batch Processing**: Use efficient batching for evaluation to handle large test sets

### Evaluation Metrics
1. **Classification Metrics**: Accuracy, Precision, Recall, F1-score for cultural classification tasks
2. **Generation Metrics**: Answer extraction and comparison for generative tasks
3. **Detailed Reporting**: Per-class metrics and confusion matrices where applicable

## Implementation Plan

### Phase 1: Core Evaluation Infrastructure
1. Create `eval_simplified_culturemoe.py` with model loading capabilities
2. Implement dataset loading and preprocessing for test sets
3. Add basic accuracy evaluation for classification tasks

### Phase 2: Shell Script and Integration
1. Create `run_eval_simplified_culturemoe.sh` with parameter handling
2. Add configuration validation and error checking
3. Integrate with existing project structure and conventions

### Phase 3: Enhanced Metrics and Reporting
1. Implement comprehensive evaluation metrics
2. Add detailed evaluation reports and logging
3. Support for different evaluation modes (classification vs generation)

## Success Criteria

### Functional Requirements
- [ ] Successfully load any trained CultureMoE model from output directory
- [ ] Evaluate models on correct test set splits (8:1:1 division)
- [ ] Generate accurate classification metrics matching training evaluation
- [ ] Handle all supported model configurations (LLaMA/Qwen, different MoE settings)

### Quality Requirements
- [ ] Follow existing code style and naming conventions
- [ ] Provide clear error messages for common failure modes
- [ ] Include comprehensive logging and progress reporting
- [ ] Maintain compatibility with existing training outputs

### Performance Requirements
- [ ] Efficient memory usage during evaluation (similar to training constraints)
- [ ] Support for GPU acceleration where available
- [ ] Reasonable evaluation time for typical test set sizes

## Risks and Mitigation

### Technical Risks
- **Model Loading Complexity**: CultureMoE models have complex architecture with LoRA + MoE components
  - *Mitigation*: Reuse existing SimplifiedCultureMoEAdapter loading logic
- **Memory Constraints**: Large models may exceed available GPU memory during evaluation
  - *Mitigation*: Implement batch processing and memory optimization techniques
- **Configuration Compatibility**: Different training configurations may require different loading approaches
  - *Mitigation*: Comprehensive configuration validation and error handling

### Project Risks
- **Code Duplication**: Risk of duplicating existing training logic
  - *Mitigation*: Reference but don't modify existing code, create new independent modules
- **Testing Limitations**: Cannot test locally without GPU environment
  - *Mitigation*: Focus on code review and logical validation, design for robustness

## Dependencies

### Existing Components
- `SimplifiedCultureMoEAdapter` for model architecture
- `CultureLLMNewFormatDataset` for data loading
- Training configuration and weight saving formats
- Dataset split pickle files from training pipeline

### New Components
- `eval_simplified_culturemoe.py`: Core evaluation script
- `run_eval_simplified_culturemoe.sh`: Shell wrapper script
- Evaluation metric calculation utilities
- Report generation and logging systems

## Timeline

This proposal can be implemented immediately as it builds on existing, stable training infrastructure. The implementation is self-contained and does not require modifications to existing code.

**Estimated effort**: 2-3 development sessions
- Session 1: Core model loading and basic evaluation
- Session 2: Shell script, error handling, and integration
- Session 3: Enhanced metrics and comprehensive testing

## Future Considerations

### Potential Extensions
- **Batch Evaluation**: Support for evaluating multiple models simultaneously
- **Comparative Analysis**: Tools for comparing different model configurations
- **Visualization**: Generate plots and charts for evaluation results
- **Integration Testing**: Automated testing of training → evaluation pipeline

### Maintenance
- Keep evaluation system synchronized with training pipeline changes
- Update metric calculations as research requirements evolve
- Maintain compatibility with new model architectures and datasets