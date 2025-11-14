# Implementation Tasks for End-to-End CultureMoE Training

## Phase 1: Core Architecture Changes

### 1. Model Initialization Refactoring
- [ ] Remove LoRA weight loading and merging logic from `ft_culturemoe_from_base_gen.py:845-860`
- [ ] Update model initialization to start directly from base model
- [ ] Add proper weight initialization for MoE components
- [ ] Test model creation without LoRA dependency

### 2. Layered Learning Rate Implementation
- [ ] Create `LayeredOptimizer` class to handle different learning rates per component
- [ ] Implement parameter grouping logic for base model vs MoE components
- [ ] Add learning rate scheduling for each parameter group
- [ ] Validate parameter assignment and gradient flow

### 3. Parameter Management Updates
- [ ] Remove parameter freezing logic (`lines 912-958`)
- [ ] Implement trainable parameter configuration for all components
- [ ] Add parameter counting and memory estimation utilities
- [ ] Update parameter saving/loading to handle full model state

## Phase 2: Training Pipeline Updates

### 4. Training Loop Modifications
- [ ] Implement cultural loss warmup strategy in `train_epoch` function
- [ ] Update gradient accumulation to handle larger memory requirements
- [ ] Add gradient checkpointing for base model layers
- [ ] Implement progressive learning rate adjustment

### 5. Memory Optimization
- [ ] Add mixed precision training configuration
- [ ] Implement gradient checkpointing for base model
- [ ] Add memory monitoring and reporting
- [ ] Optimize batch size and accumulation steps automatically

### 6. Loss Function Enhancement
- [ ] Implement warmup factor calculation for cultural loss
- [ ] Add router temperature scheduling
- [ ] Update loss weighting strategy for joint training
- [ ] Add loss component monitoring and logging

## Phase 3: Stability and Monitoring

### 7. Training Stability Features
- [ ] Implement aggressive gradient clipping (0.5-1.0)
- [ ] Add NaN/Inf detection and recovery mechanisms
- [ ] Implement early stopping based on validation metrics
- [ ] Add checkpoint saving at regular intervals

### 8. Enhanced Monitoring
- [ ] Add per-component gradient norm tracking
- [ ] Implement router weight distribution monitoring
- [ ] Add learning rate tracking for all parameter groups
- [ ] Create training stability dashboards

### 9. Evaluation Pipeline Updates
- [ ] Update evaluation to handle joint training checkpoints
- [ ] Add comparison metrics against two-stage baseline
- [ ] Implement cultural consistency evaluation (VSM13)
- [ ] Add memory and time profiling during evaluation

## Phase 4: Configuration and Integration

### 10. Configuration Management
- [ ] Add end-to-end training configuration options
- [ ] Update argument parsing for new training parameters
- [ ] Create configuration presets for different model sizes
- [ ] Add validation for configuration compatibility

### 11. Script Integration
- [ ] Create new training script `ft_culturemoe_end_to_end.py`
- [ ] Update shell scripts for end-to-end training
- [ ] Add backward compatibility with existing workflows
- [ ] Update documentation and usage examples

### 12. Testing and Validation
- [ ] Create unit tests for layered optimization
- [ ] Add integration tests for end-to-end training
- [ ] Implement memory usage validation tests
- [ ] Add performance comparison test suite

## Phase 5: Optimization and Tuning

### 13. Hyperparameter Optimization
- [ ] Tune layered learning rates for optimal performance
- [ ] Optimize warmup schedule and cultural loss weighting
- [ ] Find optimal gradient accumulation and batch size settings
- [ ] Validate router temperature scheduling

### 14. Performance Optimization
- [ ] Profile training pipeline for bottlenecks
- [ ] Optimize memory usage and reduce peak consumption
- [ ] Implement efficient checkpoint saving/loading
- [ ] Add distributed training support if needed

### 15. Validation and Comparison
- [ ] Run extensive training experiments comparing approaches
- [ ] Validate convergence across different datasets
- [ ] Compare final model performance with two-stage baseline
- [ ] Document performance improvements and trade-offs

## Validation Criteria

Each task must meet these criteria before completion:
- **Functionality**: Feature works as designed without errors
- **Testing**: Adequate test coverage with passing tests
- **Documentation**: Clear documentation of changes and usage
- **Performance**: No significant performance regression
- **Memory**: Memory usage remains within acceptable limits
- **Stability**: Training remains stable across multiple runs

## Dependencies

### Task Dependencies
- Tasks 1-3 must complete before Tasks 4-6
- Tasks 4-9 can be developed in parallel
- Tasks 10-12 depend on completion of Tasks 1-9
- Tasks 13-15 require completion of all previous tasks

### External Dependencies
- PyTorch >= 2.0.0 for mixed precision training
- Sufficient GPU memory for joint training (recommend 40GB+)
- Updated CUDA drivers for gradient checkpointing features
- Access to base models (LLaMA-3.1-8B, Qwen-2.5-7B)

## Success Metrics

### Implementation Success
- All tasks completed with passing tests
- End-to-end training script successfully trains model
- Memory usage stays within GPU limits
- Training converges with stable loss curves

### Performance Success
- Cultural classification accuracy ≥ current two-stage approach
- Training time per epoch ≤ 2x current approach
- Memory usage ≤ available GPU memory
- Model quality metrics maintain or improve