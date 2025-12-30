# Implementation Tasks for CSL Culture Loss

## Phase 1: Core CSL Function Implementation

### 1.1 Implement CSL Loss Function
- [ ] Create `compute_csl_culture_loss()` function in `train_joint_lora_moe.py`
- [ ] Implement L_culture_router calculation with pairwise cosine similarity
- [ ] Implement L_culture_share calculation for culture-invariant shared expert loss
- [ ] Implement L_culture_sr calculation for shared-router decoupling loss
- [ ] Add proper tensor dimension handling and batch size validation
- [ ] Include numerical stability checks for cosine similarity calculations
- [ ] Add comprehensive error handling for edge cases (empty batches, zero vectors)

### 1.2 Expert Output Extraction
- [ ] Verify JointLoRAMoEModel output format provides required expert outputs
- [ ] Implement extraction of router expert outputs (er_i vectors)
- [ ] Implement extraction of shared expert outputs (es_i vectors)
- [ ] Implement extraction of router weights (wr_i vectors)
- [ ] Add validation for expert output tensor shapes and types
- [ ] Handle cases where shared expert or router outputs are unavailable

### 1.3 Efficient Similarity Computation
- [ ] Implement vectorized cosine similarity for batch processing
- [ ] Optimize pairwise similarity calculations for memory efficiency
- [ ] Add batch size considerations for large datasets
- [ ] Implement incremental computation to avoid memory overflow
- [ ] Add proper gradient flow preservation for all similarity calculations

## Phase 2: Integration and Configuration

### 2.1 Training Pipeline Integration
- [ ] Modify training loop in `train_epoch_joint()` to detect `USE_CULTURE_LOSS=csl`
- [ ] Replace existing culture loss with CSL when appropriate
- [ ] Maintain backward compatibility with existing loss modes (`ori`, `new`, `kl`)
- [ ] Add CSL loss component logging and monitoring
- [ ] Ensure proper loss weighting and gradient accumulation

### 2.2 Configuration Updates
- [ ] Update `run_joint_lora_moe_training.sh` default `USE_CULTURE_LOSS` to `csl`
- [ ] Add CSL-specific parameter documentation in script comments
- [ ] Update configuration validation to recognize `csl` option
- [ ] Add CSL loss component weight configuration options
- [ ] Update training configuration JSON to include CSL parameters

### 2.3 Model Output Validation
- [ ] Verify JointLoRAMoEModel provides required expert output formats
- [ ] Add model output validation for CSL requirements
- [ ] Enhance model forward pass to ensure expert outputs are available
- [ ] Add fallback behavior when expert outputs are missing
- [ ] Test CSL with different model configurations (shared/no-shared, gate/no-gate)

## Phase 3: Documentation and Validation

### 3.1 Architecture Documentation Update
- [ ] Update `culturemoe.md` to reflect joint training model architecture
- [ ] Document CSL loss function mathematical formulation
- [ ] Describe three CSL components and their objectives
- [ ] Add joint training pipeline description with LoRA+MoE integration
- [ ] Include CSL parameter guidance and usage examples
- [ ] Document model output requirements for CSL functionality

### 3.2 Loss Function Documentation
- [ ] Document L_culture_router design and implementation
- [ ] Document L_culture_share culture-invariance objective
- [ ] Document L_culture_sr decoupling mechanism
- [ ] Add mathematical formulations for all three components
- [ ] Include computational complexity analysis
- [ ] Document CSL vs existing culture loss comparisons

### 3.3 Comprehensive Testing
- [ ] Test CSL with different batch sizes (1, 2, 4, 8, 16)
- [ ] Validate CSL behavior with different culture label distributions
- [ ] Test CSL memory usage and computational overhead
- [ ] Verify backward compatibility with existing training configurations
- [ ] Test CSL with different model architectures (LLaMA, Qwen)
- [ ] Validate CSL gradient flow and optimization stability

## Phase 4: Quality Assurance and Optimization

### 4.1 Performance Validation
- [ ] Benchmark CSL computational overhead vs existing culture loss
- [ ] Measure memory usage impact of CSL implementation
- [ ] Profile CSL similarity calculations for optimization opportunities
- [ ] Test CSL scalability with large batch sizes and long sequences
- [ ] Validate CSL training convergence and stability

### 4.2 Edge Case Handling
- [ ] Test CSL with single-sample batches
- [ ] Handle cases with uniform culture labels (all same/all different)
- [ ] Test CSL with missing or invalid expert outputs
- [ ] Validate CSL with zero vectors and numerical edge cases
- [ ] Test CSL with extreme similarity values (0.0, 1.0)

### 4.3 Integration Testing
- [ ] Test CSL with distributed training (multi-GPU)
- [ ] Validate CSL with gradient accumulation and mixed precision
- [ ] Test CSL with different optimizer configurations
- [ ] Verify CSL compatibility with existing evaluation pipelines
- [ ] Test CSL with different dataset formats and sizes

## Validation Criteria

### Functional Validation
- CSL loss function produces three meaningful loss components
- CSL integrates seamlessly with existing joint training pipeline
- Backward compatibility maintained for all existing configurations
- CSL parameters configurable through training script

### Performance Validation
- CSL computational overhead < 10% of total training time
- Memory usage increase < 20% compared to baseline training
- Training convergence stability maintained with CSL
- All CSL components contribute non-zero gradients during training

### Quality Validation
- Comprehensive error handling for all edge cases
- Clear documentation of CSL architecture and usage
- Numerical stability in all similarity calculations
- Proper tensor management and gradient flow preservation