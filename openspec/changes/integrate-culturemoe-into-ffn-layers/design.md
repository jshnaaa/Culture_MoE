# Design: FFN-Integrated CultureMoE Architecture

## Context

The current Culture_Moe project implements an Enhanced CultureMoE as an independent layer on top of frozen backbone models. While this approach allows for efficient training, it suffers from several limitations:

1. **Route Collapse**: Expert utilization analysis shows severe underutilization (load balance loss: 0.458, entropy loss: 0.194)
2. **Shallow Cultural Integration**: Cultural awareness only applied at the final layer
3. **Performance Bottlenecks**: Per-sample expert dispatch loops causing training inefficiency
4. **Limited Feature Learning**: Cultural information doesn't participate in intermediate transformations

This design addresses these issues by integrating CultureMoE directly into the Transformer's FFN layers while maintaining training efficiency and numerical stability.

## Goals / Non-Goals

### Goals
- Replace inefficient per-sample expert routing with vectorized batch operations
- Integrate cultural awareness into every Transformer layer through FiLM modulation
- Prevent route collapse through proper load balancing and entropy regularization
- Maintain memory efficiency through optimization strategies and progressive training
- Preserve model interface compatibility for seamless integration
- Achieve better cultural understanding through deeper integration

### Non-Goals
- Maintaining backward compatibility with existing checkpoints (breaking change accepted)
- Supporting arbitrary numbers of cultures (limited to 6 continents for initial implementation)
- Implementing dynamic expert addition/removal during training
- Supporting non-LLaMA architectures in initial version

## Decisions

### Architecture Decision: Vectorized Expert Dispatch

**Decision**: Implement fully vectorized expert dispatch/combine mechanism instead of per-sample loops.

**Rationale**:
- Current implementation's `for b in range(batch_size)` loops are training bottlenecks
- Vectorized operations enable GPU parallelism and memory coalescing
- Switch Transformer and GLaM papers demonstrate successful vectorized MoE implementations

**Implementation**:
- Use capacity factor (1.25) to pre-allocate expert computation buffers
- Implement batch-wise token dispatching with expert-specific tensor slicing
- Add overflow handling when expert capacity is exceeded

**Alternatives considered**:
- **Grouped convolution approach**: More complex implementation, unclear performance benefits
- **Hybrid batching**: Partial vectorization, still contains sequential bottlenecks

### Architecture Decision: Per-Layer Cultural Injection

**Decision**: Use FiLM (Feature-wise Linear Modulation) for cultural injection in every layer instead of single-layer embedding.

**Rationale**:
- Different layers learn different levels of abstraction; cultural context should be available at all levels
- FiLM has proven effectiveness in conditional generation tasks
- Low-rank decomposition keeps parameter overhead manageable
- Layer-specific cultural embeddings allow specialized cultural feature learning

**Implementation**:
- Each layer has its own cultural embedding table (6 cultures × 256 dimensions)
- Low-rank FiLM networks: culture_dim → film_rank → hidden_dim
- Gated modulation to control cultural influence strength per layer

**Alternatives considered**:
- **Cross-attention mechanism**: Higher computational cost, more complex implementation
- **Adapter layers**: Less integrated, potential for gradient flow issues
- **Single global embedding**: Insufficient for multi-level cultural understanding

### Training Decision: Progressive Layer Training

**Decision**: Implement three-stage progressive training (shallow → middle → all layers).

**Rationale**:
- Training all MoE layers simultaneously can be unstable due to complex loss landscape
- Progressive training allows each stage to stabilize before adding complexity
- Reduces memory pressure during initial training phases
- Enables better hyperparameter tuning for each phase

**Implementation**:
- Stage 1: Train first 1/3 of layers (epochs 1-2)
- Stage 2: Train first 2/3 of layers (epochs 3-4)
- Stage 3: Train all layers (epochs 5-8)
- Different learning rates for router vs. expert parameters

**Alternatives considered**:
- **End-to-end training**: Risk of training instability and route collapse
- **Layer-wise sequential training**: Too slow, doesn't leverage inter-layer dependencies
- **Random layer sampling**: Unpredictable convergence, difficult to tune

### Memory Decision: Multi-Strategy Optimization

**Decision**: Combine gradient checkpointing, mixed precision, expert parallelism, and capacity control.

**Rationale**:
- Single optimization strategy insufficient for multi-layer MoE memory requirements
- Different strategies target different memory bottlenecks (activations, weights, gradients)
- Allows scaling to larger models and batch sizes

**Implementation**:
- Gradient checkpointing for activation memory
- FP16/BF16 for weight and gradient memory
- Expert parallelism for weight distribution across devices
- Capacity factor tuning to control expert computation memory

**Alternatives considered**:
- **Model parallelism only**: Insufficient for activation memory
- **Offloading strategies**: Too slow for training throughput requirements
- **Reduced model size**: Contradicts goal of improving model capability

### Interface Decision: Explicit Culture ID Propagation

**Decision**: Modify model forward pass to explicitly accept and propagate `culture_ids` parameter.

**Rationale**:
- Explicit is better than implicit for debugging and understanding
- Avoids global state or context managers that complicate distributed training
- Allows for batch-wise cultural variation
- Clear interface contract for users

**Implementation**:
- Add `culture_ids: Optional[torch.LongTensor]` to all relevant forward methods
- Provide default values (zeros) when not specified
- Validate culture_ids shape and range during forward pass

**Alternatives considered**:
- **Global context manager**: Complex interaction with distributed training
- **Model state variables**: Not thread-safe, unclear lifecycle management
- **Embedding in input_ids**: Reduces sequence length, complicates tokenization

## Risks / Trade-offs

### Risk: Memory Usage Increase
**Impact**: High - Multi-layer MoE significantly increases memory requirements
**Mitigation**:
- Progressive training reduces peak memory during early stages
- Memory optimization strategies (checkpointing, mixed precision, parallelism)
- Capacity factor tuning to control expert computation overhead
- Monitoring and alerting for memory usage during training

### Risk: Training Complexity
**Impact**: Medium - More hyperparameters and training phases to manage
**Mitigation**:
- Comprehensive configuration validation and recommended defaults
- Automated hyperparameter search for key MoE parameters
- Clear documentation and training guides
- Progressive training strategy reduces complexity per phase

### Risk: Route Collapse Persistence
**Impact**: High - New architecture might still suffer from expert underutilization
**Mitigation**:
- Multiple regularization strategies (load balance + entropy + noise injection)
- Real-time expert utilization monitoring during training
- Automatic detection and intervention when route collapse detected
- Ablation studies to validate effectiveness of each regularization method

### Risk: Performance Regression
**Impact**: Medium - Vectorization might not fully compensate for increased computation
**Mitigation**:
- Comprehensive benchmarking against current implementation
- Performance profiling to identify and optimize bottlenecks
- Gradual rollout with performance monitoring
- Fallback to current architecture if performance targets not met

### Trade-off: Model Complexity vs. Cultural Understanding
**Decision**: Accept increased complexity for better cultural integration
**Justification**: Current route collapse issues indicate that architectural improvements are necessary for the project's core goals

### Trade-off: Memory Usage vs. Model Capability
**Decision**: Use memory optimization strategies rather than reducing model size
**Justification**: Cultural understanding requires sufficient model capacity; optimization strategies can address memory constraints

## Migration Plan

### Phase 1: Parallel Implementation (Weeks 1-4)
- Implement new architecture alongside existing code
- Create comprehensive test suite for new components
- Validate individual component functionality
- No disruption to current training pipelines

### Phase 2: Integration Testing (Weeks 5-6)
- End-to-end testing with small-scale training runs
- Performance benchmarking and optimization
- Memory usage profiling and optimization
- Hyperparameter tuning for new architecture

### Phase 3: Pilot Training (Weeks 7-8)
- Full-scale training run with new architecture
- Cultural evaluation and comparison with baseline
- Expert utilization analysis and route collapse monitoring
- Documentation and training guide creation

### Phase 4: Production Rollout (Weeks 9-10)
- Update all training scripts to use new architecture
- Migrate configuration and hyperparameter settings
- Archive old implementation as fallback
- Team training on new architecture and tools

### Rollback Plan
- Maintain current implementation until new architecture is fully validated
- Automated performance regression detection
- Quick rollback capability if critical issues discovered
- Clear rollback procedures documented and tested

## Open Questions

1. **Expert Capacity Tuning**: What capacity factor works best for different model sizes and batch sizes?
   - **Investigation needed**: Systematic study of capacity factor vs. performance trade-offs
   - **Timeline**: Week 2-3 during component testing

2. **Cultural Embedding Dimensionality**: Is 256 dimensions optimal for cultural representation across all layers?
   - **Investigation needed**: Ablation study on cultural embedding size vs. model performance
   - **Timeline**: Week 5-6 during integration testing

3. **Progressive Training Schedule**: Are the proposed epoch distributions (2-2-4) optimal?
   - **Investigation needed**: Experiment with different training phase lengths and transitions
   - **Timeline**: Week 7 during pilot training

4. **Expert Parallelism Strategy**: How should experts be distributed across multiple GPUs for optimal performance?
   - **Investigation needed**: Performance analysis of different expert placement strategies
   - **Timeline**: Week 4-5 during memory optimization implementation

5. **Load Balancing Weights**: What combination of load balance, entropy, and culture loss weights works best?
   - **Investigation needed**: Grid search or Bayesian optimization of loss component weights
   - **Timeline**: Week 6-7 during hyperparameter tuning