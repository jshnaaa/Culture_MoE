# Change: Integrate CultureMoE into Transformer FFN Layers

## Why

The current Enhanced CultureMoE implementation operates as an independent layer on top of the frozen backbone model, which limits its effectiveness in several ways:

1. **Shallow Cultural Integration**: Cultural awareness is only applied at the final layer, missing opportunities for deeper cultural understanding throughout the model
2. **Limited Feature Learning**: Cultural information doesn't participate in intermediate feature transformations, reducing the model's ability to learn culture-specific representations
3. **Suboptimal Performance**: Independent MoE layers cannot leverage the rich intermediate representations from the backbone model's FFN layers
4. **Route Collapse Issues**: Current implementation suffers from expert underutilization (load balance loss: 0.458, entropy loss: 0.194) as documented in `ablation.md`

## What Changes

- **Replace LLaMA FFN layers** with vectorized CultureMoE_FFN layers that maintain the same input/output interface
- **Implement per-layer cultural injection** using FiLM (Feature-wise Linear Modulation) instead of single-layer cultural embedding
- **Add vectorized expert dispatch/combine** mechanism to eliminate the current inefficient per-sample loops
- **Integrate load balancing and entropy regularization** to prevent route collapse
- **Support explicit culture_ids propagation** through the model's forward pass
- **Add progressive training strategy** to manage training complexity across multiple layers
- **Implement memory optimization** with gradient checkpointing, mixed precision, and expert parallelism

### Technical Architecture Changes

1. **VectorizedCultureMoE_FFN**: Replace each `LlamaMLP` with culture-aware MoE that includes:
   - Cultural injector with FiLM modulation per layer
   - Vectorized cultural router with enhanced pooling
   - Batch-processed expert computation
   - Load balancing and entropy losses

2. **Cultural Information Flow**: Modify model forward pass to:
   - Accept `culture_ids` parameter explicitly
   - Propagate cultural context through all decoder layers
   - Apply layer-specific cultural transformations

3. **Training Infrastructure**: Add:
   - Progressive training strategy (shallow → middle → all layers)
   - Specialized optimizers with layer-wise learning rates
   - Memory optimization strategies
   - Enhanced loss computation with MoE auxiliary losses

## Impact

### Affected Specs
- `moe-architecture`: New FFN-integrated MoE architecture
- `training-pipeline`: Enhanced training procedures with progressive strategy

### Affected Code
- `src/llamafactory/model/enhanced_culturemoe.py`: Core model architecture
- `src/llamafactory/model/cultural_components.py`: Cultural awareness components
- `ft_enhanced_culturemoe_gen.py`: Training script modifications
- Model forward pass in LLaMA decoder layers

### Performance Impact
- **Memory Usage**: Increased due to multiple expert layers (mitigated by optimization strategies)
- **Training Speed**: Initially slower due to more parameters (improved by vectorization)
- **Model Quality**: Expected significant improvement in cultural understanding and reduced route collapse
- **Inference Speed**: Comparable to current implementation due to Top-K sparse activation

### Breaking Changes
- **BREAKING**: Model architecture changes require retraining from scratch
- **BREAKING**: Forward pass interface modified to accept `culture_ids`
- **BREAKING**: Checkpoint format incompatible with current implementation

### Migration Path
1. Save current model evaluation results for comparison
2. Implement new architecture in parallel
3. Train new models using progressive strategy
4. Validate performance improvements before deprecating old architecture
5. Update all training and evaluation scripts to use new interface