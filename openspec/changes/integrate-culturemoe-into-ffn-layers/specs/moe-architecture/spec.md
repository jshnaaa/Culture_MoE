## ADDED Requirements

### Requirement: Vectorized CultureMoE FFN Layer
The system SHALL implement a VectorizedCultureMoE_FFN class that replaces the standard LlamaMLP layer while maintaining interface compatibility and providing culture-aware expert routing.

#### Scenario: FFN Layer Replacement
- **WHEN** a LlamaDecoderLayer is initialized with culture_config
- **THEN** the mlp attribute SHALL be replaced with VectorizedCultureMoE_FFN
- **AND** the layer SHALL accept culture_ids parameter in forward pass
- **AND** the layer SHALL return both hidden states and auxiliary MoE information

#### Scenario: Vectorized Expert Dispatch
- **WHEN** processing a batch with multiple samples and experts
- **THEN** the system SHALL use vectorized operations for expert selection and routing
- **AND** SHALL NOT use per-sample loops that degrade performance
- **AND** SHALL implement capacity factor control to manage expert load

### Requirement: Cultural Information Injection
The system SHALL implement per-layer cultural information injection using FiLM (Feature-wise Linear Modulation) to enable culture-aware feature transformations at all model depths.

#### Scenario: Layer-Specific Cultural Embedding
- **WHEN** processing input with culture_ids in any decoder layer
- **THEN** the layer SHALL apply layer-specific cultural embeddings
- **AND** SHALL use FiLM modulation to condition hidden states on cultural context
- **AND** SHALL maintain separate cultural parameters for each layer

#### Scenario: Cultural Strength Gating
- **WHEN** applying cultural modulation to hidden states
- **THEN** the system SHALL compute adaptive cultural influence strength
- **AND** SHALL apply gated modulation to control cultural impact per sample
- **AND** SHALL preserve original hidden state information through residual connections

### Requirement: Load Balancing and Route Collapse Prevention
The system SHALL implement comprehensive load balancing mechanisms to prevent expert underutilization and route collapse during training.

#### Scenario: Load Balance Loss Computation
- **WHEN** computing MoE auxiliary losses during training
- **THEN** the system SHALL calculate Switch Transformer style load balancing loss
- **AND** SHALL track expert token assignment ratios (f_i) and router probabilities (P_i)
- **AND** SHALL apply load balancing loss coefficient to encourage expert utilization balance

#### Scenario: Entropy Regularization
- **WHEN** expert weights are computed by the router
- **THEN** the system SHALL apply entropy regularization to encourage diverse expert selection
- **AND** SHALL use numerically stable entropy computation methods
- **AND** SHALL monitor expert weight distribution variance as entropy proxy

#### Scenario: Training Noise Injection
- **WHEN** model is in training mode
- **THEN** the router SHALL inject small amounts of Gaussian noise to router logits
- **AND** SHALL use configurable noise epsilon (default 1e-2)
- **AND** SHALL disable noise during inference/evaluation

### Requirement: Enhanced Cultural Router
The system SHALL implement a vectorized cultural router that combines content-driven, culture-driven, and affinity-based routing decisions.

#### Scenario: Multi-Dimensional Routing
- **WHEN** routing tokens to experts
- **THEN** the router SHALL compute content-based routing scores from hidden states
- **AND** SHALL compute culture-based routing scores from culture embeddings
- **AND** SHALL apply learnable culture-expert affinity matrix
- **AND** SHALL combine all routing dimensions with learnable fusion weights

#### Scenario: Enhanced Pooling
- **WHEN** creating pooled representations for routing decisions
- **THEN** the system SHALL use enhanced pooling combining mean pooling and attention pooling
- **AND** SHALL implement learnable CLS token for attention-based pooling
- **AND** SHALL apply learnable combination weights for different pooling strategies

### Requirement: Expert Capacity Management
The system SHALL implement expert capacity control mechanisms to manage computational resources and prevent memory overflow during expert processing.

#### Scenario: Capacity Factor Control
- **WHEN** dispatching tokens to experts
- **THEN** the system SHALL calculate expert capacity using configurable capacity factor (default 1.25)
- **AND** SHALL pre-allocate expert computation buffers based on capacity
- **AND** SHALL handle token overflow when expert capacity is exceeded

#### Scenario: Top-K Expert Selection
- **WHEN** selecting experts for token processing
- **THEN** the system SHALL select top-k experts (default k=2) based on routing weights
- **AND** SHALL renormalize selected expert weights using softmax
- **AND** SHALL create sparse expert weight matrix for efficient computation

### Requirement: Memory Optimization Integration
The system SHALL provide memory optimization capabilities to enable training of multi-layer MoE models within reasonable hardware constraints.

#### Scenario: Gradient Checkpointing Support
- **WHEN** gradient checkpointing is enabled
- **THEN** the MoE layers SHALL be compatible with gradient checkpointing mechanisms
- **AND** SHALL properly handle intermediate activation recomputation
- **AND** SHALL maintain numerical stability during recomputation

#### Scenario: Mixed Precision Training
- **WHEN** using FP16 or BF16 training
- **THEN** the MoE components SHALL handle mixed precision operations correctly
- **AND** SHALL use FP32 for numerically sensitive operations (softmax, loss computation)
- **AND** SHALL prevent gradient underflow in expert parameters

### Requirement: Auxiliary Loss Integration
The system SHALL collect and propagate auxiliary loss information from all MoE layers to enable proper training with regularization losses.

#### Scenario: Auxiliary Information Collection
- **WHEN** MoE layers complete forward pass
- **THEN** each layer SHALL return auxiliary information dictionary
- **AND** SHALL include expert weights, router logits, and computed losses
- **AND** SHALL propagate auxiliary information through model forward pass

#### Scenario: Multi-Layer Loss Aggregation
- **WHEN** computing total training loss
- **THEN** the system SHALL aggregate auxiliary losses from all MoE layers
- **AND** SHALL apply layer-wise averaging for load balance and entropy losses
- **AND** SHALL combine with language modeling loss using configurable weights