# CSL Culture Loss Function Specification

## ADDED Requirements

### Requirement: CSL Loss Function Implementation
The system SHALL implement a Culture Similarity Loss (CSL) function that computes three complementary loss components when `USE_CULTURE_LOSS=csl` is specified.

#### Scenario: CSL Loss Computation
**Given** a batch of samples with culture labels and expert outputs
**When** `USE_CULTURE_LOSS=csl` is configured
**Then** the system SHALL compute `L_culture = L_culture_router + L_culture_share + L_culture_sr`
**And** each component SHALL contribute meaningful gradients to model optimization

#### Scenario: Router Expert Culture Similarity Loss
**Given** router expert weight vectors `wr_i` for all samples in batch
**When** computing `L_culture_router`
**Then** for all sample pairs (i,j) where i < j:
- **If** `cul_i == cul_j` **Then** add `(1 - sim(wr_i, wr_j))` to encourage similarity
- **If** `cul_i != cul_j` **Then** add `sim(wr_i, wr_j)` to penalize similarity
**And** the final loss SHALL be averaged over all pairs

#### Scenario: Shared Expert Culture-Invariant Loss
**Given** shared expert output vectors `es_i` for all samples in batch
**When** computing `L_culture_share`
**Then** for all sample pairs (i,j) where i < j:
- Add `(1 - sim(es_i, es_j))` regardless of culture labels
**And** the final loss SHALL be averaged over all pairs
**And** this SHALL force shared expert to learn culture-invariant representations

#### Scenario: Shared-Router Decoupling Loss
**Given** shared expert outputs `es_i` and router expert outputs `er_i` for all samples
**When** computing `L_culture_sr`
**Then** for all samples i:
- Add `sim(es_i, er_i)` to penalize similarity
**And** the final loss SHALL be averaged over the batch
**And** this SHALL encourage complementary representations between shared and router experts

### Requirement: Cosine Similarity Implementation
The system SHALL implement cosine similarity as `sim(x, y) = (x · y) / (||x|| * ||y||)` with proper numerical stability.

#### Scenario: Numerical Stability
**Given** two vectors x and y
**When** computing cosine similarity
**Then** the system SHALL check for zero vectors (norm < 1e-8)
**And** SHALL return zero similarity for zero vector cases
**And** SHALL handle NaN and Inf values gracefully

#### Scenario: Batch Vectorization
**Given** large batches requiring similarity computations
**When** computing pairwise similarities
**Then** the system SHALL use vectorized operations where possible
**And** SHALL manage memory efficiently to avoid OOM errors
**And** SHALL maintain gradient flow for all similarity calculations

### Requirement: Expert Output Extraction
The system SHALL extract required expert outputs from JointLoRAMoEModel for CSL computation.

#### Scenario: Router Expert Output Extraction
**Given** model outputs from JointLoRAMoEModel forward pass
**When** CSL loss computation is required
**Then** the system SHALL extract router expert fusion outputs `er_i`
**And** SHALL validate tensor shapes and types
**And** SHALL handle cases where router outputs are unavailable

#### Scenario: Shared Expert Output Extraction
**Given** model outputs from JointLoRAMoEModel forward pass
**When** CSL loss computation is required
**Then** the system SHALL extract shared expert outputs `es_i`
**And** SHALL validate tensor shapes and types
**And** SHALL handle cases where shared expert is not configured

#### Scenario: Router Weight Extraction
**Given** model outputs from JointLoRAMoEModel forward pass
**When** CSL loss computation is required
**Then** the system SHALL extract router weight vectors `wr_i` from softmax outputs
**And** SHALL ensure weights are K-dimensional vectors
**And** SHALL validate weight normalization properties

### Requirement: Training Pipeline Integration
The system SHALL integrate CSL loss into the joint training pipeline when configured.

#### Scenario: CSL Activation
**Given** training configuration with `USE_CULTURE_LOSS=csl`
**When** training epoch is executed
**Then** the system SHALL use CSL loss instead of existing culture loss
**And** SHALL maintain backward compatibility with other loss modes
**And** SHALL preserve all existing training functionality

#### Scenario: Loss Component Logging
**Given** CSL loss computation during training
**When** logging training metrics
**Then** the system SHALL log individual CSL loss components
- `L_culture_router` value
- `L_culture_share` value
- `L_culture_sr` value
- Total `L_culture` value
**And** SHALL include CSL components in training progress displays

#### Scenario: Configuration Validation
**Given** training script execution
**When** `USE_CULTURE_LOSS=csl` is specified
**Then** the system SHALL validate model provides required expert outputs
**And** SHALL validate culture labels are available
**And** SHALL provide clear error messages for configuration issues

### Requirement: Performance Optimization
The system SHALL implement CSL loss computation efficiently to minimize training overhead.

#### Scenario: Computational Overhead
**Given** CSL loss computation during training
**When** measuring training performance
**Then** CSL overhead SHALL be less than 10% of total training time
**And** SHALL scale reasonably with batch size increases
**And** SHALL maintain training throughput within acceptable bounds

#### Scenario: Memory Management
**Given** large batches and long sequences
**When** computing CSL loss
**Then** the system SHALL manage memory efficiently
**And** SHALL avoid storing unnecessary intermediate tensors
**And** SHALL handle memory constraints gracefully

#### Scenario: Gradient Flow Preservation
**Given** CSL loss components
**When** backpropagation is performed
**Then** the system SHALL preserve gradient flow through all similarity calculations
**And** SHALL ensure all three loss components contribute meaningful gradients
**And** SHALL maintain gradient stability during optimization

## MODIFIED Requirements

### Requirement: Culture Loss Configuration
The system SHALL support `csl` as a new option for `USE_CULTURE_LOSS` parameter.

#### Scenario: Default Configuration Update
**Given** `run_joint_lora_moe_training.sh` script
**When** no `USE_CULTURE_LOSS` parameter is specified
**Then** the system SHALL default to `USE_CULTURE_LOSS=csl`
**And** SHALL maintain support for existing options (`ori`, `new`, `kl`, `false`)

#### Scenario: Configuration Validation
**Given** training script execution
**When** `USE_CULTURE_LOSS` parameter is provided
**Then** the system SHALL validate the parameter value
**And** SHALL accept `csl` as a valid option
**And** SHALL provide clear error messages for invalid options

### Requirement: Total Loss Computation
The system SHALL maintain the existing total loss formulation while using CSL for culture component.

#### Scenario: Loss Formulation Preservation
**Given** CSL culture loss implementation
**When** computing total training loss
**Then** the system SHALL maintain `L_total = L_generation + λ × (α × L_aux + β × L_culture)`
**And** SHALL use CSL result as `L_culture` when `USE_CULTURE_LOSS=csl`
**And** SHALL preserve existing loss weighting mechanisms