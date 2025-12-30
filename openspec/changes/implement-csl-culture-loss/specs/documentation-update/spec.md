# Documentation Update Specification

## MODIFIED Requirements

### Requirement: CultureMoE Architecture Documentation
The system documentation SHALL be updated to reflect the joint training model architecture and CSL loss function design.

#### Scenario: Joint Training Model Architecture
**Given** `culturemoe.md` documentation file
**When** updating architecture descriptions
**Then** the documentation SHALL describe joint LoRA+MoE training approach
**And** SHALL explain simultaneous optimization of base LoRA and MoE expert layers
**And** SHALL document the relationship between shared experts and router experts
**And** SHALL include architectural diagrams and component descriptions

#### Scenario: CSL Loss Function Documentation
**Given** `culturemoe.md` documentation file
**When** documenting loss function design
**Then** the documentation SHALL include mathematical formulation of CSL
**And** SHALL describe the three CSL components:
- L_culture_router for router expert cultural specialization
- L_culture_share for shared expert culture-invariance
- L_culture_sr for shared-router representation decoupling
**And** SHALL explain the design objectives for each component
**And** SHALL include mathematical notation and formulas

#### Scenario: Model Component Documentation
**Given** `culturemoe.md` documentation file
**When** describing model components
**Then** the documentation SHALL describe JointLoRAMoEModel architecture
**And** SHALL explain the integration of LoRA adapters with MoE layers
**And** SHALL document expert routing mechanisms and weight computation
**And** SHALL describe shared expert functionality and culture-invariant objectives
**And** SHALL explain the interaction between router experts and shared experts

#### Scenario: Training Pipeline Documentation
**Given** `culturemoe.md` documentation file
**When** documenting training procedures
**Then** the documentation SHALL describe joint training methodology
**And** SHALL explain differential learning rates for LoRA and MoE components
**And** SHALL document CSL loss integration and parameter configuration
**And** SHALL include training script usage examples and parameter guidance
**And** SHALL describe the relationship to existing simplified CultureMoE approach

### Requirement: Loss Function Mathematical Formulation
The documentation SHALL include comprehensive mathematical descriptions of the CSL loss function.

#### Scenario: Mathematical Notation
**Given** CSL loss function documentation
**When** describing mathematical formulation
**Then** the documentation SHALL define all symbols and notation:
- B: batch size
- i, j: sample indices
- cul_i: culture label for sample i
- wr_i: router expert weight vector for sample i (K-dimensional)
- es_i: shared expert output vector for sample i
- er_i: router expert fusion output vector for sample i
- sim(x, y): cosine similarity function
**And** SHALL use consistent mathematical notation throughout

#### Scenario: Loss Component Formulas
**Given** CSL loss function documentation
**When** presenting mathematical formulas
**Then** the documentation SHALL include precise formulas for each component:
- L_culture_router with pairwise similarity calculations
- L_culture_share with culture-invariant objectives
- L_culture_sr with decoupling mechanisms
**And** SHALL explain the averaging and normalization procedures
**And** SHALL describe the integration into total loss computation

#### Scenario: Design Rationale
**Given** CSL loss function documentation
**When** explaining design decisions
**Then** the documentation SHALL explain the motivation for three-component design
**And** SHALL describe how CSL addresses limitations of previous approaches
**And** SHALL explain the relationship between cultural specialization and representation learning
**And** SHALL discuss the balance between specialization and generalization

### Requirement: Configuration and Usage Documentation
The documentation SHALL provide clear guidance on CSL configuration and usage.

#### Scenario: Parameter Configuration
**Given** training configuration documentation
**When** describing CSL parameters
**Then** the documentation SHALL document `USE_CULTURE_LOSS=csl` option
**And** SHALL explain the default parameter change from `new` to `csl`
**And** SHALL describe the relationship to other culture loss modes
**And** SHALL provide parameter selection guidance for different use cases

#### Scenario: Model Requirements
**Given** CSL usage documentation
**When** describing system requirements
**Then** the documentation SHALL specify model output requirements for CSL
**And** SHALL document the need for expert output extraction capabilities
**And** SHALL explain compatibility with different model configurations
**And** SHALL describe fallback behavior when requirements are not met

#### Scenario: Performance Considerations
**Given** CSL implementation documentation
**When** describing performance characteristics
**Then** the documentation SHALL document computational overhead expectations
**And** SHALL provide guidance on batch size considerations
**And** SHALL describe memory usage implications
**And** SHALL include performance optimization recommendations

## ADDED Requirements

### Requirement: Joint Training vs Simplified Training Comparison
The documentation SHALL clearly distinguish between joint training and simplified training approaches.

#### Scenario: Approach Comparison
**Given** model architecture documentation
**When** comparing training approaches
**Then** the documentation SHALL contrast joint training with simplified training
**And** SHALL explain the advantages of simultaneous LoRA+MoE optimization
**And** SHALL describe when to use each approach
**And** SHALL document the differences in model architecture and capabilities

#### Scenario: Migration Guidance
**Given** existing simplified CultureMoE users
**When** considering joint training adoption
**Then** the documentation SHALL provide migration guidance
**And** SHALL explain compatibility considerations
**And** SHALL describe expected performance differences
**And** SHALL include configuration translation examples

### Requirement: CSL Implementation Examples
The documentation SHALL include practical examples of CSL usage and configuration.

#### Scenario: Configuration Examples
**Given** CSL usage documentation
**When** providing practical examples
**Then** the documentation SHALL include complete training script examples
**And** SHALL show CSL parameter configuration variations
**And** SHALL demonstrate integration with different model backbones
**And** SHALL provide troubleshooting guidance for common issues

#### Scenario: Loss Component Analysis
**Given** CSL training documentation
**When** explaining loss behavior
**Then** the documentation SHALL describe expected loss component patterns
**And** SHALL explain how to interpret CSL loss values during training
**And** SHALL provide guidance on loss component balancing
**And** SHALL include examples of successful training curves