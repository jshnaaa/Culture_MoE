# End-to-End CultureMoE Training Specification

## ADDED Requirements

### Requirement: Joint Parameter Optimization
The system MUST support simultaneous training of base language model and MoE components without parameter freezing.

#### Scenario: Initialize End-to-End Training
- **Given** a pre-trained base model (LLaMA-3.1-8B-Instruct or Qwen-2.5-7B-Instruct)
- **And** MoE architecture configuration (number of experts, hidden dimensions)
- **When** initializing the CultureMoE model for end-to-end training
- **Then** all model parameters must be trainable (no frozen layers)
- **And** MoE components must be initialized with appropriate random weights
- **And** base model parameters must retain their pre-trained values

#### Scenario: Configure Layered Learning Rates
- **Given** a CultureMoE model with base model and MoE components
- **When** setting up the optimizer for end-to-end training
- **Then** base model parameters must use a lower learning rate (1e-6 to 5e-6)
- **And** MoE router parameters must use a medium learning rate (1e-5 to 5e-5)
- **And** expert network parameters must use a higher learning rate (1e-4 to 5e-4)
- **And** shared expert parameters must use a medium learning rate (1e-5 to 5e-5)

### Requirement: Memory Efficient Joint Training
The system MUST handle memory constraints during joint training through optimization techniques.

#### Scenario: Apply Mixed Precision Training
- **Given** a CultureMoE model ready for end-to-end training
- **When** configuring training precision
- **Then** base model layers must use bfloat16 precision
- **And** MoE components must use float32 precision for stability
- **And** gradient scaling must be applied appropriately for mixed precision

#### Scenario: Implement Gradient Accumulation
- **Given** memory constraints during joint training
- **When** training with limited GPU memory
- **Then** gradient accumulation steps must be configurable (default 4-8)
- **And** effective batch size must equal batch_size × accumulation_steps
- **And** gradients must be properly normalized by accumulation steps

#### Scenario: Enable Gradient Checkpointing
- **Given** base model layers consuming significant memory
- **When** memory usage exceeds available GPU memory
- **Then** gradient checkpointing must be enabled for base model layers
- **And** memory usage must be reduced at the cost of computation time
- **And** gradient computation must remain mathematically correct

### Requirement: Training Stability Management
The system MUST ensure stable convergence during joint optimization.

#### Scenario: Implement Cultural Loss Warmup
- **Given** a joint training setup with cultural loss components
- **When** starting training from epoch 1
- **Then** epochs 1-5 must use only language modeling loss (warmup_factor = 0.0)
- **And** epochs 6-10 must gradually increase cultural loss weight (warmup_factor = 0.0 to 1.0)
- **And** epochs 11+ must use full cultural loss weight (warmup_factor = 1.0)
- **And** total loss must be: language_loss + warmup_factor × culture_loss_lambda × culture_loss

#### Scenario: Apply Gradient Clipping
- **Given** joint training with potential gradient instability
- **When** computing gradients for parameter updates
- **Then** gradient norms must be clipped to maximum value 1.0
- **And** clipping must be applied to all parameter groups
- **And** gradient clipping must prevent training divergence

#### Scenario: Schedule Router Temperature
- **Given** MoE router prone to expert collapse
- **When** training progresses through epochs
- **Then** router temperature must start at 5.0 in early epochs
- **And** temperature must gradually decrease to 2.0 over training
- **And** temperature scheduling must prevent expert collapse

### Requirement: Comprehensive Training Monitoring
The system MUST provide detailed monitoring of joint training progress.

#### Scenario: Track Component-Specific Metrics
- **Given** end-to-end training with multiple parameter groups
- **When** monitoring training progress
- **Then** gradient norms must be tracked separately for base model and MoE components
- **And** learning rates must be logged for each parameter group
- **And** loss components must be tracked individually (language, cultural, specialization, diversity)
- **And** router weight distributions must be monitored for collapse detection

#### Scenario: Detect Training Instabilities
- **Given** joint training susceptible to numerical issues
- **When** processing each training batch
- **Then** NaN or infinite losses must be detected immediately
- **And** problematic batches must be skipped with logging
- **And** gradient explosion must be detected via gradient norms
- **And** training must continue gracefully after instability events

### Requirement: Configuration and Backwards Compatibility
The system MUST support flexible configuration while maintaining compatibility.

#### Scenario: Configure End-to-End Training Parameters
- **Given** a training configuration for end-to-end approach
- **When** parsing command line arguments or configuration files
- **Then** end-to-end mode must be selectable via `--training_mode end_to_end`
- **And** layered learning rates must be configurable per component type
- **And** warmup schedule parameters must be configurable
- **And** memory optimization settings must be configurable

#### Scenario: Maintain Backwards Compatibility
- **Given** existing two-stage training workflows
- **When** using the updated training system
- **Then** two-stage training mode must remain available via `--training_mode two_stage`
- **And** existing configuration files must continue to work
- **And** existing shell scripts must run without modification
- **And** model output formats must remain compatible

## MODIFIED Requirements

### Requirement: Model Parameter Management
The system MUST update parameter management to support joint training instead of selective freezing.

#### Scenario: Initialize Trainable Parameters
- **Given** a CultureMoE model for training
- **When** setting up parameter optimization
- **Then** all model parameters must be trainable by default
- **And** parameter groups must be created based on component type
- **And** no parameters must be frozen unless explicitly configured
- **And** parameter counting must reflect all trainable parameters

### Requirement: Training Loop Integration
The system MUST modify the training loop to handle joint optimization requirements.

#### Scenario: Execute Joint Training Step
- **Given** a training batch and joint optimization setup
- **When** performing a training step
- **Then** forward pass must compute all loss components
- **And** backward pass must compute gradients for all parameter groups
- **And** optimizer step must update all parameter groups with respective learning rates
- **And** gradient accumulation must work correctly across all parameter groups

## REMOVED Requirements

### Requirement: LoRA Weight Loading and Merging
Remove dependency on pre-trained LoRA weights for CultureMoE training.

#### Scenario: Skip LoRA Pre-training Stage
- **Given** end-to-end training configuration
- **When** initializing the CultureMoE model
- **Then** LoRA weight loading must be skipped
- **And** LoRA merging operations must be bypassed
- **And** model must start directly from base pre-trained weights

### Requirement: Selective Parameter Freezing
Remove parameter freezing mechanisms that prevent joint optimization.

#### Scenario: Eliminate Parameter Freezing
- **Given** a CultureMoE model ready for training
- **When** setting up the training configuration
- **Then** base model parameter freezing must be disabled
- **And** selective parameter unfreezing logic must be removed
- **And** all parameters must participate in gradient computation