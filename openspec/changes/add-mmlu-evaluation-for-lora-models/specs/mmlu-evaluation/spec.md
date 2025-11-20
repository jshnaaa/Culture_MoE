# MMLU Evaluation Specification

## ADDED Requirements

### Requirement: LoRA Model MMLU Evaluation
The system SHALL provide MMLU (Massive Multitask Language Understanding) evaluation capability for LoRA-only trained models without requiring permanent storage of merged models.

#### Scenario: Evaluate LoRA model on MMLU
- **WHEN** user provides base model path and LoRA adapter path
- **THEN** system loads both components, merges in memory, and evaluates on MMLU benchmark
- **AND** returns per-subject scores (STEM, Social Sciences, Humanities, Other) and overall accuracy

#### Scenario: Memory-efficient evaluation
- **WHEN** evaluation is performed on large models (8B+ parameters)
- **THEN** system merges LoRA weights in memory without saving to disk
- **AND** evaluation completes within reasonable memory constraints

### Requirement: LLaMA-Factory Integration
The system SHALL integrate with existing LLaMA-Factory evaluation infrastructure for MMLU assessment.

#### Scenario: Use existing evaluator
- **WHEN** MMLU evaluation is requested
- **THEN** system uses `src/llamafactory/eval/evaluator.py` with proper configuration
- **AND** supports standard MMLU tasks (mmlu_test) with few-shot prompting

#### Scenario: Configuration compatibility
- **WHEN** evaluation configuration is provided
- **THEN** system accepts standard LLaMA-Factory YAML configuration format
- **AND** supports parameters: model_name_or_path, adapter_name_or_path, task, batch_size, n_shot

### Requirement: Multi-Architecture Support
The system SHALL support MMLU evaluation for both LLaMA and Qwen model architectures used in Culture_Moe.

#### Scenario: LLaMA model evaluation
- **WHEN** LLaMA 3.1-8B-Instruct base model with LoRA adapters is evaluated
- **THEN** system properly loads model and tokenizer
- **AND** generates correct MMLU responses with A/B/C/D choice selection

#### Scenario: Qwen model evaluation
- **WHEN** Qwen 2.5-7B-Instruct base model with LoRA adapters is evaluated
- **THEN** system handles Qwen-specific tokenization and generation
- **AND** applies appropriate generation parameters (max_tokens=3, repetition_penalty=1.2)

### Requirement: Comprehensive Reporting
The system SHALL generate detailed MMLU evaluation reports with subject-specific and overall metrics.

#### Scenario: Subject-specific results
- **WHEN** MMLU evaluation completes
- **THEN** system reports accuracy for each MMLU subject category
- **AND** includes STEM, Social Sciences, Humanities, Other, and Average scores

#### Scenario: Result persistence
- **WHEN** evaluation results are generated
- **THEN** system saves results in JSON format with individual question responses
- **AND** creates summary log with formatted score breakdown

### Requirement: Culture_Moe Integration
The system SHALL integrate seamlessly with existing Culture_Moe model paths and naming conventions.

#### Scenario: Standard model path support
- **WHEN** evaluation uses Culture_Moe trained models
- **THEN** system accepts standard base model paths (e.g., /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct)
- **AND** supports LoRA adapter paths following pattern: ft/ft_lora_only_gen_{dataset}_{backbone}_{timestamp}/best_lora/

#### Scenario: Execution script integration
- **WHEN** MMLU evaluation is executed
- **THEN** system provides shell script wrapper similar to existing evaluation scripts
- **AND** supports parameter configuration for different datasets and backbones

### Requirement: Performance Optimization
The system SHALL optimize MMLU evaluation performance for efficient resource usage.

#### Scenario: Batch processing
- **WHEN** MMLU evaluation processes multiple questions
- **THEN** system uses configurable batch sizes for efficient GPU utilization
- **AND** supports batch_size parameter (default: 4 for most models)

#### Scenario: Inference optimization
- **WHEN** model generates responses
- **THEN** system uses torch.inference_mode() for memory efficiency
- **AND** applies proper attention mask handling for batched inference