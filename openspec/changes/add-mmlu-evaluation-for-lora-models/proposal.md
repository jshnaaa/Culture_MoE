# Add MMLU Evaluation for LoRA Models

## Summary
Implement MMLU (Massive Multitask Language Understanding) evaluation capability for LoRA-only trained models using the existing LLaMA-Factory evaluation infrastructure. This allows comprehensive academic benchmarking of cultural fine-tuned models on standardized tasks.

## Problem Statement
Currently, the Culture_Moe project focuses on cultural classification tasks but lacks standardized academic benchmark evaluation. MMLU is a critical benchmark for measuring general knowledge and reasoning capabilities across diverse domains (STEM, Social Sciences, Humanities, etc.).

The challenge is that LoRA-only models exist as separate components:
- Base model (e.g., LLaMA 3.1-8B-Instruct, Qwen 2.5-7B-Instruct)
- LoRA adapter weights (trained separately)
- Need temporary merging without saving full merged models

## Proposed Solution
Leverage LLaMA-Factory's built-in MMLU evaluation infrastructure to create a streamlined evaluation pipeline that:

1. **Dynamically loads and merges** base model + LoRA weights in memory
2. **Evaluates on MMLU** using LLaMA-Factory's evaluator with proper few-shot prompting
3. **Reports comprehensive metrics** across MMLU subjects (STEM, Social Sciences, Humanities, Other)
4. **Integrates with existing model paths** used in Culture_Moe training scripts
5. **Supports both LLaMA and Qwen** model architectures
6. **Avoids saving merged models** to disk (memory-efficient approach)

## Benefits
- **Academic Benchmarking**: Enables comparison with other models on standardized MMLU tasks
- **Quality Assessment**: Measures whether cultural fine-tuning preserves general knowledge
- **Research Validation**: Provides evidence for paper publications and research claims
- **Efficient Evaluation**: Uses existing LLaMA-Factory infrastructure (no reinvention)
- **Memory Efficient**: No need to save large merged models permanently
- **Comprehensive Coverage**: Evaluates across multiple knowledge domains

## Implementation Scope
This change affects:
- **Evaluation Scripts**: New MMLU evaluation script for LoRA models
- **Model Loading**: Integration with existing LoRA merging patterns
- **Configuration**: YAML configuration files for MMLU evaluation
- **Documentation**: Usage instructions and parameter explanations

## Success Criteria
- MMLU evaluation script successfully loads LoRA-only models (base + adapter)
- Evaluation runs on all MMLU subjects with proper few-shot prompting
- Results include per-subject accuracy and overall MMLU score
- Memory usage remains reasonable (no permanent model saving required)
- Integration works with existing Culture_Moe model paths and naming conventions
- Supports both LLaMA 3.1-8B and Qwen 2.5-7B model families