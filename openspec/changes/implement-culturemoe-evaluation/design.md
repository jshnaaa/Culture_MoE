# Design Document: CultureMoE Evaluation System

## Architecture Overview

The CultureMoE evaluation system is designed as a standalone component that reconstructs trained models from saved parameters and evaluates them on test datasets. The system follows the project's modular architecture and maintains compatibility with existing training outputs.

## System Components

### 1. Model Reconstruction Pipeline

```
Base Model → LoRA Integration → MoE Integration → Complete CultureMoE Model
     ↓              ↓                 ↓                    ↓
Load from      Load from         Load from        Ready for
base path    lora_weights/    moe_weights.pt      evaluation
```

**Design Rationale**: The reconstruction pipeline mirrors the training process in reverse, ensuring compatibility with all training configurations while maintaining the "single source of truth" principle established in the training code.

### 2. Data Loading Architecture

```
Training Output Directory
├── config.json (training configuration)
├── data_split_8_1_1.pkl (dataset splits)
├── best_simplified_culturemoe/
│   ├── lora_weights/ (LoRA parameters)
│   ├── moe_weights.pt (MoE parameters)
│   └── simplified_culturemoe_config.json (model config)
└── training.log (training history)
```

**Design Rationale**: Reuse existing file structures and formats to ensure seamless integration with training outputs. The evaluation system reads the same split files used during training to guarantee test set integrity.

### 3. Evaluation Framework

```
Model + Test Data → Batch Processing → Metric Calculation → Report Generation
       ↓                   ↓                 ↓                    ↓
   Tokenization      Memory-efficient    Multi-metric        Formatted
   & Formatting       GPU utilization     evaluation          output
```

## Key Design Decisions

### 1. Model Loading Strategy

**Decision**: Use composition over inheritance for model reconstruction
**Rationale**:
- Maintains compatibility with existing SimplifiedCultureMoEAdapter
- Allows independent testing of each loading component
- Reduces risk of breaking existing training functionality

**Implementation**:
```python
class CultureMoEEvaluator:
    def __init__(self, model_dir, base_model_path):
        self.config = self._load_config(model_dir)
        self.base_model = self._load_base_model(base_model_path)
        self.model_adapter = self._reconstruct_culturemoe_model()
```

### 2. Memory Management

**Decision**: Implement progressive memory optimization
**Rationale**:
- Large models may exceed GPU memory during evaluation
- Evaluation doesn't require gradient computation
- Batch processing can reduce memory footprint

**Implementation**:
- Use `torch.no_grad()` context for all evaluation
- Implement configurable batch sizes
- Add memory monitoring and garbage collection

### 3. Metric Calculation

**Decision**: Separate metric calculation from model inference
**Rationale**:
- Enables testing of metric calculations independently
- Allows for different evaluation modes (classification vs generation)
- Facilitates addition of new metrics without modifying core evaluation

**Implementation**:
```python
class EvaluationMetrics:
    @staticmethod
    def calculate_classification_metrics(predictions, labels):
        # Accuracy, Precision, Recall, F1-score
        pass

    @staticmethod
    def calculate_generation_metrics(generated_texts, reference_texts):
        # Answer extraction and comparison
        pass
```

## Interface Design

### 1. Shell Script Interface

```bash
run_eval_simplified_culturemoe.sh <model_output_dir> <base_model_path> [options]
```

**Parameters**:
- `model_output_dir`: Directory containing trained model (required)
- `base_model_path`: Path to base model (LLaMA/Qwen) (required)
- `--batch_size`: Evaluation batch size (optional, default: 4)
- `--device`: GPU device to use (optional, default: auto-detect)
- `--output_file`: Custom output file path (optional)

**Design Rationale**: Simple interface matching training script conventions, with sensible defaults and optional parameters for advanced users.

### 2. Python API

```python
evaluator = CultureMoEEvaluator(model_dir, base_model_path)
results = evaluator.evaluate(test_dataset, metrics=['accuracy', 'f1', 'precision', 'recall'])
evaluator.save_results(results, output_path)
```

**Design Rationale**: Clean API for programmatic use, supporting both simple and advanced evaluation scenarios.

## Error Handling Strategy

### 1. Validation Hierarchy

```
Input Validation → Configuration Validation → Model Loading Validation → Runtime Validation
       ↓                    ↓                        ↓                      ↓
   Check paths        Check config files       Check model weights     Check evaluation
   and permissions    and compatibility        and architecture        progress and results
```

### 2. Error Recovery

**Strategy**: Fail fast with informative error messages
**Implementation**:
- Validate all inputs before starting expensive operations
- Provide specific error messages with suggested solutions
- Log detailed error information for debugging

**Example Error Messages**:
```
❌ Model directory not found: /path/to/model
   Suggestion: Check that training completed successfully and path is correct

❌ Base model configuration mismatch: expected llama, found qwen
   Suggestion: Use correct base model path or retrain with specified backbone

❌ GPU memory insufficient for evaluation batch size
   Suggestion: Reduce batch size or use CPU evaluation
```

## Performance Considerations

### 1. Memory Optimization

**Strategies**:
- Use half-precision (fp16) for evaluation when supported
- Implement gradient checkpointing for large models
- Clear intermediate tensors and call garbage collection
- Support CPU evaluation as fallback

### 2. Computational Efficiency

**Strategies**:
- Batch processing for multiple test examples
- Vectorized metric calculations
- Efficient tensor operations using PyTorch optimizations
- Progress tracking to monitor evaluation speed

## Security and Safety

### 1. Input Validation

**Measures**:
- Validate all file paths and check permissions
- Sanitize configuration parameters
- Limit resource usage (memory, disk space)
- Prevent code injection through configuration files

### 2. Resource Management

**Measures**:
- Set maximum memory usage limits
- Implement timeouts for long-running evaluations
- Clean up temporary files and GPU memory
- Monitor system resources during evaluation

## Testing Strategy

### 1. Unit Testing

**Components**:
- Configuration loading and validation
- Model reconstruction from saved weights
- Dataset loading and preprocessing
- Metric calculation accuracy

### 2. Integration Testing

**Scenarios**:
- End-to-end evaluation with known model outputs
- Compatibility testing with different training configurations
- Error handling with invalid inputs
- Performance testing with large models

### 3. Validation Testing

**Approach**:
- Compare evaluation results with training validation metrics
- Test with multiple model architectures and configurations
- Verify test set isolation (no data leakage)
- Cross-validate metric calculations

## Compatibility Matrix

| Component | LLaMA 3.1-8B | Qwen 2.5-7B | LoRA Only | MoE Only | LoRA + MoE |
|-----------|--------------|-------------|-----------|----------|------------|
| Base Model Loading | ✓ | ✓ | ✓ | ✓ | ✓ |
| LoRA Weight Loading | ✓ | ✓ | ✓ | N/A | ✓ |
| MoE Weight Loading | ✓ | ✓ | N/A | ✓ | ✓ |
| Classification Eval | ✓ | ✓ | ✓ | ✓ | ✓ |
| Generation Eval | ✓ | ✓ | ✓ | ✓ | ✓ |

## Future Extensions

### 1. Batch Evaluation

**Design**: Support for evaluating multiple models simultaneously
**Interface**: `run_batch_eval_culturemoe.sh <models_dir> <base_model_path>`
**Benefits**: Efficient comparison of different training configurations

### 2. Visualization Support

**Design**: Generate evaluation plots and charts
**Components**: Metric visualization, confusion matrices, performance comparisons
**Integration**: Optional dependency with graceful fallback

### 3. Export Capabilities

**Design**: Export results in multiple formats (JSON, CSV, LaTeX tables)
**Use Cases**: Research paper generation, automated reporting, data analysis
**Implementation**: Pluggable export modules

## Maintenance Considerations

### 1. Version Compatibility

**Strategy**: Maintain compatibility with training pipeline changes
**Implementation**: Version checks in configuration files, migration utilities for old formats

### 2. Code Maintenance

**Strategy**: Keep evaluation code synchronized with training improvements
**Implementation**: Shared utility modules, consistent coding patterns, regular compatibility testing

### 3. Documentation

**Strategy**: Maintain comprehensive documentation for users and developers
**Components**: Usage examples, troubleshooting guides, API documentation, design rationale