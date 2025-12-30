# CultureMoE Evaluation System Specification

## ADDED Requirements

### Requirement: Complete model reconstruction from training artifacts
The evaluation system SHALL reconstruct complete CultureMoE models from saved training parameters, including base model, LoRA weights, and MoE weights.

#### Scenario: Loading trained CultureMoE model
```bash
# Given: A training output directory with complete model artifacts
MODEL_DIR="/path/to/best_simplified_culturemoe"
BASE_MODEL="/path/to/llama-3.1-8b-instruct"

# When: Loading the model for evaluation
./run_eval_simplified_culturemoe.sh $MODEL_DIR $BASE_MODEL

# Then: System successfully reconstructs complete model
# - Loads SimplifiedCultureMoEConfig from simplified_culturemoe_config.json
# - Loads base model from specified path
# - Applies LoRA weights from lora_weights/ directory
# - Applies MoE weights from moe_weights.pt file
# - Model ready for evaluation with all parameters loaded
```

#### Scenario: Handling missing model components
```bash
# Given: Incomplete model directory missing MoE weights
MODEL_DIR="/path/to/incomplete_model"

# When: Attempting to load model
./run_eval_simplified_culturemoe.sh $MODEL_DIR $BASE_MODEL

# Then: System provides clear error message
# - Identifies missing moe_weights.pt file
# - Suggests checking training completion
# - Exits gracefully with error code
```

### Requirement: Proper test set loading from training splits
The evaluation system SHALL load test sets using the exact same 8:1:1 splits created during training to ensure test set isolation.

#### Scenario: Loading test dataset from split file
```python
# Given: Training output with data split file
split_file = "/path/to/data_split_8_1_1.pkl"

# When: Loading test dataset
test_loader, num_samples = evaluator.load_test_dataset(split_file)

# Then: System loads correct test set
# - Reads test indices from pickle file
# - Loads original dataset from stored path
# - Creates test subset using exact indices from training
# - Returns DataLoader with correct test samples
assert len(test_loader.dataset) == num_samples
```

#### Scenario: Validating dataset consistency
```python
# Given: Split file with metadata
split_info = {
    'test': [100, 200, 300],  # Test indices
    'original_data_path': '/path/to/data.json',
    'max_length': 512
}

# When: Loading dataset with different original file
# (original file moved or renamed)

# Then: System detects inconsistency
# - Raises FileNotFoundError with clear message
# - Indicates original data file location
# - Suggests checking file paths or training output integrity
```

### Requirement: Multi-metric evaluation for classification tasks
The evaluation system SHALL provide comprehensive metrics including accuracy, precision, recall, F1-score, and detailed classification reports.

#### Scenario: Classification task evaluation
```python
# Given: Trained CultureMoE model and test dataset
model = CultureMoEEvaluator(model_dir, base_model_path)
test_loader = model.load_test_dataset(split_file)

# When: Running evaluation
metrics = model.evaluate_classification(test_loader)

# Then: System produces comprehensive metrics
assert 'accuracy' in metrics
assert 'precision' in metrics
assert 'recall' in metrics
assert 'f1_score' in metrics
assert 'classification_report' in metrics
assert 'num_samples' in metrics
assert 0.0 <= metrics['accuracy'] <= 1.0
```

#### Scenario: Generative task evaluation
```python
# Given: Model trained on generative cultural tasks
# Test samples with text generation outputs

# When: Evaluating generative performance
metrics = model.evaluate_classification(test_loader)

# Then: System extracts and compares answers
# - Decodes generated text from model predictions
# - Extracts answers using extract_answer_from_text()
# - Compares against ground truth answers
# - Converts string answers to numeric labels for metric calculation
assert metrics['num_samples'] > 0
```

### Requirement: User-friendly shell script interface
The evaluation system SHALL provide a shell script interface that validates inputs, provides clear error messages, and handles various model configurations.

#### Scenario: Basic evaluation execution
```bash
# Given: Complete training output and base model
MODEL_DIR="/root/autodl-fs/simplified_culturemoe/llama_CulturalBench_20241230_143022"
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"

# When: Running evaluation
./run_eval_simplified_culturemoe.sh $MODEL_DIR $BASE_MODEL

# Then: System completes evaluation successfully
# - Validates all required files exist
# - Loads model and test dataset
# - Runs evaluation with progress reporting
# - Saves results to timestamped JSON file
# - Displays summary of key metrics
# Exit code: 0
```

#### Scenario: Parameter validation and error handling
```bash
# Given: Invalid parameters
# When: Running with missing parameters
./run_eval_simplified_culturemoe.sh

# Then: System shows usage help
# - Displays parameter requirements
# - Shows example usage
# - Lists optional parameters
# Exit code: 1

# When: Running with non-existent model directory
./run_eval_simplified_culturemoe.sh /invalid/path /valid/base/model

# Then: System validates inputs
# - Checks model directory exists
# - Checks required files present
# - Provides specific error messages
# Exit code: 1
```

#### Scenario: Memory optimization support
```bash
# Given: Large model requiring memory optimization
# When: Running with memory efficient mode
./run_eval_simplified_culturemoe.sh $MODEL_DIR $BASE_MODEL --memory_efficient --batch_size 2

# Then: System applies optimizations
# - Sets memory-efficient PyTorch backends
# - Uses smaller batch sizes
# - Performs garbage collection during evaluation
# - Completes evaluation within memory constraints
```

### Requirement: Structured evaluation results storage
The evaluation system SHALL save comprehensive results in structured JSON format with model metadata, metrics, and evaluation context.

#### Scenario: Results file generation
```json
{
  "evaluation_info": {
    "timestamp": "2024-12-30T14:30:22.123456",
    "model_dir": "/path/to/model",
    "base_model_path": "/path/to/base",
    "device": "cuda:0",
    "evaluator_version": "1.0.0"
  },
  "model_info": {
    "backbone": "llama",
    "num_moe_experts": 4,
    "num_activated_experts": 2,
    "use_shared": true,
    "use_lora": true
  },
  "test_metrics": {
    "accuracy": 0.8542,
    "precision": 0.8501,
    "recall": 0.8542,
    "f1_score": 0.8521,
    "loss": 0.3421,
    "num_samples": 1000,
    "classification_report": {...}
  },
  "test_samples": 1000,
  "training_config": {...}
}
```

#### Scenario: Results summary display
```bash
# Given: Completed evaluation
# When: Evaluation finishes successfully
./run_eval_simplified_culturemoe.sh $MODEL_DIR $BASE_MODEL

# Then: System displays results summary
# Output includes:
# - Test set accuracy: 0.8542
# - Precision: 0.8501
# - Recall: 0.8542
# - F1 score: 0.8521
# - Model architecture: llama
# - MoE experts: 4
# - Test samples: 1000
```

### Requirement: Support for multiple model configurations
The evaluation system SHALL support all CultureMoE model configurations including different backbones (LLaMA/Qwen), MoE settings, and LoRA configurations.

#### Scenario: LLaMA model evaluation
```bash
# Given: LLaMA-based CultureMoE model
MODEL_DIR="/path/to/llama_culturemoe"
BASE_MODEL="/path/to/Meta-Llama-3.1-8B-Instruct"

# When: Running evaluation
./run_eval_simplified_culturemoe.sh $MODEL_DIR $BASE_MODEL

# Then: System detects and handles LLaMA architecture
# - Identifies backbone as 'llama' from path or config
# - Sets appropriate tokenizer configuration
# - Loads model with correct architecture settings
```

#### Scenario: Qwen model evaluation
```bash
# Given: Qwen-based CultureMoE model
MODEL_DIR="/path/to/qwen_culturemoe"
BASE_MODEL="/path/to/Qwen2.5-7B-Instruct"

# When: Running evaluation
./run_eval_simplified_culturemoe.sh $MODEL_DIR $BASE_MODEL

# Then: System detects and handles Qwen architecture
# - Identifies backbone as 'qwen' from path or config
# - Sets pad_token to '<|endoftext|>'
# - Applies Qwen-specific model loading
```

#### Scenario: GPU memory management
```python
# Given: Large model that may exceed GPU memory
# When: Evaluation encounters memory pressure
try:
    outputs = model(input_ids, attention_mask, labels)
except torch.cuda.OutOfMemoryError:
    # Then: System handles gracefully
    torch.cuda.empty_cache()
    # Reduces batch size or switches to CPU
    # Provides clear error message with suggestions
```

### Requirement: Seamless integration with training outputs
The evaluation system SHALL work with all outputs from the training pipeline without requiring manual configuration or file modifications.

#### Scenario: Training-to-evaluation workflow
```bash
# Given: Completed training run
./run_simplified_culturemoe.sh llama 2

# Training produces:
# - /root/autodl-fs/simplified_culturemoe/llama_CulturalBench_20241230_143022/
#   ├── best_simplified_culturemoe/
#   │   ├── lora_weights/
#   │   ├── moe_weights.pt
#   │   └── simplified_culturemoe_config.json
#   ├── data_split_8_1_1.pkl
#   └── config.json

# When: Running evaluation immediately after training
./run_eval_simplified_culturemoe.sh \
    /root/autodl-fs/simplified_culturemoe/llama_CulturalBench_20241230_143022/best_simplified_culturemoe \
    /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct

# Then: System uses training outputs directly
# - No file copying or modification required
# - Uses exact test set from training splits
# - Maintains evaluation integrity
```