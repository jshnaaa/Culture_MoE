# Project Context

## Purpose
**Culture_Moe** is a research project that implements **CultureMoE (Culture-aware Mixture of Experts)** - a specialized fine-tuning approach for Large Language Models (LLMs) to improve cultural awareness and alignment. The project extends the LLaMA-Factory framework to enable efficient cultural fine-tuning of LLMs through a Mixture of Experts architecture where each expert specializes in different cultural dimensions.

### Core Goals
- Fine-tune LLMs (LLaMA 3.1-8B-Instruct and Qwen 2.5-7B-Instruct) with cultural understanding
- Implement MoE architecture with culture-specific expert specialization
- Support multi-class cultural classification tasks (binary, ternary, quaternary, quinary)
- Enable both discriminative (classification) and generative task training
- Provide efficient fine-tuning through LoRA (Low-Rank Adaptation) and MoE combinations

## Tech Stack

### Core Framework
- **LLaMA-Factory**: Unified efficient fine-tuning framework (base)
- **PyTorch**: >= 2.0.0 (deep learning framework)
- **Transformers**: >= 4.49.0 (Hugging Face model library)
- **Python**: >= 3.9.0 (developed with 3.12)

### Fine-tuning & Training
- **PEFT**: LoRA implementation and parameter-efficient fine-tuning
- **TRL**: Training utilities for reinforcement learning
- **Accelerate**: >= 1.3.0 (distributed training)
- **DeepSpeed**: Integration for large-scale training
- **bitsandbytes**: Quantization support

### Data & Evaluation
- **Datasets**: Data loading and processing
- **NumPy, Pandas, SciPy**: Scientific computing
- **scikit-learn**: Machine learning utilities
- **tokenizers**: Text tokenization

### API & Interface
- **Gradio**: Web interface for model interaction
- **FastAPI + Uvicorn**: API server implementation

## Project Conventions

### Code Style
- **File Naming**: Snake_case for Python files, kebab-case for shell scripts
- **Class Naming**: PascalCase (e.g., `CultureMoE`, `ExpertRouter`)
- **Function Naming**: Snake_case (e.g., `load_model`, `train_culturemoe`)
- **Constants**: UPPER_SNAKE_CASE
- **Configuration**: Use dataclasses for structured configuration (e.g., `MoEArguments`)

### Architecture Patterns
- **Modular Design**: Separate modules for model, training, data, and evaluation
- **Factory Pattern**: Model loading through centralized factory methods
- **Configuration-Driven**: Extensive use of argument dataclasses and JSON configs
- **Layered Architecture**: Clear separation between model implementation, training logic, and evaluation
- **Component Composition**: Experts, routers, and shared layers as composable components

### Testing Strategy
- **Evaluation Scripts**: Comprehensive evaluation for different model configurations
- **VSM13 Testing**: Cultural consistency testing using VSM13 cultural dimensions
- **Multi-Metric Evaluation**: Accuracy, Precision, Recall, F1-score across classification tasks
- **Generative Evaluation**: Answer generation with post-evaluation quality assessment
- **Component Testing**: Individual testing of LoRA-only vs CultureMoE configurations

### Git Workflow
- **Branch Strategy**: Feature branches from main branch
- **Commit Convention**: Descriptive commits with date prefixes (e.g., "1112：run_ft_culturemoe_gen.sh...")
- **Script Versioning**: Shell scripts with versioned improvements and parameter tuning
- **Documentation**: Maintain QUICK_START.md and STRUCTURE_IMPROVEMENTS.md for architecture changes

## Domain Context

### Cultural AI & LLM Fine-tuning
- **Cultural Dimensions**: Understanding of cultural value systems and their representation in LLMs
- **Mixture of Experts**: Sparse and dense MoE architectures, expert routing mechanisms
- **LoRA Fine-tuning**: Low-rank adaptation for parameter-efficient training
- **Culture Specialization**: Techniques to encourage expert specialization on cultural dimensions

### Model Architectures
- **LLaMA Models**: Understanding of LLaMA 3.1-8B-Instruct architecture and tokenization
- **Qwen Models**: Knowledge of Qwen 2.5-7B-Instruct specifics and SDPA requirements
- **Expert Systems**: Router networks, gating mechanisms, and expert layer design
- **Shared vs Expert Layers**: Capacity balancing between shared and expert computations

### Training Methodologies
- **Two-Stage Training**: LoRA pre-training followed by MoE enhancement
- **Culture Loss Functions**: Mutual information minimization and JS divergence
- **Distributed Training**: DDP and multi-GPU training strategies
- **Warmup Strategies**: MoE integration with gradual gating activation

## Important Constraints

### Technical Constraints
- **SDPA Numerical Stability**: Qwen2.5 requires disabling FlashAttention (force PyTorch math backend)
- **Memory Requirements**: Large model fine-tuning requires careful memory management and quantization
- **Distributed Training**: Proper rank/world_size setup required for DDP configurations
- **Model Merging**: LoRA weights must be merged before MoE training in certain configurations

### Performance Constraints
- **Expert Capacity**: 6 experts maximum for current architecture
- **Shared Layer Weakening**: 4096→1024→4096 capacity reduction for balanced learning
- **Batch Size Limitations**: GPU memory constraints affect batch size selection
- **Training Time**: Multi-stage training requires significant computational resources

### Data Constraints
- **JSON Format**: Specific dataset format requirements (instruction, input, output fields)
- **Cultural Classification**: Limited to 2-5 class cultural classification tasks
- **Language Support**: Primarily designed for English and Chinese cultural contexts

## External Dependencies

### Model Repositories
- **Hugging Face Hub**: Base model downloads (LLaMA 3.1-8B-Instruct, Qwen 2.5-7B-Instruct)
- **Model Checkpoints**: Pre-trained model weights and tokenizers

### Datasets
- **CulturalBench_Hard_merge.json**: Binary cultural classification dataset
- **normad_ed_merge.json**: Ternary cultural classification dataset
- **wvs_all_llama_merge_4.json**: Quaternary cultural classification dataset
- **wvs_all_llama_merge_5.json**: Quinary cultural classification dataset

### Infrastructure
- **CUDA Environment**: GPU support required for efficient training
- **Docker**: Containerized CUDA environments for reproducible training
- **Distributed Computing**: Multi-GPU setups for large-scale training

### APIs & Services
- **Gradio Interface**: Web-based model interaction and testing
- **FastAPI Server**: RESTful API for model serving and evaluation
