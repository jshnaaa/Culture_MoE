# Culture_Moe 项目文件结构说明

本文档整理了 Culture_Moe 项目目录中的全部文件，包括文件位置、文件名及其功能。项目基于 LLaMA-Factory 框架二次开发，在其中新增了 CultureMoE 联合训练模型、MoE 专家架构、CSL 文化损失函数及配套的训练/评估脚本。

**图例说明**：标注 🔥 的文件为本项目核心新增/深度修改的文件；标注 🔧 的文件为基于 LLaMA-Factory 修改的文件；其余为 LLaMA-Factory 原有文件。

---

## 1. 根目录文件

### 1.1 项目文档与配置

| 文件 | 功能 |
|------|------|
| `culturemoe.md` | CultureMoE 详细架构文档（联合训练架构、CSL 损失、MASK 机制等） |
| `mad/MAD.md` | Multi-Agent Debate (MAD) 评估方案文档，介绍多代理辩论系统用于文化理解任务评估的设计 |
| `mad/mad_prompt_5cultural.md` | 5-Agent 文化视角 MAD 系统的完整 Prompt 文档（亚洲/欧美等 5 个文化 Agent 的各轮提示词） |
| `MANIFEST.in` | Python 打包清单文件 |
| `pyproject.toml` | 项目构建与 ruff lint 配置（Python 3.9+，行宽 119） |
| `setup.py` | Python 包安装入口 |
| `requirements.txt` | Python 依赖列表 |
| `.gitignore` | Git 忽略规则 |
| `.gitattributes` | Git 属性配置 |
| `.env.local` | 本地环境变量配置 |
| `.pre-commit-config.yaml` | Pre-commit 钩子配置（语法检查、大文件检查、调试语句检查等） |

### 1.2 核心训练脚本 🔥

| 文件 | 功能 |
|------|------|
| `train_joint_lora_moe.py` | **联合训练主脚本**：端到端同时训练预训练 LoRA 适配器 + 新增 MoE 专家层，实现真正的联合优化，针对 48GB×2 卡优化 |
| `train_simplified_culturemoe.py` | 简化版 FFN CultureMoE 训练脚本：基于 MixLoRA 实现，在所有层使用 MoE 并添加文化损失 |
| `ft_lora_only_gen.py` | LoRA Only 基线微调脚本：使用标准语言建模损失在新格式数据集上微调 |
| `ft_base_gen.py` | Base 基线微调/评测脚本：在新格式 CultureLLM 数据集上评估基础模型（LLaMA 或 Qwen） |
| `ft_milora_gen.py` | MiLoRA (Mixture of LoRA) 训练脚本：基于提示感知路由机制的混合 LoRA 专家训练（对比基线） |
| `ft_mixlora.py` | MixLoRA 微调脚本：结合 LoRA 参数高效性与 MoE 性能的对比基线方法 |
| `enhanced_moe_losses.py` | 增强的 MoE 损失函数模块：实现 L = L_h + αL_aux + βL_o + γL_v（主任务损失、负载均衡、专家输出正交化、路由方差损失） |

### 1.3 评估脚本 🔥

| 文件 | 功能 |
|------|------|
| `eval_joint_culturemoe.py` | 联合训练 CultureMoE 模型评估脚本：评估 `train_joint_lora_moe.py` 训练的联合 LoRA+MoE 模型，支持 8:1:1 划分测试集与完整数据集评估 |
| `eval_simplified_culturemoe.py` | 简化版 CultureMoE 模型评估脚本：从训练输出目录加载模型并在测试集上评估 |
| `eval_lora_only_from_components.py` | 从 Base 模型 + LoRA 权重还原完整模型并评估（Qwen 模型特别优化版本） |
| `eval_mixlora.py` | MixLoRA 模型评估脚本：从保存的训练参数还原完整模型并在测试集上评估 |
| `eval_mistral.py` | Mistral-Small-3.1 API 评测脚本：调用 API 对文化数据集进行评测 |
| `eval_vsm13.py` | VSM13（霍夫斯泰德文化价值观维度）评估脚本：生成式版本，计算各国维度分数并与标准分数对比 |
| `mad/eval_mad.py` | Multi-Agent Debate (MAD) 评估脚本：基于 Prompt 的多代理辩论系统，用于文化理解任务评估 |
| `mad/eval_mad_5cultural.py` | 5-Agent 文化视角 MAD 评估脚本：实现亚洲等 5 个文化视角的多代理辩论评估 |

### 1.4 训练/评估执行 Shell 脚本 🔥

| 文件 | 功能 |
|------|------|
| `run_joint_lora_moe_training.sh` | **联合训练执行脚本（主入口）**：参数顺序为 backbone、data_id、use_shared、use_mask、use_gate、use_culture_loss、num_moe_experts、num_activated_experts、lora_rank、lora_alpha、use_lora、num_gpus |
| `run_simplified_culturemoe.sh` | 简化版 CultureMoE 训练执行脚本 |
| `run_ablation_study.sh` | 简化版 CultureMoE 消融实验批处理脚本：自动化测试不同组件配置（shared/mask/gate/culture_loss）的性能，内部直接调用 `eval_simplified_culturemoe.py` |
| `run_eval_joint_culturemoe.sh` | 联合模型评估执行脚本 |
| `run_eval_lora_only_from_components.sh` | 从 Base + LoRA 权重还原模型并评估的执行脚本（增强版本） |
| `run_eval_mixlora.sh` | MixLoRA 模型评估执行脚本（评估 `run_ft_mixlora.sh` 训练的模型） |
| `run_eval_vsm13.sh` | VSM13 数据集评估执行脚本 |
| `mad/run_eval_mad.sh` | Multi-Agent Debate 评估执行脚本（内部通过脚本所在目录定位 eval_mad.py，支持从任意目录调用） |
| `mad/run_eval_mad_5cultural.sh` | 5-Agent 文化 MAD 评估执行脚本（同上） |
| `run_eval_gpt.sh` | 多模型 API 评测执行脚本：支持 DeepSeek-R1、Mistral、GPT 等模型的 API 评测 |
| `run_ft_base.sh` | Base 基线训练/评测执行脚本（参数：BACKBONE、DATA_ID） |
| `run_ft_lora_only_gen.sh` | LoRA Only 微调执行脚本（新数据格式） |
| `run_ft_milora_gen.sh` | MiLoRA 训练执行脚本 |
| `run_ft_mixlora.sh` | MixLoRA 微调执行脚本 |

### 1.5 辅助脚本与工具

| 文件 | 功能 |
|------|------|
| `count_params.py` | 统计各方法（DC-Finetune、Vanilla MoE、MixLoRA、CulDPD）在统一 LoRA rank=16/alpha=32 配置下、LLaMA-3.1-8B 与 Qwen2.5-7B 骨干上的可训练参数量 |
| `git.sh` | Git 辅助脚本：拉取远程最新代码并强制切换到指定远程分支 |

### 1.6 数据与产出文件

> 以下论文返修相关文件已统一移至 `rebuttal/` 目录。

| 文件 | 功能 |
|------|------|
| `rebuttal/review3.docx` | 审稿意见及修改对照 Word 文档（第三轮） |
| `rebuttal/review5.docx` | 审稿意见及修改对照 Word 文档（第五轮） |
| `rebuttal/review总结.docx` | 审稿意见总结 Word 文档 |
| `rebuttal/review_rebuttal.docx` | Rebuttal 回复 Word 文档 |
| `rebuttal/plot_rebuttal_figures.py` | 生成 rebuttal 图表：专家×文化区域激活比例热力图与专家数量消融折线图 |
| `rebuttal/expert-region.png` | 专家×文化区域激活比例热力图（由 `rebuttal/plot_rebuttal_figures.py` 生成） |
| `rebuttal/Expert-count.png` | 专家数量消融实验折线图（由 `rebuttal/plot_rebuttal_figures.py` 生成） |
| `rebuttal/culturemoe.png` | CultureMoE 架构图 |

> 说明：训练数据集（unified_all_datasets.json、CulturalBench_merge_gen.json、normad_merge_gen.json、cultureLLM_merge_gen.json）不在仓库目录中，运行时通过 DATA_ID 参数从外部数据路径加载。

---

## 2. src/ — 源代码目录

### 2.1 src/ 根目录（框架入口）

| 文件 | 功能 |
|------|------|
| `src/train.py` | 训练启动入口（LLaMA-Factory 标准入口） |
| `src/llamafactory/__init__.py` | 包初始化及版本声明 |

### 2.2 src/llamafactory/ — 框架核心

| 文件 | 功能 |
|------|------|
| `src/llamafactory/cli.py` | 命令行接口，分发 train/talk/webui/chat 等子命令 |
| `src/llamafactory/launcher.py` | 多卡/分布式训练启动器 |

### 2.3 src/llamafactory/model/ — 模型层（本项目核心修改区）🔥

**CultureMoE 核心实现：**

| 文件 | 功能 |
|------|------|
| `model/CultureMoE.py` | 主要 MoE 模型类：包含跨注意力块（CrossAttentionBlock，Shared 层输出作为 Q、专家+Shared 输出作为 KV）等核心 MoE 组件 |
| `model/joint_lora_moe_model.py` | **联合训练模型 `JointLoRAMoEModel`**：端到端 LoRA+MoE 联合优化架构，包含 LoRAExpert（参数高效专家网络）、MoERouter（动态专家选择与权重分配）、MoEFFNLoRA（专家混合前馈网络）等主要组件 |
| `model/simplified_culturemoe.py` | 简化版 CultureMoE 配置定义 |
| `model/simplified_culturemoe_adapter.py` | 简化版 CultureMoE 适配器实现 |
| `model/moe_args.py` | MoE 相关命令行/配置参数定义 |

**对比基线方法实现：**

| 文件 | 功能 |
|------|------|
| `model/mixlora.py` | MixLoRA 模型实现（对比基线） |
| `model/mixlora_adapter.py` | MixLoRA 适配器实现 |
| `model/milora.py` | MiLoRA (Mixture of LoRA) 实现：基于提示感知路由机制的混合 LoRA 专家（对比基线） |

**LLaMA-Factory 框架原有模型文件（🔧 有适配修改）：**

| 文件 | 功能 |
|------|------|
| `model/adapter.py` | LoRA/Freeze/Full 等微调适配器挂载逻辑 |
| `model/loader.py` | 模型加载器（处理预训练权重加载、适配器挂载） |
| `model/patcher.py` | 模型补丁工具（对齐注入、梯度检查点设置等） |
| `model/__init__.py` | 模型层包初始化 |
| `model/model_utils/attention.py` | 注意力模块工具（如 longlora attention 补丁） |
| `model/model_utils/checkpointing.py` | 梯度检查点工具 |
| `model/model_utils/embedding.py` | 嵌入层工具（可训练嵌入等） |
| `model/model_utils/kv_cache.py` | KV Cache 配置工具 |
| `model/model_utils/liger_kernel.py` | Liger Kernel 融合算子集成 |
| `model/model_utils/longlora.py` | LongLoRA 支持工具 |
| `model/model_utils/misc.py` | 杂项模型工具 |
| `model/model_utils/mod.py` | 模块丢弃（mod）工具 |
| `model/model_utils/moe.py` | MoE 模型（如 Qwen-MoE）支持工具 |
| `model/model_utils/packing.py` | 序列打包工具 |
| `model/model_utils/quantization.py` | 量化工具（BLOOM/bitsandbytes 兼容处理） |
| `model/model_utils/rope.py` | RoPE 编码调整工具 |
| `model/model_utils/unsloth.py` | Unsloth 加速集成 |
| `model/model_utils/valuehead.py` | PPO/RM 用的 Value Head 模块 |
| `model/model_utils/visual.py` | 视觉语言模型（VLM）工具 |
| `model/model_utils/__init__.py` | 模型工具包初始化 |

### 2.4 src/llamafactory/data/ — 数据层 🔧

| 文件 | 功能 |
|------|------|
| `data/loader.py` | 数据集加载器（解析 dataset_info.json 并加载各类数据集） |
| `data/converter.py` | ShareGPT/Alpaca 等数据格式转换器 |
| `data/parser.py` | 数据集参数解析器 |
| `data/formatter.py` | 数据格式化器 |
| `data/collator.py` | 数据整理器（pad、标签掩码等） |
| `data/data_utils.py` | 数据工具函数（tokenize、截断等） |
| `data/template.py` | 对话模板定义（各模型的 chat template） |
| `data/tool_utils.py` | 工具调用格式工具 |
| `data/mm_plugin.py` | 多模态（图像/视频）插件 |
| `data/classification_processor.py` 🔥 | 分类任务数据处理器（本项目新增，用于文化数据集处理） |
| `data/dual_classification_processor.py` 🔥 | 双路分类任务数据处理器（本项目新增，配合 MASK 双路输入） |
| `data/dual_classification_collator.py` 🔥 | 双路分类数据整理器（本项目新增） |
| `data/processor/supervised.py` | 有监督微调数据处理器 |
| `data/processor/unsupervised.py` | 无监督预训练数据处理器 |
| `data/processor/pretrain.py` | 预训练数据处理器 |
| `data/processor/pairwise.py` | 成对偏好（DPO）数据处理器 |
| `data/processor/feedback.py` | 反馈数据（KTO）处理器 |
| `data/processor/processor_utils.py` | 处理器通用工具 |
| `data/__init__.py` / `data/processor/__init__.py` | 包初始化 |

### 2.5 src/llamafactory/train/ — 训练层 🔧

**通用训练组件：**

| 文件 | 功能 |
|------|------|
| `train/tuner.py` | 训练流程统一封装（run_sft/run_rm 等） |
| `train/trainer_utils.py` | Trainer 工具（参数打包、回调注册） |
| `train/callbacks.py` | 训练回调（日志上报等） |
| `train/test_utils.py` | 测试工具（CI 用） |
| `train/__init__.py` | 包初始化 |

**分类训练（本项目新增模块）🔥：**

| 文件 | 功能 |
|------|------|
| `train/classification/trainer.py` | 分类任务 Trainer 实现 |
| `train/classification/workflow.py` | 分类训练流程 |
| `train/classification/culture_loss.py` | 文化损失函数实现（CSL 相关） |
| `train/classification/callbacks.py` | 分类训练回调 |
| `train/classification/metrics.py` | 分类指标计算 |
| `train/classification/__init__.py` | 包初始化 |

**标准训练算法（LLaMA-Factory 原有）：**

| 文件 | 功能 |
|------|------|
| `train/sft/trainer.py` / `train/sft/workflow.py` / `train/sft/metric.py` | SFT（监督微调）训练器、流程与指标 |
| `train/pt/trainer.py` / `train/pt/workflow.py` | PT（预训练）训练器与流程 |
| `train/dpo/trainer.py` / `train/dpo/workflow.py` | DPO 训练器与流程 |
| `train/kto/trainer.py` / `train/kto/workflow.py` | KTO 训练器与流程 |
| `train/rm/trainer.py` / `train/rm/workflow.py` / `train/rm/metric.py` | RM（奖励模型）训练器、流程与指标 |
| `train/ppo/trainer.py` / `train/ppo/workflow.py` / `train/ppo/ppo_utils.py` | PPO 训练器、流程与工具 |
| `train/sft/__init__.py` / `train/pt/__init__.py` / `train/dpo/__init__.py` / `train/kto/__init__.py` / `train/rm/__init__.py` / `train/ppo/__init__.py` | 上述各训练算法子目录的包初始化文件 |

### 2.6 src/llamafactory/api/ — API 服务

| 文件 | 功能 |
|------|------|
| `api/app.py` | OpenAI 风格 API 服务应用 |
| `api/chat.py` | Chat/Completion 推理逻辑 |
| `api/common.py` | API 公共工具（模板构造、流式输出） |
| `api/protocol.py` | API 协议数据结构定义 |
| `api/__init__.py` | 包初始化 |

### 2.7 src/llamafactory/eval/ — 评估器

| 文件 | 功能 |
|------|------|
| `eval/evaluator.py` | 标准基准（MMLU/CMMLU 等）评估器 |
| `eval/template.py` | 评估模板定义 |
| `eval/__init__.py` | 包初始化 |

### 2.8 src/llamafactory/extras/ — 通用工具

| 文件 | 功能 |
|------|------|
| `extras/constants.py` | 常量定义（模型模板、方法名映射等） |
| `extras/env.py` | 环境变量管理（序列化训练参数） |
| `extras/logging.py` | 日志记录工具 |
| `extras/misc.py` | 杂项工具（seed、GPU 检测等） |
| `extras/packages.py` | 依赖包检测工具 |
| `extras/ploting.py` | 训练损失可视化（TensorBoard） |
| `extras/__init__.py` | 包初始化 |

---

## 3. scripts/ — 实用工具脚本（LLaMA-Factory 原有）

| 文件 | 功能 |
|------|------|
| `scripts/api_example/test_image.py` | API 调用示例：多模态图像测试 |
| `scripts/api_example/test_toolcall.py` | API 调用示例：工具调用测试 |
| `scripts/convert_ckpt/llamafy_baichuan2.py` | Baichuan2 权重转换为 LLaMA 格式 |
| `scripts/convert_ckpt/llamafy_qwen.py` | Qwen 权重转换为 LLaMA 格式 |
| `scripts/convert_ckpt/tiny_llama4.py` | Llama4 权重精简转换 |
| `scripts/eval_bleu_rouge.py` | BLEU/ROUGE 生成指标评估 |
| `scripts/llama_pro.py` | LLaMA Pro 块扩展工具 |
| `scripts/loftq_init.py` | LoftQ 量化初始化 |
| `scripts/pissa_init.py` | PiSSA 量化初始化 |
| `scripts/qwen_omni_merge.py` | Qwen-Omni 权重合并 |
| `scripts/vllm_infer.py` | vLLM 批量推理脚本 |
| `scripts/stat_utils/cal_flops.py` | 训练 FLOPs 计算工具 |
| `scripts/stat_utils/cal_lr.py` | 学习率计算工具 |
| `scripts/stat_utils/cal_mfu.py` | MFU（模型算力利用率）计算工具（注：其内部引用的 `examples/deepspeed/` 配置目录已删除，使用时需自行恢复配置路径） |
| `scripts/stat_utils/cal_ppl.py` | 困惑度计算工具 |
| `scripts/stat_utils/length_cdf.py` | 序列长度分布（CDF）统计工具 |

---

## 4. .github/ — GitHub 仓库配置（LLaMA-Factory 原有）

| 文件 | 功能 |
|------|------|
| `.github/workflows/tests.yml` | CI 测试工作流 |
| `.github/workflows/docker.yml` | Docker 镜像自动构建发布工作流 |
| `.github/workflows/publish.yml` | PyPI 发布工作流 |
| `.github/workflows/label_issue.yml` | Issue 自动打标签工作流 |
| `.github/ISSUE_TEMPLATE/1-bug-report.yml` | Bug 报告 Issue 模板 |
| `.github/ISSUE_TEMPLATE/2-feature-request.yml` | 功能请求 Issue 模板 |
| `.github/ISSUE_TEMPLATE/config.yml` | Issue 模板选择页配置 |
| `.github/CODE_OF_CONDUCT.md` | 社区行为准则 |
| `.github/CONTRIBUTING.md` | 贡献指南 |
| `.github/SECURITY.md` | 安全策略 |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR 模板 |

> 注意：`.github/workflows/docker.yml` 引用的 `docker/` 目录已删除，若仓库推送到 GitHub 该工作流会失败；`.github/workflows/publish.yml` 引用的 readme 亦已删除。如不再使用 CI 可自行移除这两个工作流。

---

## 5. 核心工作流速查

```mermaid
flowchart LR
    A[数据集<br/>CulturalBench/cultureLLM/normad] --> B[run_joint_lora_moe_training.sh]
    B --> C[train_joint_lora_moe.py<br/>联合LoRA+MoE训练]
    C --> D[joint_lora_moe_model.py<br/>模型架构+CSL损失]
    D --> E[训练输出模型]
    E --> F[run_eval_joint_culturemoe.sh]
    F --> G[eval_joint_culturemoe.py<br/>测试集评估]
    G --> H[rebuttal/plot_rebuttal_figures.py<br/>生成图表]
```

**对比基线工作流**：`run_ft_lora_only_gen.sh`（LoRA Only）、`run_ft_mixlora.sh`（MixLoRA）、`run_ft_milora_gen.sh`（MiLoRA）、`run_ft_base.sh`（Base 模型）分别训练/评估基线，与 CultureMoE 主方法对比。

**主训练命令示例**（详见 `run_joint_lora_moe_training.sh`）：

```bash
./run_joint_lora_moe_training.sh llama 2 true true true csl 4 2 16 32 true 2
# 参数依次为: backbone 数据集ID 共享专家 MASK机制 门控网络
# 文化损失模式 专家总数 激活专家数 LoRA秩 LoRA缩放 基础LoRA GPU数量
```

---

*文档生成时间：2026-08-23，基于当前工作目录实际文件整理。*
