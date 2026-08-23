# Multi-Agent Debate (MAD) 评估方案

## 项目概述

### 什么是Multi-Agent Debate (MAD)

Multi-Agent Debate (MAD) 是一种通过多个AI代理进行结构化辩论来提升推理质量的方法。在文化理解任务中，不同的"思考角度"可以帮助模型更全面地考虑文化差异和细微之处。

### 核心特点

- **纯Prompt-Based**：无需训练，完全通过精心设计的prompt实现
- **同质模型多角色**：使用相同的base model，通过不同角色prompt产生多样性
- **反思机制**：每轮辩论都包含对自己和对方观点的批判性反思
- **三角色设计**：正方、反方、裁判三个角色协同决策

### 与传统方法的对比

| 方法 | 模型数量 | 训练需求 | 推理成本 | 适用场景 |
|------|---------|---------|---------|---------|
| 单模型直接推理 | 1 | 无 | 低 | 简单任务 |
| Self-Consistency | 1 | 无 | 中 | 需要多样性 |
| **MAD (本方案)** | 1 (多角色) | 无 | 中-高 | 复杂推理、文化理解 |
| Ensemble | 多 | 可选 | 高 | 需要模型多样性 |

---

## 系统架构

### 三角色设计

```
┌─────────────────────────────────────────────────────────┐
│                    MAD System Architecture               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────────┐              ┌──────────────┐       │
│   │  Agent A     │◄────────────►│  Agent B     │       │
│   │  (正方)      │   相互辩论    │  (反方)      │       │
│   │  Affirmative │              │  Negative    │       │
│   └──────┬───────┘              └──────┬───────┘       │
│          │                              │               │
│          │         辩论历史              │               │
│          └──────────┬───────────────────┘               │
│                     ▼                                   │
│              ┌──────────────┐                           │
│              │  Agent C     │                           │
│              │  (裁判)      │                           │
│              │  Judge       │                           │
│              └──────────────┘                           │
│                     │                                   │
│                     ▼                                   │
│              Final Answer                               │
└─────────────────────────────────────────────────────────┘
```

### 角色职责

**Agent A (正方/Affirmative)**
- 职责：从初步直觉出发，给出第一反应答案
- 特点：代表"第一印象"和"直接推理"
- 反思：评估反方观点，决定是否调整

**Agent B (反方/Negative)**
- 职责：从批判性角度思考，挑战常规答案
- 特点：代表"深度思考"和"多角度分析"
- 反思：评估正方观点，提供替代视角

**Agent C (裁判/Judge)**
- 职责：综合双方论证，给出最终判断
- 特点：中立、全面、注重论证质量
- 标准：逻辑严密性、文化敏感性、证据充分性

---

## 辩论流程详解

### 完整流程图

```
┌─────────────────────────────────────────────────────────┐
│                   MAD Debate Process                     │
└─────────────────────────────────────────────────────────┘

Round 0: 初始轮 (Independent Reasoning)
┌──────────────────────────────────────────────────────────┐
│ Agent A (正方)                  Agent B (反方)            │
│ ├─ 输入: Question + Context    ├─ 输入: Question + Context│
│ ├─ 角色: 正方立场               ├─ 角色: 反方立场          │
│ └─ 输出: Answer_A0, Reason_A0  └─ 输出: Answer_B0, Reason_B0│
└──────────────────────────────────────────────────────────┘
                            ▼
Round 1: 第一轮辩论 (First Debate with Reflection)
┌──────────────────────────────────────────────────────────┐
│ Agent A                          Agent B                  │
│ ├─ 输入: 自己的(A0, R_A0)       ├─ 输入: 自己的(B0, R_B0) │
│ │        对方的(B0, R_B0)       │        对方的(A0, R_A0) │
│ ├─ 反思:                        ├─ 反思:                  │
│ │  - 对方观点的优点             │  - 对方观点的优点        │
│ │  - 对方观点的缺点             │  - 对方观点的缺点        │
│ │  - 我的观点是否需要调整       │  - 我的观点是否需要调整  │
│ └─ 输出: Answer_A1, Reason_A1   └─ 输出: Answer_B1, Reason_B1│
│          Reflection_A1                   Reflection_B1    │
└──────────────────────────────────────────────────────────┘
                            ▼
Round 2: 第二轮辩论 (Second Debate with Deeper Reflection)
┌──────────────────────────────────────────────────────────┐
│ Agent A                          Agent B                  │
│ ├─ 输入: 完整历史(R0, R1)       ├─ 输入: 完整历史(R0, R1) │
│ ├─ 深度反思:                    ├─ 深度反思:              │
│ │  - 辩论中的新见解             │  - 辩论中的新见解        │
│ │  - 最终立场确认               │  - 最终立场确认          │
│ └─ 输出: Answer_A2, Reason_A2   └─ 输出: Answer_B2, Reason_B2│
│          Final_Position_A                Final_Position_B │
└──────────────────────────────────────────────────────────┘
                            ▼
Final Decision: 裁判决策 (Judge's Final Decision)
┌──────────────────────────────────────────────────────────┐
│ Agent C (裁判)                                            │
│ ├─ 输入: Question + 完整辩论历史(R0, R1, R2)              │
│ ├─ 评估维度:                                              │
│ │  - 论证逻辑性 (Logical Coherence)                      │
│ │  - 文化敏感性 (Cultural Sensitivity)                   │
│ │  - 证据充分性 (Evidence Sufficiency)                   │
│ │  - 推理深度 (Reasoning Depth)                          │
│ ├─ 决策过程:                                              │
│ │  - 分析双方论证质量                                     │
│ │  - 识别关键分歧点                                       │
│ │  - 综合判断最佳答案                                     │
│ └─ 输出: Final_Answer, Decision_Reasoning, Confidence    │
└──────────────────────────────────────────────────────────┘
                            ▼
                      Final Result
```

### 各轮次详细说明

#### Round 0: 初始轮 (Independent Reasoning)

**目的**：获取两个独立的初始答案，避免锚定效应

**Agent A (正方) Prompt结构**：
```
角色设定: 你是一位文化专家，需要从直觉和第一印象出发回答问题
任务: 仔细分析问题，给出你的初步答案
输出要求: 答案 + 推理过程 + 置信度
```

**Agent B (反方) Prompt结构**：
```
角色设定: 你是一位批判性思考专家，需要从多角度深入分析
任务: 质疑常规答案，探索其他可能性
输出要求: 答案 + 推理过程 + 置信度
```

**关键点**：
- 两个agent看到的prompt略有不同，引导不同的思考方式
- 但都不是真正的"立场对立"，而是"思考角度差异"
- 独立生成，互不干扰

---

#### Round 1: 第一轮辩论 (First Debate with Reflection)

**目的**：通过看到对方观点，进行第一次反思和调整

**Prompt核心要素**：
1. **展示对方观点**：完整呈现对方的答案和推理
2. **引导反思**：
   - "对方的观点有哪些合理之处？"
   - "对方的推理有哪些不足？"
   - "我的初始答案是否需要调整？为什么？"
3. **鼓励更新**：明确说明"可以改变答案，也可以坚持己见"
4. **要求说明**：如果改变，说明原因；如果坚持，说明理由

**反思机制设计**：
```
Reflection Template:
1. 对方观点分析:
   - 优点: [具体说明]
   - 缺点: [具体说明]
2. 自我审视:
   - 我的推理是否充分？
   - 是否考虑了所有文化因素？
3. 决策:
   - 是否改变答案: [Yes/No]
   - 理由: [详细说明]
```

---

#### Round 2: 第二轮辩论 (Second Debate with Deeper Reflection)

**目的**：基于第一轮的交流，进行更深入的分析

**与Round 1的区别**：
- 输入包含完整的历史（R0 + R1）
- 可以看到对方的"反思"内容
- 更注重"新见解"和"最终立场"

**深度反思要素**：
1. **辩论收获**：从对方的反思中学到了什么？
2. **新的视角**：是否发现了之前忽略的文化细节？
3. **最终确信**：经过两轮辩论，最终答案的置信度如何？

---

#### Final Decision: 裁判决策

**目的**：综合双方论证，给出最终答案

**裁判的评估框架**：

```
评估维度权重:
├─ 论证逻辑性 (40%): 推理是否严密、前后一致
├─ 文化敏感性 (30%): 是否考虑了文化差异和细微之处
├─ 证据充分性 (20%): 是否有充分的理由支持答案
└─ 推理深度 (10%): 是否进行了深层次的分析
```

**决策过程**：
1. **回顾辩论**：梳理双方的核心论点和变化
2. **识别分歧**：找出双方意见不一致的关键点
3. **评估质量**：按照上述维度评分
4. **综合判断**：不是简单投票，而是基于论证质量
5. **给出答案**：最终答案 + 详细理由 + 置信度

---

## Prompt设计

### Prompt设计原则

1. **清晰的角色定位**：让模型明确自己的"身份"
2. **结构化输出**：使用固定格式，便于解析
3. **鼓励反思**：明确要求评估和自我审视
4. **避免极端立场**：不要让模型固执己见
5. **文化敏感性**：强调文化理解的重要性

---

### Round 0: 初始回答Prompt

#### Agent A (正方) - 初始Prompt

```python
PROMPT_AGENT_A_ROUND0 = """You are a cultural understanding expert. Your task is to answer the following multiple-choice question based on your initial intuition and reasoning.

**Question:**
{instruction}

**Context:**
{input}

**Instructions:**
1. Carefully read the question and context
2. Think about the cultural factors involved
3. Choose the best answer from options 1-4
4. Explain your reasoning clearly
5. Indicate your confidence level (0-100)

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your detailed reasoning in 3-5 sentences, focusing on cultural aspects]
Confidence: [0-100]

Please provide your answer now:"""
```

#### Agent B (反方) - 初始Prompt

```python
PROMPT_AGENT_B_ROUND0 = """You are a critical thinking expert specializing in cultural analysis. Your task is to answer the following question by considering multiple perspectives and challenging common assumptions.

**Question:**
{instruction}

**Context:**
{input}

**Instructions:**
1. Question the obvious answer - what might be overlooked?
2. Consider alternative cultural interpretations
3. Think about edge cases and exceptions
4. Choose the best answer from options 1-4
5. Explain your reasoning with critical analysis

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your detailed reasoning in 3-5 sentences, emphasizing alternative perspectives]
Confidence: [0-100]

Please provide your answer now:"""
```

---

### Round 1: 第一轮辩论Prompt (含反思)

#### Agent A - Round 1 Prompt

```python
PROMPT_AGENT_A_ROUND1 = """You are continuing the debate. You have seen the opposing view. Now reflect on both perspectives.

**Original Question:**
{instruction}

**Context:**
{input}

**Your Previous Answer (Round 0):**
- Answer: {answer_a0}
- Reasoning: {reasoning_a0}
- Confidence: {confidence_a0}

**Opposing View (Agent B's Answer):**
- Answer: {answer_b0}
- Reasoning: {reasoning_b0}
- Confidence: {confidence_b0}

**Instructions for Reflection:**
1. **Analyze the opposing view:**
   - What are the strengths of their argument?
   - What are the weaknesses or gaps?

2. **Self-reflection:**
   - Is my reasoning complete and accurate?
   - Did I overlook any important cultural factors?
   - Should I reconsider my answer?

3. **Make a decision:**
   - You may CHANGE your answer if convinced
   - You may KEEP your answer if you have stronger reasons
   - Explain your decision clearly

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your updated reasoning, incorporating insights from the debate]
Reflection: [Your reflection on the opposing view - strengths, weaknesses, and why you changed/kept your answer]
Confidence: [0-100]
Changed: [Yes/No]

Please provide your response:"""
```

#### Agent B - Round 1 Prompt

```python
PROMPT_AGENT_B_ROUND1 = """You are continuing the debate. You have seen the opposing view. Now reflect critically on both perspectives.

**Original Question:**
{instruction}

**Context:**
{input}

**Your Previous Answer (Round 0):**
- Answer: {answer_b0}
- Reasoning: {reasoning_b0}
- Confidence: {confidence_b0}

**Opposing View (Agent A's Answer):**
- Answer: {answer_a0}
- Reasoning: {reasoning_a0}
- Confidence: {confidence_a0}

**Instructions for Critical Reflection:**
1. **Evaluate the opposing view:**
   - What assumptions are they making?
   - Are there cultural nuances they missed?

2. **Challenge your own view:**
   - Is my alternative perspective truly better?
   - Am I being contrarian for the sake of it?
   - What evidence supports my answer?

3. **Refine your position:**
   - Strengthen your argument or adjust your answer
   - Provide clearer reasoning
   - Explain your decision

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your refined reasoning, addressing the opposing view]
Reflection: [Your critical analysis - what you learned and why your answer is justified]
Confidence: [0-100]
Changed: [Yes/No]

Please provide your response:"""
```

---

### Round 2: 第二轮辩论Prompt (深度反思)

#### Agent A - Round 2 Prompt

```python
PROMPT_AGENT_A_ROUND2 = """This is the final round of debate. Review the entire discussion and provide your final position.

**Original Question:**
{instruction}

**Context:**
{input}

**Debate History:**

Round 0 (Initial):
- Your answer: {answer_a0} (Confidence: {confidence_a0})
- Their answer: {answer_b0} (Confidence: {confidence_b0})

Round 1 (First Debate):
- Your answer: {answer_a1} (Changed: {changed_a1}, Confidence: {confidence_a1})
- Their answer: {answer_b1} (Changed: {changed_b1}, Confidence: {confidence_b1})
- Your reflection: {reflection_a1}
- Their reflection: {reflection_b1}

**Instructions for Final Reflection:**
1. **Review the debate trajectory:**
   - What new insights emerged?
   - How did the discussion evolve?

2. **Final decision:**
   - Based on all arguments, what is your final answer?
   - What is the strongest evidence for this answer?
   - What is your final confidence level?

3. **Acknowledge uncertainty:**
   - If still uncertain, explain why
   - If confident, explain what convinced you

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your final, comprehensive reasoning]
Final_Reflection: [What you learned from this debate and your final position]
Confidence: [0-100]
Changed_From_R1: [Yes/No]

Please provide your final response:"""
```

#### Agent B - Round 2 Prompt

```python
PROMPT_AGENT_B_ROUND2 = """This is the final round of debate. Review the entire discussion and provide your final critical assessment.

**Original Question:**
{instruction}

**Context:**
{input}

**Debate History:**

Round 0 (Initial):
- Your answer: {answer_b0} (Confidence: {confidence_b0})
- Their answer: {answer_a0} (Confidence: {confidence_a0})

Round 1 (First Debate):
- Your answer: {answer_b1} (Changed: {changed_b1}, Confidence: {confidence_b1})
- Their answer: {answer_a1} (Changed: {changed_a1}, Confidence: {confidence_a1})
- Your reflection: {reflection_b1}
- Their reflection: {reflection_a1}

**Instructions for Final Critical Assessment:**
1. **Synthesize the debate:**
   - What were the key points of disagreement?
   - Where did we find common ground?

2. **Final judgment:**
   - What is the most culturally accurate answer?
   - Have I been too critical or not critical enough?
   - What is my final confidence?

3. **Closing statement:**
   - Summarize your final position
   - Acknowledge any remaining doubts

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your final, comprehensive reasoning]
Final_Reflection: [Your synthesis of the debate and final critical assessment]
Confidence: [0-100]
Changed_From_R1: [Yes/No]

Please provide your final response:"""
```

---

### Final Decision: 裁判决策Prompt

```python
PROMPT_JUDGE_FINAL = """You are an impartial judge tasked with making the final decision based on a debate between two agents. Your role is to evaluate the quality of arguments, not simply count votes.

**Original Question:**
{instruction}

**Context:**
{input}

**Complete Debate History:**

=== ROUND 0: Initial Answers ===
Agent A (Affirmative):
- Answer: {answer_a0}
- Reasoning: {reasoning_a0}
- Confidence: {confidence_a0}

Agent B (Negative):
- Answer: {answer_b0}
- Reasoning: {reasoning_b0}
- Confidence: {confidence_b0}

=== ROUND 1: First Debate ===
Agent A:
- Answer: {answer_a1} (Changed: {changed_a1})
- Reasoning: {reasoning_a1}
- Reflection: {reflection_a1}
- Confidence: {confidence_a1}

Agent B:
- Answer: {answer_b1} (Changed: {changed_b1})
- Reasoning: {reasoning_b1}
- Reflection: {reflection_b1}
- Confidence: {confidence_b1}

=== ROUND 2: Final Positions ===
Agent A:
- Answer: {answer_a2} (Changed from R1: {changed_a2})
- Reasoning: {reasoning_a2}
- Final Reflection: {final_reflection_a2}
- Confidence: {confidence_a2}

Agent B:
- Answer: {answer_b2} (Changed from R1: {changed_b2})
- Reasoning: {reasoning_b2}
- Final Reflection: {final_reflection_b2}
- Confidence: {confidence_b2}

**Your Task as Judge:**

1. **Evaluate Argument Quality (NOT just agreement):**
   - Logical Coherence (40%): Are the arguments logically sound?
   - Cultural Sensitivity (30%): Do they demonstrate cultural understanding?
   - Evidence Sufficiency (20%): Are the reasons well-supported?
   - Reasoning Depth (10%): How deep is the analysis?

2. **Identify Key Points:**
   - Where do they agree/disagree?
   - What are the strongest arguments on each side?
   - Were there any critical insights during the debate?

3. **Make Final Decision:**
   - Choose the answer with the strongest overall argument
   - This may NOT be the answer both agents converged to
   - Explain your reasoning thoroughly

4. **Assess Confidence:**
   - How confident are you in this decision?
   - What factors create uncertainty?

**Output Format:**
Final_Answer: [1/2/3/4]
Decision_Reasoning: [Your comprehensive analysis in 4-6 sentences, explaining:
  - Why this answer is best
  - Which arguments were most convincing
  - What cultural factors are most relevant
  - Any remaining uncertainties]
Argument_Quality_Assessment: [Brief evaluation of both agents' arguments]
Confidence: [0-100]

Please provide your final judgment:"""
```

---

## 实现方案

### 系统组件设计

#### 1. 核心类: MADDebateEngine

```python
class MADDebateEngine:
    """
    Multi-Agent Debate引擎

    职责:
    - 管理辩论流程
    - 调用模型生成
    - 解析和记录输出
    - 控制辩论轮次
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.debate_history = []

    def run_debate(self, question, context, true_answer):
        """
        运行完整的MAD流程

        Returns:
            {
                'final_answer': str,
                'correct': bool,
                'debate_history': list,
                'judge_reasoning': str
            }
        """
        # Round 0: 初始回答
        agent_a_r0 = self.generate_initial_answer(question, context, role='affirmative')
        agent_b_r0 = self.generate_initial_answer(question, context, role='negative')

        # Round 1: 第一轮辩论
        agent_a_r1 = self.generate_debate_round(
            question, context, agent_a_r0, agent_b_r0, role='affirmative', round_num=1
        )
        agent_b_r1 = self.generate_debate_round(
            question, context, agent_b_r0, agent_a_r0, role='negative', round_num=1
        )

        # Round 2: 第二轮辩论
        agent_a_r2 = self.generate_debate_round(
            question, context, agent_a_r1, agent_b_r1, role='affirmative', round_num=2,
            history={'r0': (agent_a_r0, agent_b_r0), 'r1': (agent_a_r1, agent_b_r1)}
        )
        agent_b_r2 = self.generate_debate_round(
            question, context, agent_b_r1, agent_a_r1, role='negative', round_num=2,
            history={'r0': (agent_a_r0, agent_b_r0), 'r1': (agent_a_r1, agent_b_r1)}
        )

        # Final Decision: 裁判决策
        final_decision = self.generate_judge_decision(
            question, context,
            r0=(agent_a_r0, agent_b_r0),
            r1=(agent_a_r1, agent_b_r1),
            r2=(agent_a_r2, agent_b_r2)
        )

        # 记录完整历史
        self.debate_history = {
            'round_0': {'agent_a': agent_a_r0, 'agent_b': agent_b_r0},
            'round_1': {'agent_a': agent_a_r1, 'agent_b': agent_b_r1},
            'round_2': {'agent_a': agent_a_r2, 'agent_b': agent_b_r2},
            'final_decision': final_decision
        }

        return {
            'final_answer': final_decision['answer'],
            'correct': final_decision['answer'] == true_answer,
            'debate_history': self.debate_history,
            'judge_reasoning': final_decision['reasoning']
        }

    def generate_initial_answer(self, question, context, role):
        """生成初始回答"""
        # 根据role选择prompt模板
        # 调用模型生成
        # 解析输出
        pass

    def generate_debate_round(self, question, context, my_prev, other_prev, role, round_num, history=None):
        """生成辩论轮次回答"""
        # 构建包含历史的prompt
        # 调用模型生成
        # 解析输出（包括reflection）
        pass

    def generate_judge_decision(self, question, context, r0, r1, r2):
        """生成裁判决策"""
        # 构建包含完整历史的prompt
        # 调用模型生成
        # 解析最终决策
        pass

    def parse_model_output(self, output_text, expected_fields):
        """解析模型输出"""
        # 使用正则表达式或简单解析
        # 提取Answer, Reasoning, Confidence等字段
        pass
```

---

#### 2. 主评估脚本: eval_mad.py

```python
# 伪代码结构

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=int, required=True,
                       help='1: LLaMA 3.1, 2: Qwen 2.5')
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    # 根据model_type加载模型
    if args.model_type == 1:
        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.model_type == 2:
        model_path = "Qwen/Qwen2.5-7B-Instruct"
    else:
        raise ValueError("model_type must be 1 or 2")

    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载数据
    with open(args.data_file, 'r') as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

    # 初始化MAD引擎
    mad_engine = MADDebateEngine(model, tokenizer, device='cuda')

    # 运行评估
    results = []
    for idx, item in enumerate(tqdm(data)):
        result = mad_engine.run_debate(
            question=item['instruction'],
            context=item['input'],
            true_answer=item['output']
        )
        results.append(result)

    # 保存结果
    save_results(results, args.output_dir)

    # 打印统计
    print_statistics(results)

if __name__ == '__main__':
    main()
```

---

#### 3. Shell脚本: run_eval_mad.sh

```bash
#!/bin/bash

# Multi-Agent Debate (MAD) 评估脚本
#
# 参数说明:
#   $1: MODEL_TYPE - 1=LLaMA 3.1, 2=Qwen 2.5
#   $2: DATA_ID - 数据集编号 (1=unified, 2=CulturalBench, 3=NORMAD, 4=CultureLLM)
#   $3: MAX_SAMPLES (可选) - 最大样本数，用于快速测试

MODEL_TYPE=${1:-1}  # 默认使用LLaMA
DATA_ID=${2:-2}     # 默认使用CulturalBench
MAX_SAMPLES=${3:-""}

# 模型路径映射
if [ "$MODEL_TYPE" -eq 1 ]; then
    MODEL_NAME="llama3.1-8b"
    MODEL_PATH="/path/to/Meta-Llama-3.1-8B-Instruct"
elif [ "$MODEL_TYPE" -eq 2 ]; then
    MODEL_NAME="qwen2.5-7b"
    MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
else
    echo "错误: MODEL_TYPE 必须是 1 (LLaMA) 或 2 (Qwen)"
    exit 1
fi

# 数据集路径映射
case $DATA_ID in
    1)
        DATA_FILE="/path/to/unified_all_datasets.json"
        DATA_NAME="unified"
        ;;
    2)
        DATA_FILE="/path/to/CulturalBench_merge_gen.json"
        DATA_NAME="culturalbench"
        ;;
    3)
        DATA_FILE="/path/to/normad_merge_gen.json"
        DATA_NAME="normad"
        ;;
    4)
        DATA_FILE="/path/to/cultureLLM_merge_gen.json"
        DATA_NAME="culturellm"
        ;;
    *)
        echo "错误: DATA_ID 必须是 1-4"
        exit 1
        ;;
esac

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results/mad_${MODEL_NAME}_${DATA_NAME}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# 日志文件
LOG_FILE="${OUTPUT_DIR}/eval.log"

echo "========================================" | tee -a $LOG_FILE
echo "Multi-Agent Debate (MAD) Evaluation" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Model Type: $MODEL_TYPE ($MODEL_NAME)" | tee -a $LOG_FILE
echo "Model Path: $MODEL_PATH" | tee -a $LOG_FILE
echo "Data ID: $DATA_ID ($DATA_NAME)" | tee -a $LOG_FILE
echo "Data File: $DATA_FILE" | tee -a $LOG_FILE
echo "Output Dir: $OUTPUT_DIR" | tee -a $LOG_FILE
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples: $MAX_SAMPLES (test mode)" | tee -a $LOG_FILE
fi
echo "========================================" | tee -a $LOG_FILE

# 运行评估
python eval_mad.py \
    --model_type $MODEL_TYPE \
    --data_file $DATA_FILE \
    --output_dir $OUTPUT_DIR \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
    2>&1 | tee -a $LOG_FILE

echo "========================================" | tee -a $LOG_FILE
echo "Evaluation completed!" | tee -a $LOG_FILE
echo "Results saved to: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
```

---

### 输出格式规范

#### 单样本结果JSON

```json
{
  "question": "Give me the answer from 1 to 4: ...",
  "context": "This question is for ...",
  "true_answer": "2",

  "debate_history": {
    "round_0": {
      "agent_a": {
        "answer": "2",
        "reasoning": "...",
        "confidence": 75
      },
      "agent_b": {
        "answer": "3",
        "reasoning": "...",
        "confidence": 60
      }
    },
    "round_1": {
      "agent_a": {
        "answer": "2",
        "reasoning": "...",
        "reflection": "...",
        "confidence": 85,
        "changed": false
      },
      "agent_b": {
        "answer": "2",
        "reasoning": "...",
        "reflection": "...",
        "confidence": 70,
        "changed": true
      }
    },
    "round_2": {
      "agent_a": {
        "answer": "2",
        "reasoning": "...",
        "final_reflection": "...",
        "confidence": 90,
        "changed": false
      },
      "agent_b": {
        "answer": "2",
        "reasoning": "...",
        "final_reflection": "...",
        "confidence": 80,
        "changed": false
      }
    },
    "final_decision": {
      "answer": "2",
      "reasoning": "...",
      "argument_quality": "...",
      "confidence": 85
    }
  },

  "final_answer": "2",
  "correct": true,

  "debate_metrics": {
    "convergence_round": 1,
    "total_changes": 1,
    "initial_agreement": false,
    "final_agreement": true
  }
}
```

---

## 评估指标

### 基础指标

1. **准确率 (Accuracy)**
   ```
   Accuracy = Correct Predictions / Total Samples
   ```

2. **置信度 (Confidence)**
   - 平均置信度
   - 正确预测的平均置信度
   - 错误预测的平均置信度

---

### MAD特定指标

3. **收敛率 (Convergence Rate)**
   ```
   Convergence Rate = Samples with Agreement / Total Samples
   ```
   - 初始收敛率 (Round 0)
   - 第一轮收敛率 (Round 1)
   - 第二轮收敛率 (Round 2)

4. **答案变化率 (Answer Change Rate)**
   ```
   Change Rate = Samples with Answer Changes / Total Samples
   ```
   - Agent A 变化率
   - Agent B 变化率
   - 总体变化率

5. **辩论轮数分布**
   - 初始一致: X%
   - 第1轮收敛: Y%
   - 第2轮收敛: Z%
   - 最终不一致: W%

6. **置信度变化**
   ```
   Confidence Gain = Final Confidence - Initial Confidence
   ```
   - 平均置信度增长
   - 正确答案的置信度增长
   - 错误答案的置信度变化

---

### 对比指标

7. **MAD增益 (MAD Gain)**
   ```
   MAD Gain = Accuracy_MAD - max(Accuracy_AgentA, Accuracy_AgentB)
   ```
   衡量MAD相比单个agent的提升

8. **反思有效性 (Reflection Effectiveness)**
   ```
   Reflection Effectiveness =
     (Correct Changes + Correct Holds) / Total Samples
   ```
   - Correct Changes: 从错误改为正确
   - Correct Holds: 正确答案坚持不变

---

### 评估报告示例

```
========================================
MAD Evaluation Report
========================================
Model: LLaMA 3.1-8B-Instruct
Dataset: CulturalBench (1,000 samples)
Date: 2026-03-13

--- Basic Metrics ---
Accuracy: 78.5%
Average Confidence: 82.3

--- Baseline Comparison ---
Agent A (Round 0) Accuracy: 72.1%
Agent B (Round 0) Accuracy: 69.8%
MAD Gain: +6.4%

--- Convergence Analysis ---
Initial Agreement (R0): 58.2%
  - Both Correct: 45.3%
  - Both Wrong: 12.9%
Round 1 Convergence: 76.5%
Round 2 Convergence: 89.3%
Final Disagreement: 10.7%

--- Answer Changes ---
Agent A Changes:
  - R0→R1: 18.5% (12.3% correct changes)
  - R1→R2: 6.2% (4.1% correct changes)
Agent B Changes:
  - R0→R1: 24.7% (16.8% correct changes)
  - R1→R2: 8.3% (5.5% correct changes)

--- Confidence Dynamics ---
Initial Avg Confidence: 68.5
Final Avg Confidence: 82.3
Confidence Gain: +13.8

Correct Predictions:
  - Initial: 71.2
  - Final: 86.7
Wrong Predictions:
  - Initial: 63.8
  - Final: 65.4

--- Reflection Effectiveness ---
Correct Changes: 14.2%
Correct Holds: 64.3%
Wrong Changes: 3.8%
Wrong Holds: 17.7%
Reflection Effectiveness: 78.5%

========================================
```

---

## 使用指南

### 快速开始

#### 1. 在CulturalBench上使用LLaMA 3.1评估

```bash
# 完整评估
bash run_eval_mad.sh 1 2

# 快速测试（只评估100个样本）
bash run_eval_mad.sh 1 2 100
```

#### 2. 在NORMAD上使用Qwen 2.5评估

```bash
# 完整评估
bash run_eval_mad.sh 2 3

# 快速测试
bash run_eval_mad.sh 2 3 50
```

#### 3. 批量评估所有数据集

```bash
# LLaMA 3.1
for data_id in 2 3 4; do
    bash run_eval_mad.sh 1 $data_id
done

# Qwen 2.5
for data_id in 2 3 4; do
    bash run_eval_mad.sh 2 $data_id
done
```

---

### 参数详细说明

#### run_eval_mad.sh参数

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| MODEL_TYPE | 模型类型 | 1 (LLaMA 3.1) / 2 (Qwen 2.5) | 1 |
| DATA_ID | 数据集编号 | 1-4 | 2 |
| MAX_SAMPLES | 最大样本数 | 任意正整数 | 无限制 |

#### 数据集编号映射

| DATA_ID | 数据集名称 | 文件名 |
|---------|-----------|--------|
| 1 | Unified All | unified_all_datasets.json |
| 2 | CulturalBench | CulturalBench_merge_gen.json |
| 3 | NORMAD | normad_merge_gen.json |
| 4 | CultureLLM | cultureLLM_merge_gen.json |

---

### 结果解读

#### 输出文件结构

```
results/mad_llama3.1_culturalbench_20260313_143022/
├── eval.log                    # 评估日志
├── detailed_results.json       # 详细结果（每个样本）
├── summary_statistics.json     # 统计摘要
├── debate_analysis.json        # 辩论分析
└── convergence_report.txt      # 收敛报告
```

#### 关键结果文件

**detailed_results.json**: 每个样本的完整辩论历史
**summary_statistics.json**: 汇总统计
```json
{
  "accuracy": 0.785,
  "total_samples": 1000,
  "correct_predictions": 785,
  "mad_gain": 0.064,
  "convergence_rate": 0.893,
  "avg_confidence": 82.3
}
```

**debate_analysis.json**: 辩论过程分析
```json
{
  "convergence": {
    "round_0": 0.582,
    "round_1": 0.765,
    "round_2": 0.893
  },
  "answer_changes": {
    "agent_a": {"r0_r1": 0.185, "r1_r2": 0.062},
    "agent_b": {"r0_r1": 0.247, "r1_r2": 0.083}
  },
  "reflection_effectiveness": 0.785
}
```

---

## 技术细节

### 显存优化策略

#### 策略1: 单模型实例复用（推荐）

```python
# 只加载一次模型，通过不同prompt调用
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用半精度
    device_map="auto"           # 自动分配设备
)

# 所有agent共享同一个模型实例
agent_a_output = generate_with_prompt(model, prompt_a)
agent_b_output = generate_with_prompt(model, prompt_b)
judge_output = generate_with_prompt(model, prompt_judge)
```

**优点**：
- 显存占用最小（约15-20GB for 8B model）
- 实现简单

**缺点**：
- 无法并行生成
- 总时间较长

---

#### 策略2: 批量生成优化

```python
# 将多个prompt打包成batch
batch_prompts = [prompt_a, prompt_b]
batch_outputs = model.generate(
    input_ids=batch_input_ids,
    attention_mask=batch_attention_mask,
    max_new_tokens=512,
    do_sample=False
)
```

**优点**：
- 减少模型调用次数
- 提升吞吐量

**缺点**：
- 需要padding对齐
- 显存占用稍高

---

### 并行策略

#### 样本级并行

```python
# 使用DataLoader进行样本级并行
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=1,  # MAD需要顺序处理
    num_workers=0  # 不使用多进程
)

for batch in dataloader:
    result = mad_engine.run_debate(batch)
```

#### GPU级并行（多卡）

```python
# 如果有多张GPU，可以并行处理不同样本
# GPU 0: 处理样本 0, 2, 4, ...
# GPU 1: 处理样本 1, 3, 5, ...

import torch.distributed as dist

def parallel_evaluate(rank, world_size, data):
    # 每个进程处理一部分数据
    local_data = data[rank::world_size]

    # 在本地GPU上运行
    device = f'cuda:{rank}'
    model = load_model(device)

    results = []
    for item in local_data:
        result = mad_engine.run_debate(item)
        results.append(result)

    return results
```

---

### 错误处理

#### 生成失败处理

```python
def safe_generate(model, prompt, max_retries=3):
    """安全生成，包含重试机制"""
    for attempt in range(max_retries):
        try:
            output = model.generate(...)
            return output
        except RuntimeError as e:
            if "out of memory" in str(e):
                # 清理显存
                torch.cuda.empty_cache()
                # 减少max_new_tokens
                max_new_tokens = max_new_tokens // 2
            else:
                raise e

    # 如果所有重试都失败，返回默认值
    return None
```

#### 解析失败处理

```python
def parse_with_fallback(output_text, expected_format):
    """解析输出，包含fallback"""
    try:
        # 尝试标准解析
        parsed = parse_standard(output_text)
        return parsed
    except ParseError:
        # 尝试宽松解析
        try:
            parsed = parse_relaxed(output_text)
            return parsed
        except:
            # 返回默认值
            return {
                'answer': '1',  # 默认答案
                'reasoning': output_text,  # 原始输出
                'confidence': 50,  # 低置信度
                'parse_failed': True
            }
```

---

### 性能优化建议

1. **使用KV Cache**：
   ```python
   outputs = model.generate(
       ...,
       use_cache=True,  # 启用KV cache
       past_key_values=past_kv  # 复用之前的cache
   )
   ```

2. **减少最大生成长度**：
   ```python
   # 根据任务调整max_new_tokens
   # 初始回答: 256 tokens
   # 辩论轮次: 384 tokens
   # 裁判决策: 512 tokens
   ```

3. **使用Flash Attention**（如果支持）：
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       attn_implementation="flash_attention_2"
   )
   ```

4. **批量处理相同轮次**：
   ```python
   # 同时生成Agent A和Agent B的Round 0
   batch_outputs = generate_batch([prompt_a_r0, prompt_b_r0])
   ```

---

## 预期效果与分析

### 理论预期

基于Multi-Agent Debate的相关研究（如MAD论文），我们预期：

1. **准确率提升**：
   - 相比单模型：+5-10%
   - 相比简单投票：+2-5%

2. **置信度校准**：
   - 正确答案置信度提升
   - 错误答案置信度下降
   - 整体校准误差减小

3. **文化理解深度**：
   - 通过反思机制发现初始遗漏的文化细节
   - 减少文化刻板印象导致的错误

---

### 可能的挑战

1. **收敛困难**：
   - 某些问题可能无法在2轮内收敛
   - 解决方案：增加轮次或改进prompt

2. **过度自信**：
   - 模型可能在辩论中变得更自信，即使答案错误
   - 解决方案：裁判需要独立评估，不受置信度影响

3. **计算成本**：
   - MAD需要多次生成，时间成本高
   - 解决方案：并行优化、缓存优化

---

### 消融实验建议

为了验证MAD的有效性，建议进行以下消融实验：

1. **Single Agent Baseline**：
   - 只使用Agent A的Round 0答案
   - 只使用Agent B的Round 0答案

2. **Simple Voting**：
   - Agent A和Agent B的Round 0答案投票

3. **No Reflection**：
   - 移除反思机制，只进行简单的"看到对方答案"

4. **No Judge**：
   - 直接使用Round 2的投票结果，不经过裁判

5. **Different Round Numbers**：
   - 1轮辩论 vs 2轮辩论 vs 3轮辩论

---

## 未来扩展方向

### 短期扩展（1-2周）

1. **支持更多模型**：
   - 添加MODEL_TYPE=3: Mistral
   - 添加MODEL_TYPE=4: GPT-4（API调用）

2. **自适应轮次**：
   - 根据置信度和收敛情况动态决定是否继续辩论

3. **可视化工具**：
   - 辩论过程可视化
   - 答案变化轨迹图

---

### 中期扩展（1-2个月）

1. **异质模型辩论**：
   - LLaMA vs Qwen跨模型辩论
   - 更丰富的视角

2. **专家分工**：
   - 不同agent专注不同文化区域
   - 基于文化背景的角色分配

3. **元学习优化**：
   - 学习最优的prompt模板
   - 学习最优的轮次数

---

### 长期研究方向（3-6个月）

1. **端到端训练**：
   - 训练专门的"辩论模型"
   - 强化学习优化辩论策略

2. **文化知识库集成**：
   - 集成外部文化知识库
   - 检索增强的辩论

3. **多模态辩论**：
   - 结合图像、视频等多模态信息
   - 更丰富的文化理解

---

## 附录

### A. 相关研究

1. **Multi-Agent Debate (MAD)**
   - Paper: "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
   - 核心思想：多个agent通过辩论提升推理质量

2. **Self-Consistency**
   - Paper: "Self-Consistency Improves Chain of Thought Reasoning"
   - 方法：多次采样，投票决定

3. **Reflexion**
   - Paper: "Reflexion: Language Agents with Verbal Reinforcement Learning"
   - 核心：通过反思改进推理

---

### B. Prompt工程技巧

1. **角色设定要清晰但不极端**
   - ✅ "You are a critical thinker..."
   - ❌ "You must disagree with everything..."

2. **鼓励反思而非固执**
   - ✅ "You may change your answer if convinced..."
   - ❌ "Defend your answer at all costs..."

3. **结构化输出便于解析**
   - 使用固定的标签：Answer:, Reasoning:, Confidence:
   - 避免自由格式输出

4. **文化敏感性提示**
   - 明确要求考虑文化差异
   - 避免文化刻板印象

---

### C. 常见问题 (FAQ)

**Q1: 为什么使用相同模型而不是不同模型？**
A: 相同模型通过不同prompt产生多样性，更容易控制和分析。未来可以扩展到异质模型。

**Q2: 2轮辩论够吗？**
A: 根据MAD论文，2-3轮通常足够。可以通过实验确定最优轮次。

**Q3: 裁判会不会总是选择置信度高的答案？**
A: 裁判的prompt强调"论证质量"而非"置信度"，应该能够独立判断。

**Q4: 如果两个agent始终不一致怎么办？**
A: 裁判会综合评估双方论证质量，给出最终答案。不一致本身也是有价值的信息。

**Q5: 计算成本如何？**
A: 相比单次推理，MAD需要约7次生成（2+2+2+1），时间成本约为7倍。但准确率提升可能值得这个代价。

---

## 总结

本方案设计了一个完整的Multi-Agent Debate (MAD)评估系统，用于提升文化理解任务的推理质量。核心特点包括：

✅ **纯Prompt-Based**：无需训练，灵活部署
✅ **三角色设计**：正方、反方、裁判协同决策
✅ **反思机制**：每轮辩论都包含批判性反思
✅ **易于使用**：通过简单的shell脚本即可运行
✅ **可扩展性**：支持多种模型和数据集

**预期效果**：
- 准确率提升5-10%
- 更好的置信度校准
- 更深入的文化理解

**下一步**：
1. 实现核心代码（eval_mad.py）
2. 在小规模数据上测试
3. 优化prompt模板
4. 批量评估并分析结果

---

*文档版本*: v1.0
*创建日期*: 2026-03-13
*最后更新*: 2026-03-13
