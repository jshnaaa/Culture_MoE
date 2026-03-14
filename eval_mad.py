#!/usr/bin/env python3
"""
Multi-Agent Debate (MAD) 评估脚本

实现基于prompt的多代理辩论系统，用于文化理解任务评估。

使用方法：
    python eval_mad.py \
        --model_type 1 \
        --data_file /path/to/data.json \
        --output_dir /path/to/output \
        [--max_samples 100]

模型类型：
    1: LLaMA 3.1-8B-Instruct
    2: Qwen 2.5-7B-Instruct
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Prompt模板
# ============================================================================

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


PROMPT_AGENT_B_ROUND0 = """You are a cultural understanding expert with a focus on verification and validation. Your task is to independently answer the following question by carefully analyzing all aspects.

**Question:**
{instruction}

**Context:**
{input}

**Instructions:**
1. Carefully read the question and context
2. Think about the cultural factors involved
3. Consider all options systematically (1-4)
4. Choose the answer that best fits the cultural context
5. Explain your reasoning clearly and objectively

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your detailed reasoning in 3-5 sentences, focusing on cultural accuracy]
Confidence: [0-100]

Please provide your answer now:"""


PROMPT_AGENT_A_ROUND1 = """You are continuing the collaborative discussion to find the correct answer. You have seen another perspective.

**Original Question:**
{instruction}

**Context:**
{input}

**Your Previous Answer (Round 0):**
- Answer: {answer_a0}
- Reasoning: {reasoning_a0}
- Confidence: {confidence_a0}

**Alternative Perspective (Agent B's Answer):**
- Answer: {answer_b0}
- Reasoning: {reasoning_b0}
- Confidence: {confidence_b0}

**Instructions:**
1. **Goal: Find the most culturally accurate answer**
   - Review both perspectives objectively
   - Consider what cultural evidence each provides

2. **Update your answer if needed:**
   - Keep your answer if you still believe it's correct
   - Change your answer if the other perspective has stronger cultural evidence
   - Explain your reasoning clearly

3. **Be honest about your confidence:**
   - Higher confidence if evidence is strong
   - Lower confidence if uncertain

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your updated reasoning, focusing on cultural accuracy]
Reflection: [Why this answer is most culturally accurate, considering both perspectives]
Confidence: [0-100]
Changed: [Yes/No]

Please provide your response:"""


PROMPT_AGENT_B_ROUND1 = """You are continuing the collaborative discussion to find the correct answer. You have seen another perspective.

**Original Question:**
{instruction}

**Context:**
{input}

**Your Previous Answer (Round 0):**
- Answer: {answer_b0}
- Reasoning: {reasoning_b0}
- Confidence: {confidence_b0}

**Alternative Perspective (Agent A's Answer):**
- Answer: {answer_a0}
- Reasoning: {reasoning_a0}
- Confidence: {confidence_a0}

**Instructions:**
1. **Goal: Find the most culturally accurate answer**
   - Review both perspectives objectively
   - Consider what cultural evidence each provides

2. **Update your answer if needed:**
   - Keep your answer if you still believe it's correct
   - Change your answer if the other perspective has stronger cultural evidence
   - Explain your reasoning clearly

3. **Be honest about your confidence:**
   - Higher confidence if evidence is strong
   - Lower confidence if uncertain

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your updated reasoning, focusing on cultural accuracy]
Reflection: [Why this answer is most culturally accurate, considering both perspectives]
Confidence: [0-100]
Changed: [Yes/No]

Please provide your response:"""


PROMPT_AGENT_A_ROUND2 = """This is the final round of collaborative discussion. Review all perspectives and provide your final answer.

**Original Question:**
{instruction}

**Context:**
{input}

**Discussion History:**

Round 0 (Initial):
- Your answer: {answer_a0} (Confidence: {confidence_a0})
- Their answer: {answer_b0} (Confidence: {confidence_b0})

Round 1 (First Discussion):
- Your answer: {answer_a1} (Changed: {changed_a1}, Confidence: {confidence_a1})
- Their answer: {answer_b1} (Changed: {changed_b1}, Confidence: {confidence_b1})
- Your reflection: {reflection_a1}
- Their reflection: {reflection_b1}

**Instructions:**
1. **Review the full discussion**
   - Consider all rounds and perspectives
   - What cultural evidence emerged?

2. **Make your final decision:**
   - Choose the most culturally accurate answer
   - Explain your reasoning
   - Indicate your confidence level

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your final reasoning, emphasizing cultural accuracy]
Final_Reflection: [Why this answer is most culturally accurate based on the full discussion]
Confidence: [0-100]
Changed_From_R1: [Yes/No]

Please provide your final response:"""


PROMPT_AGENT_B_ROUND2 = """This is the final round of collaborative discussion. Review all perspectives and provide your final answer.

**Original Question:**
{instruction}

**Context:**
{input}

**Discussion History:**

Round 0 (Initial):
- Your answer: {answer_b0} (Confidence: {confidence_b0})
- Their answer: {answer_a0} (Confidence: {confidence_a0})

Round 1 (First Discussion):
- Your answer: {answer_b1} (Changed: {changed_b1}, Confidence: {confidence_b1})
- Their answer: {answer_a1} (Changed: {changed_a1}, Confidence: {confidence_a1})
- Your reflection: {reflection_b1}
- Their reflection: {reflection_a1}

**Instructions:**
1. **Review the full discussion**
   - Consider all rounds and perspectives
   - What cultural evidence emerged?

2. **Make your final decision:**
   - Choose the most culturally accurate answer
   - Explain your reasoning
   - Indicate your confidence level

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your final reasoning, emphasizing cultural accuracy]
Final_Reflection: [Why this answer is most culturally accurate based on the full discussion]
Confidence: [0-100]
Changed_From_R1: [Yes/No]

Please provide your final response:"""


PROMPT_JUDGE_FINAL = """You are a judge. Two cultural experts have discussed a question and reached their final answers.

**Question:**
{instruction}

**Context:**
{input}

**Final Answers (Round 2):**
- Agent A: Answer {answer_a2} (Confidence: {confidence_a2})
  Reasoning: {reasoning_a2}

- Agent B: Answer {answer_b2} (Confidence: {confidence_b2})
  Reasoning: {reasoning_b2}

**Your Task:**
Select which agent's answer is most likely correct.

**Decision Rules:**
1. If both agents give the SAME answer → Choose that answer
2. If agents give DIFFERENT answers → Choose the one with:
   - Stronger cultural reasoning
   - Higher confidence
   - More convincing evidence

**Output Format:**
Selected_Agent: [A/B]
Final_Answer: [1/2/3/4]
Reasoning: [1-2 sentences: why you chose this agent's answer]
Confidence: [0-100]

Your decision:"""


# ============================================================================
# 辅助函数
# ============================================================================

def parse_model_output(output_text: str, expected_fields: List[str]) -> Dict[str, str]:
    """
    解析模型输出，提取结构化字段

    Args:
        output_text: 模型生成的文本
        expected_fields: 期望的字段列表，如 ['Answer', 'Reasoning', 'Confidence']

    Returns:
        包含提取字段的字典
    """
    result = {}

    for field in expected_fields:
        # 使用正则表达式提取字段
        # 匹配模式：Field: content 或 Field:\ncontent
        pattern = rf"{field}:\s*(.+?)(?=\n[A-Z][a-z_]+:|$)"
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)

        if match:
            content = match.group(1).strip()
            result[field] = content
        else:
            # 如果没有找到，尝试更宽松的匹配
            pattern_loose = rf"{field}[:\s]+(.+?)(?=\n\n|$)"
            match_loose = re.search(pattern_loose, output_text, re.DOTALL | re.IGNORECASE)
            if match_loose:
                result[field] = match_loose.group(1).strip()
            else:
                result[field] = ""

    return result


def extract_answer(text: str) -> str:
    """
    从文本中提取答案（1-4）

    Args:
        text: 包含答案的文本

    Returns:
        答案字符串 ('1', '2', '3', '4') 或 '1' (默认)
    """
    # 尝试多种模式
    patterns = [
        r'Answer:\s*([1-4])',
        r'Final_Answer:\s*([1-4])',
        r'^([1-4])$',
        r'\b([1-4])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)

    # 如果都没找到，返回默认值
    return '1'


def extract_confidence(text: str) -> int:
    """
    从文本中提取置信度（0-100）

    Args:
        text: 包含置信度的文本

    Returns:
        置信度整数
    """
    pattern = r'Confidence:\s*(\d+)'
    match = re.search(pattern, text)

    if match:
        conf = int(match.group(1))
        return max(0, min(100, conf))  # 限制在0-100范围
    else:
        return 50  # 默认置信度


def extract_yes_no(text: str) -> bool:
    """
    从文本中提取Yes/No答案

    Args:
        text: 包含Yes/No的文本

    Returns:
        True (Yes) 或 False (No)
    """
    text_lower = text.lower()
    if 'yes' in text_lower:
        return True
    elif 'no' in text_lower:
        return False
    else:
        return False  # 默认为No


# ============================================================================
# MAD Debate Engine
# ============================================================================

class MADDebateEngine:
    """
    Multi-Agent Debate引擎

    管理完整的辩论流程，包括：
    - Round 0: 初始回答
    - Round 1: 第一轮辩论（含反思）
    - Round 2: 第二轮辩论（深度反思）
    - Final: 裁判决策
    """

    def __init__(self, model, tokenizer, device: str = 'cuda', verbose: bool = False):
        """
        初始化MAD引擎

        Args:
            model: 预训练的语言模型
            tokenizer: 对应的tokenizer
            device: 运行设备
            verbose: 是否输出详细日志
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose
        self.model.eval()
        # 添加线程锁,保护CUDA操作
        self._generation_lock = threading.Lock()

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False
    ) -> str:
        """
        通用生成函数

        Args:
            prompt: 输入prompt
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样

        Returns:
            生成的文本
        """
        # Tokenize (可以并行,不需要锁)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 确保attention_mask存在
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

        # Generate (使用锁保护CUDA操作,确保线程安全)
        with self._generation_lock:
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if do_sample else 1.0,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_beams=1,
                        early_stopping=True,
                        use_cache=True  # 启用KV cache加速
                    )
                except Exception as e:
                    print(f"\n⚠️  Generation error: {str(e)}")
                    raise

        # Decode只生成的部分
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

    def generate_initial_answer(
        self,
        instruction: str,
        input_text: str,
        role: str
    ) -> Dict[str, any]:
        """
        生成初始回答（Round 0）

        Args:
            instruction: 问题指令
            input_text: 问题上下文
            role: 角色 ('affirmative' 或 'negative')

        Returns:
            包含answer, reasoning, confidence的字典
        """
        # 选择prompt模板
        if role == 'affirmative':
            prompt_template = PROMPT_AGENT_A_ROUND0
        else:
            prompt_template = PROMPT_AGENT_B_ROUND0

        # 构建prompt
        prompt = prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else "(No additional context)"
        )

        # 生成
        output_text = self.generate_response(prompt, max_new_tokens=384)

        # 解析输出
        parsed = parse_model_output(output_text, ['Answer', 'Reasoning', 'Confidence'])

        result = {
            'answer': extract_answer(parsed.get('Answer', output_text)),
            'reasoning': parsed.get('Reasoning', output_text),
            'confidence': extract_confidence(parsed.get('Confidence', '50')),
            'raw_output': output_text
        }

        return result

    def generate_debate_round(
        self,
        instruction: str,
        input_text: str,
        my_prev: Dict,
        other_prev: Dict,
        role: str,
        round_num: int,
        history: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        生成辩论轮次回答（Round 1 或 Round 2）

        Args:
            instruction: 问题指令
            input_text: 问题上下文
            my_prev: 自己上一轮的回答
            other_prev: 对方上一轮的回答
            role: 角色 ('affirmative' 或 'negative')
            round_num: 轮次号 (1 或 2)
            history: 完整历史（仅Round 2需要）

        Returns:
            包含answer, reasoning, reflection, confidence, changed的字典
        """
        # 选择prompt模板
        if round_num == 1:
            if role == 'affirmative':
                prompt_template = PROMPT_AGENT_A_ROUND1
                prompt = prompt_template.format(
                    instruction=instruction,
                    input=input_text if input_text else "(No additional context)",
                    answer_a0=my_prev['answer'],
                    reasoning_a0=my_prev['reasoning'],
                    confidence_a0=my_prev['confidence'],
                    answer_b0=other_prev['answer'],
                    reasoning_b0=other_prev['reasoning'],
                    confidence_b0=other_prev['confidence']
                )
            else:
                prompt_template = PROMPT_AGENT_B_ROUND1
                prompt = prompt_template.format(
                    instruction=instruction,
                    input=input_text if input_text else "(No additional context)",
                    answer_b0=my_prev['answer'],
                    reasoning_b0=my_prev['reasoning'],
                    confidence_b0=my_prev['confidence'],
                    answer_a0=other_prev['answer'],
                    reasoning_a0=other_prev['reasoning'],
                    confidence_a0=other_prev['confidence']
                )
        else:  # round_num == 2
            r0_my, r0_other = history['r0']
            r1_my, r1_other = history['r1']

            if role == 'affirmative':
                prompt_template = PROMPT_AGENT_A_ROUND2
                prompt = prompt_template.format(
                    instruction=instruction,
                    input=input_text if input_text else "(No additional context)",
                    answer_a0=r0_my['answer'],
                    confidence_a0=r0_my['confidence'],
                    answer_b0=r0_other['answer'],
                    confidence_b0=r0_other['confidence'],
                    answer_a1=r1_my['answer'],
                    changed_a1='Yes' if r1_my.get('changed', False) else 'No',
                    confidence_a1=r1_my['confidence'],
                    answer_b1=r1_other['answer'],
                    changed_b1='Yes' if r1_other.get('changed', False) else 'No',
                    confidence_b1=r1_other['confidence'],
                    reflection_a1=r1_my.get('reflection', 'N/A'),
                    reflection_b1=r1_other.get('reflection', 'N/A')
                )
            else:
                prompt_template = PROMPT_AGENT_B_ROUND2
                prompt = prompt_template.format(
                    instruction=instruction,
                    input=input_text if input_text else "(No additional context)",
                    answer_b0=r0_my['answer'],
                    confidence_b0=r0_my['confidence'],
                    answer_a0=r0_other['answer'],
                    confidence_a0=r0_other['confidence'],
                    answer_b1=r1_my['answer'],
                    changed_b1='Yes' if r1_my.get('changed', False) else 'No',
                    confidence_b1=r1_my['confidence'],
                    answer_a1=r1_other['answer'],
                    changed_a1='Yes' if r1_other.get('changed', False) else 'No',
                    confidence_a1=r1_other['confidence'],
                    reflection_b1=r1_my.get('reflection', 'N/A'),
                    reflection_a1=r1_other.get('reflection', 'N/A')
                )

        # 生成
        output_text = self.generate_response(prompt, max_new_tokens=512)

        # 解析输出
        if round_num == 1:
            expected_fields = ['Answer', 'Reasoning', 'Reflection', 'Confidence', 'Changed']
        else:
            expected_fields = ['Answer', 'Reasoning', 'Final_Reflection', 'Confidence', 'Changed_From_R1']

        parsed = parse_model_output(output_text, expected_fields)

        result = {
            'answer': extract_answer(parsed.get('Answer', output_text)),
            'reasoning': parsed.get('Reasoning', output_text),
            'confidence': extract_confidence(parsed.get('Confidence', '50')),
            'raw_output': output_text
        }

        if round_num == 1:
            result['reflection'] = parsed.get('Reflection', 'N/A')
            result['changed'] = extract_yes_no(parsed.get('Changed', 'No'))
        else:
            result['final_reflection'] = parsed.get('Final_Reflection', 'N/A')
            result['changed'] = extract_yes_no(parsed.get('Changed_From_R1', 'No'))

        return result

    def generate_judge_decision(
        self,
        instruction: str,
        input_text: str,
        r0: Tuple[Dict, Dict],
        r1: Tuple[Dict, Dict],
        r2: Tuple[Dict, Dict]
    ) -> Dict[str, any]:
        """
        生成裁判决策（Final Decision）

        Args:
            instruction: 问题指令
            input_text: 问题上下文
            r0: Round 0的(agent_a, agent_b)结果
            r1: Round 1的(agent_a, agent_b)结果
            r2: Round 2的(agent_a, agent_b)结果

        Returns:
            包含final_answer, reasoning, assessment, confidence的字典
        """
        agent_a_r0, agent_b_r0 = r0
        agent_a_r1, agent_b_r1 = r1
        agent_a_r2, agent_b_r2 = r2

        # 构建prompt
        prompt = PROMPT_JUDGE_FINAL.format(
            instruction=instruction,
            input=input_text if input_text else "(No additional context)",
            # Round 0
            answer_a0=agent_a_r0['answer'],
            reasoning_a0=agent_a_r0['reasoning'],
            confidence_a0=agent_a_r0['confidence'],
            answer_b0=agent_b_r0['answer'],
            reasoning_b0=agent_b_r0['reasoning'],
            confidence_b0=agent_b_r0['confidence'],
            # Round 1
            answer_a1=agent_a_r1['answer'],
            changed_a1='Yes' if agent_a_r1.get('changed', False) else 'No',
            reasoning_a1=agent_a_r1['reasoning'],
            reflection_a1=agent_a_r1.get('reflection', 'N/A'),
            confidence_a1=agent_a_r1['confidence'],
            answer_b1=agent_b_r1['answer'],
            changed_b1='Yes' if agent_b_r1.get('changed', False) else 'No',
            reasoning_b1=agent_b_r1['reasoning'],
            reflection_b1=agent_b_r1.get('reflection', 'N/A'),
            confidence_b1=agent_b_r1['confidence'],
            # Round 2
            answer_a2=agent_a_r2['answer'],
            changed_a2='Yes' if agent_a_r2.get('changed', False) else 'No',
            reasoning_a2=agent_a_r2['reasoning'],
            final_reflection_a2=agent_a_r2.get('final_reflection', 'N/A'),
            confidence_a2=agent_a_r2['confidence'],
            answer_b2=agent_b_r2['answer'],
            changed_b2='Yes' if agent_b_r2.get('changed', False) else 'No',
            reasoning_b2=agent_b_r2['reasoning'],
            final_reflection_b2=agent_b_r2.get('final_reflection', 'N/A'),
            confidence_b2=agent_b_r2['confidence']
        )

        # 生成
        output_text = self.generate_response(prompt, max_new_tokens=256)

        # 解析输出
        parsed = parse_model_output(
            output_text,
            ['Selected_Agent', 'Final_Answer', 'Reasoning', 'Confidence']
        )

        # 提取答案 - 优先从Final_Answer提取,否则根据Selected_Agent选择
        final_answer = extract_answer(parsed.get('Final_Answer', ''))

        # 如果Final_Answer提取失败,尝试根据Selected_Agent选择
        if final_answer == '1' and 'Selected_Agent' in parsed:
            selected = parsed['Selected_Agent'].strip().upper()
            if 'A' in selected:
                final_answer = agent_a_r2['answer']
            elif 'B' in selected:
                final_answer = agent_b_r2['answer']

        result = {
            'answer': final_answer,
            'reasoning': parsed.get('Reasoning', output_text[:200]),
            'selected_agent': parsed.get('Selected_Agent', 'N/A'),
            'confidence': extract_confidence(parsed.get('Confidence', '50')),
            'raw_output': output_text
        }

        return result

    def run_debate(
        self,
        instruction: str,
        input_text: str,
        true_answer: str
    ) -> Dict:
        """
        运行完整的MAD辩论流程 (并行优化版本)

        Args:
            instruction: 问题指令
            input_text: 问题上下文
            true_answer: 正确答案

        Returns:
            完整的辩论结果字典
        """
        # Round 0: 初始回答 (并行执行)
        if self.verbose:
            print("  [R0] Agent A & B generating in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                self.generate_initial_answer, instruction, input_text, 'affirmative'
            )
            future_b = executor.submit(
                self.generate_initial_answer, instruction, input_text, 'negative'
            )
            agent_a_r0 = future_a.result()
            agent_b_r0 = future_b.result()

        # Round 1: 第一轮辩论 (并行执行)
        if self.verbose:
            print("  [R1] Agent A & B generating in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                self.generate_debate_round,
                instruction, input_text, agent_a_r0, agent_b_r0,
                'affirmative', 1
            )
            future_b = executor.submit(
                self.generate_debate_round,
                instruction, input_text, agent_b_r0, agent_a_r0,
                'negative', 1
            )
            agent_a_r1 = future_a.result()
            agent_b_r1 = future_b.result()

        # Round 2: 第二轮辩论 (并行执行)
        if self.verbose:
            print("  [R2] Agent A & B generating in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                self.generate_debate_round,
                instruction, input_text, agent_a_r1, agent_b_r1,
                'affirmative', 2,
                {'r0': (agent_a_r0, agent_b_r0), 'r1': (agent_a_r1, agent_b_r1)}
            )
            future_b = executor.submit(
                self.generate_debate_round,
                instruction, input_text, agent_b_r1, agent_a_r1,
                'negative', 2,
                {'r0': (agent_b_r0, agent_a_r0), 'r1': (agent_b_r1, agent_a_r1)}
            )
            agent_a_r2 = future_a.result()
            agent_b_r2 = future_b.result()

        # Final Decision: 裁判决策 (串行执行)
        if self.verbose:
            print("  [Judge] Generating final decision...")
        final_decision = self.generate_judge_decision(
            instruction, input_text,
            r0=(agent_a_r0, agent_b_r0),
            r1=(agent_a_r1, agent_b_r1),
            r2=(agent_a_r2, agent_b_r2)
        )

        # 构建完整结果
        result = {
            'question': instruction,
            'context': input_text,
            'true_answer': true_answer,
            'debate_history': {
                'round_0': {
                    'agent_a': agent_a_r0,
                    'agent_b': agent_b_r0
                },
                'round_1': {
                    'agent_a': agent_a_r1,
                    'agent_b': agent_b_r1
                },
                'round_2': {
                    'agent_a': agent_a_r2,
                    'agent_b': agent_b_r2
                },
                'final_decision': final_decision
            },
            'final_answer': final_decision['answer'],
            'correct': final_decision['answer'] == true_answer,
            'debate_metrics': self._compute_metrics(
                agent_a_r0, agent_b_r0,
                agent_a_r1, agent_b_r1,
                agent_a_r2, agent_b_r2,
                final_decision
            )
        }

        return result

    def _compute_metrics(
        self,
        a0, b0, a1, b1, a2, b2, final
    ) -> Dict:
        """
        计算辩论指标

        Returns:
            包含各种指标的字典
        """
        # 初始一致性
        initial_agreement = (a0['answer'] == b0['answer'])

        # 收敛轮次
        convergence_round = None
        if initial_agreement:
            convergence_round = 0
        elif a1['answer'] == b1['answer']:
            convergence_round = 1
        elif a2['answer'] == b2['answer']:
            convergence_round = 2

        # 答案变化统计
        total_changes = 0
        if a1.get('changed', False):
            total_changes += 1
        if b1.get('changed', False):
            total_changes += 1
        if a2.get('changed', False):
            total_changes += 1
        if b2.get('changed', False):
            total_changes += 1

        # 最终一致性
        final_agreement = (a2['answer'] == b2['answer'])

        return {
            'convergence_round': convergence_round,
            'total_changes': total_changes,
            'initial_agreement': initial_agreement,
            'final_agreement': final_agreement,
            'confidence_gain_a': a2['confidence'] - a0['confidence'],
            'confidence_gain_b': b2['confidence'] - b0['confidence']
        }


# ============================================================================
# 数据加载和评估
# ============================================================================

def load_data(
    data_file: str,
    max_samples: Optional[int] = None,
    random_p: Optional[float] = None,
    random_seed: int = 42
) -> List[Dict]:
    """
    加载评估数据

    优先级规则：
    1. 如果 max_samples > 0：直接取前 max_samples 个样本，忽略 random_p
    2. 如果 max_samples = 0 或 None：使用 random_p 进行随机采样
       - random_p = 1.0：使用全部数据
       - random_p < 1.0：随机采样指定比例

    Args:
        data_file: 数据文件路径
        max_samples: 最大样本数，0表示不限制（此时使用random_p）
        random_p: 随机采样比例（0-1之间），1.0表示全部数据
        random_seed: 随机种子，保证可复现性

    Returns:
        数据列表
    """
    print(f"Loading data from: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_size = len(data)
    print(f"Original dataset size: {original_size}")

    # 优先级1：如果指定了max_samples且大于0，直接使用（忽略random_p）
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
        print(f"Using max_samples mode: taking first {max_samples} samples (random_p ignored)")
    # 优先级2：如果max_samples为0或None，使用random_p
    else:
        if random_p is not None:
            if not 0 < random_p <= 1.0:
                raise ValueError(f"random_p must be between 0 and 1, got {random_p}")

            if random_p == 1.0:
                # random_p=1.0 表示使用全部数据
                print(f"Using all data (random_p=1.0): {original_size} samples")
            else:
                # random_p<1.0 表示随机采样
                import random
                random.seed(random_seed)

                sample_size = int(original_size * random_p)
                sample_size = max(1, sample_size)  # 至少取1个样本

                # 随机采样
                data = random.sample(data, sample_size)
                print(f"Random sampling (random_p={random_p:.1%}): {sample_size} samples")
        else:
            # 如果两个参数都没指定，使用全部数据
            print(f"No sampling specified, using all data: {original_size} samples")

    print(f"Final dataset size: {len(data)} samples")
    return data


def generate_answers_report(results: List[Dict]) -> List[Dict]:
    """
    生成答案报告,包含每个样本的问题、各智能体回答、最终答案和正确性

    Args:
        results: 结果列表

    Returns:
        答案报告列表
    """
    report = []
    for r in results:
        # 跳过错误样本
        if 'error' in r:
            continue

        try:
            # 提取各轮次的答案
            agent_a_r0 = r['debate_history']['round_0']['agent_a'].get('answer', 'N/A')
            agent_a_r1 = r['debate_history']['round_1']['agent_a'].get('answer', 'N/A')
            agent_a_r2 = r['debate_history']['round_2']['agent_a'].get('answer', 'N/A')

            agent_b_r0 = r['debate_history']['round_0']['agent_b'].get('answer', 'N/A')
            agent_b_r1 = r['debate_history']['round_1']['agent_b'].get('answer', 'N/A')
            agent_b_r2 = r['debate_history']['round_2']['agent_b'].get('answer', 'N/A')

            report.append({
                'question': r['question'],
                'context': r.get('context', ''),
                'agent_a_answers': {
                    'round_0': agent_a_r0,
                    'round_1': agent_a_r1,
                    'round_2': agent_a_r2
                },
                'agent_b_answers': {
                    'round_0': agent_b_r0,
                    'round_1': agent_b_r1,
                    'round_2': agent_b_r2
                },
                'final_answer': r.get('final_answer', 'N/A'),
                'true_answer': r.get('true_answer', 'N/A'),
                'is_correct': r.get('correct', False)
            })
        except (KeyError, TypeError) as e:
            # 如果提取失败,跳过这个样本
            continue

    return report


def compute_eval_metrics(results: List[Dict]) -> Dict:
    """
    计算评估指标: 准确率、精确率、召回率、F1分数

    对于多分类问题,将其视为二分类(正确/错误)来计算指标

    Args:
        results: 结果列表

    Returns:
        评估指标字典
    """
    # 过滤掉错误样本
    valid_results = [r for r in results if 'error' not in r]
    total = len(valid_results)

    if total == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_samples': 0,
            'correct_predictions': 0
        }

    # 计算正确预测数
    correct = sum(1 for r in valid_results if r.get('correct', False))

    # 准确率
    accuracy = correct / total

    # 对于每个样本都有预测和真实标签的情况:
    # Precision = Recall = Accuracy
    # 因为 TP=correct, FP=(total-correct), FN=(total-correct), TN=0
    precision = accuracy
    recall = accuracy

    # F1分数
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_samples': total,
        'correct_predictions': correct
    }


def save_results(results: List[Dict], output_dir: str):
    """
    保存评估结果

    Args:
        results: 结果列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细结果
    detailed_path = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {detailed_path}")

    # 计算并保存统计摘要
    summary = compute_summary_statistics(results)
    summary_path = os.path.join(output_dir, 'summary_statistics.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary statistics saved to: {summary_path}")

    # 保存辩论分析
    analysis = compute_debate_analysis(results)
    analysis_path = os.path.join(output_dir, 'debate_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"Debate analysis saved to: {analysis_path}")

    # 保存生成的答案 (用户要求的格式)
    generated_answers = generate_answers_report(results)
    answers_path = os.path.join(output_dir, 'generated_answers.json')
    with open(answers_path, 'w', encoding='utf-8') as f:
        json.dump(generated_answers, f, indent=2, ensure_ascii=False)
    print(f"Generated answers saved to: {answers_path}")

    # 保存评估指标 (准确率/精确率/召回率/F1)
    eval_metrics = compute_eval_metrics(results)
    metrics_path = os.path.join(output_dir, 'eval_results.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    print(f"Evaluation metrics saved to: {metrics_path}")


def compute_summary_statistics(results: List[Dict]) -> Dict:
    """
    计算汇总统计

    Args:
        results: 结果列表

    Returns:
        统计字典
    """
    # 过滤掉错误样本,只统计成功的样本
    valid_results = [r for r in results if 'error' not in r]
    total = len(results)
    valid_total = len(valid_results)

    if valid_total == 0:
        # 如果所有样本都失败,返回0值统计
        return {
            'accuracy': 0.0,
            'total_samples': total,
            'valid_samples': 0,
            'failed_samples': total,
            'correct_predictions': 0,
            'avg_confidence': 0.0,
            'agent_a_accuracy': 0.0,
            'agent_b_accuracy': 0.0,
            'mad_gain': 0.0
        }

    correct = sum(1 for r in valid_results if r['correct'])
    accuracy = correct / valid_total if valid_total > 0 else 0

    # 置信度统计 (安全访问)
    confidences = []
    for r in valid_results:
        try:
            conf = r['debate_history']['final_decision']['confidence']
            confidences.append(conf)
        except (KeyError, TypeError):
            confidences.append(0)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Agent A和Agent B的初始准确率 (安全访问)
    agent_a_correct = 0
    agent_b_correct = 0
    for r in valid_results:
        try:
            if r['debate_history']['round_0']['agent_a'].get('answer') == r['true_answer']:
                agent_a_correct += 1
            if r['debate_history']['round_0']['agent_b'].get('answer') == r['true_answer']:
                agent_b_correct += 1
        except (KeyError, TypeError):
            pass

    agent_a_accuracy = agent_a_correct / valid_total if valid_total > 0 else 0
    agent_b_accuracy = agent_b_correct / valid_total if valid_total > 0 else 0

    # MAD增益
    mad_gain = accuracy - max(agent_a_accuracy, agent_b_accuracy)

    return {
        'accuracy': accuracy,
        'total_samples': total,
        'valid_samples': valid_total,
        'failed_samples': total - valid_total,
        'correct_predictions': correct,
        'avg_confidence': avg_confidence,
        'agent_a_accuracy': agent_a_accuracy,
        'agent_b_accuracy': agent_b_accuracy,
        'mad_gain': mad_gain
    }


def compute_debate_analysis(results: List[Dict]) -> Dict:
    """
    计算辩论过程分析

    Args:
        results: 结果列表

    Returns:
        分析字典
    """
    # 过滤掉错误样本
    valid_results = [r for r in results if 'error' not in r]
    total = len(valid_results)

    if total == 0:
        return {
            'convergence': {'round_0': 0, 'round_1': 0, 'round_2': 0},
            'answer_changes': {
                'agent_a': {'r0_r1': 0, 'r1_r2': 0},
                'agent_b': {'r0_r1': 0, 'r1_r2': 0}
            },
            'reflection_effectiveness': 0
        }

    # 收敛统计 (安全访问)
    convergence_counts = {0: 0, 1: 0, 2: 0, None: 0}
    for r in valid_results:
        try:
            conv_round = r['debate_metrics'].get('convergence_round')
            convergence_counts[conv_round] = convergence_counts.get(conv_round, 0) + 1
        except (KeyError, TypeError):
            convergence_counts[None] += 1

    convergence_rates = {
        'round_0': convergence_counts[0] / total if total > 0 else 0,
        'round_1': (convergence_counts[0] + convergence_counts[1]) / total if total > 0 else 0,
        'round_2': (convergence_counts[0] + convergence_counts[1] + convergence_counts[2]) / total if total > 0 else 0
    }

    # 答案变化统计 (安全访问)
    agent_a_r0_r1_changes = 0
    agent_a_r1_r2_changes = 0
    agent_b_r0_r1_changes = 0
    agent_b_r1_r2_changes = 0

    for r in valid_results:
        try:
            if r['debate_history']['round_1']['agent_a'].get('changed', False):
                agent_a_r0_r1_changes += 1
            if r['debate_history']['round_2']['agent_a'].get('changed', False):
                agent_a_r1_r2_changes += 1
            if r['debate_history']['round_1']['agent_b'].get('changed', False):
                agent_b_r0_r1_changes += 1
            if r['debate_history']['round_2']['agent_b'].get('changed', False):
                agent_b_r1_r2_changes += 1
        except (KeyError, TypeError):
            pass

    answer_changes = {
        'agent_a': {
            'r0_r1': agent_a_r0_r1_changes / total if total > 0 else 0,
            'r1_r2': agent_a_r1_r2_changes / total if total > 0 else 0
        },
        'agent_b': {
            'r0_r1': agent_b_r0_r1_changes / total if total > 0 else 0,
            'r1_r2': agent_b_r1_r2_changes / total if total > 0 else 0
        }
    }

    # 反思有效性 (安全访问)
    correct_changes = 0
    correct_holds = 0
    for r in valid_results:
        try:
            a0_ans = r['debate_history']['round_0']['agent_a'].get('answer')
            a2_ans = r['debate_history']['round_2']['agent_a'].get('answer')
            b0_ans = r['debate_history']['round_0']['agent_b'].get('answer')
            b2_ans = r['debate_history']['round_2']['agent_b'].get('answer')
            true_ans = r['true_answer']

            if a0_ans != true_ans and a2_ans == true_ans:
                correct_changes += 1
            if b0_ans != true_ans and b2_ans == true_ans:
                correct_changes += 1
            if a0_ans == true_ans and a2_ans == true_ans:
                correct_holds += 1
            if b0_ans == true_ans and b2_ans == true_ans:
                correct_holds += 1
        except (KeyError, TypeError):
            pass

    reflection_effectiveness = (correct_changes + correct_holds) / (total * 2) if total > 0 else 0

    return {
        'convergence': convergence_rates,
        'answer_changes': answer_changes,
        'reflection_effectiveness': reflection_effectiveness
    }


def print_statistics(results: List[Dict]):
    """
    打印统计信息

    Args:
        results: 结果列表
    """
    summary = compute_summary_statistics(results)
    analysis = compute_debate_analysis(results)

    print("\n" + "=" * 80)
    print("MAD Evaluation Report")
    print("=" * 80)

    print("\n--- Basic Metrics ---")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Correct Predictions: {summary['correct_predictions']}/{summary['total_samples']}")
    print(f"Average Confidence: {summary['avg_confidence']:.1f}")

    print("\n--- Baseline Comparison ---")
    print(f"Agent A (Round 0) Accuracy: {summary['agent_a_accuracy']:.2%}")
    print(f"Agent B (Round 0) Accuracy: {summary['agent_b_accuracy']:.2%}")
    print(f"MAD Gain: {summary['mad_gain']:+.2%}")

    print("\n--- Convergence Analysis ---")
    print(f"Round 0 Convergence: {analysis['convergence']['round_0']:.2%}")
    print(f"Round 1 Convergence: {analysis['convergence']['round_1']:.2%}")
    print(f"Round 2 Convergence: {analysis['convergence']['round_2']:.2%}")

    print("\n--- Answer Changes ---")
    print(f"Agent A R0→R1: {analysis['answer_changes']['agent_a']['r0_r1']:.2%}")
    print(f"Agent A R1→R2: {analysis['answer_changes']['agent_a']['r1_r2']:.2%}")
    print(f"Agent B R0→R1: {analysis['answer_changes']['agent_b']['r0_r1']:.2%}")
    print(f"Agent B R1→R2: {analysis['answer_changes']['agent_b']['r1_r2']:.2%}")

    print("\n--- Reflection Effectiveness ---")
    print(f"Reflection Effectiveness: {analysis['reflection_effectiveness']:.2%}")

    print("\n" + "=" * 80)

    # 打印样本示例
    print("\n📋 Sample Examples (First 3):")
    print("-" * 80)
    for idx in range(min(3, len(results))):
        r = results[idx]
        print(f"\nSample {idx + 1}:")
        print(f"  Question: {r['question'][:80]}...")
        print(f"  True Answer: {r['true_answer']}")
        print(f"  Final Answer: {r['final_answer']}")
        print(f"  Correct: {'✅' if r['correct'] else '❌'}")
        print(f"  Agent A: {r['debate_history']['round_0']['agent_a']['answer']} → "
              f"{r['debate_history']['round_1']['agent_a']['answer']} → "
              f"{r['debate_history']['round_2']['agent_a']['answer']}")
        print(f"  Agent B: {r['debate_history']['round_0']['agent_b']['answer']} → "
              f"{r['debate_history']['round_1']['agent_b']['answer']} → "
              f"{r['debate_history']['round_2']['agent_b']['answer']}")
    print("-" * 80)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate (MAD) Evaluation")

    parser.add_argument('--model_type', type=int, required=True,
                       choices=[1, 2],
                       help='Model type: 1=LLaMA 3.1, 2=Qwen 2.5')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to evaluation data file (JSON)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples (upper limit after random sampling)')
    parser.add_argument('--random_p', type=float, default=None,
                       help='Random sampling ratio (0-1), e.g., 0.1 for 10%% of dataset')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Custom model path (overrides default)')

    args = parser.parse_args()

    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
        model_name = os.path.basename(model_path)
    else:
        # 默认使用本地路径（如果shell脚本没有传递--model_path）
        if args.model_type == 1:
            model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
            model_name = "llama"
        elif args.model_type == 2:
            model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
            model_name = "qwen"
        else:
            raise ValueError("Invalid model_type")

    print("\n" + "=" * 80)
    print("Multi-Agent Debate (MAD) Evaluation")
    print("=" * 80)
    print(f"Model Type: {args.model_type} ({model_name})")
    print(f"Model Path: {model_path}")
    print(f"Data File: {args.data_file}")
    print(f"Output Dir: {args.output_dir}")
    if args.random_p:
        print(f"Random Sampling: {args.random_p:.1%} of dataset")
        print(f"Random Seed: {args.random_seed}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples} (upper limit)")
    print("=" * 80 + "\n")

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    print("✅ Model loaded and moved to device")

    # 加载tokenizer (禁用fast tokenizer以确保多线程安全)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False  # 禁用fast tokenizer,避免多线程CUDA错误
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded (slow tokenizer for thread safety)")

    # 加载数据
    data = load_data(
        args.data_file,
        max_samples=args.max_samples,
        random_p=args.random_p,
        random_seed=args.random_seed
    )

    # 初始化MAD引擎
    print("\nInitializing MAD Debate Engine...")
    # 前3个样本启用详细日志
    mad_engine = MADDebateEngine(model, tokenizer, device, verbose=True)
    print("✅ MAD Engine initialized")

    # Warm-up：第一次生成通常很慢，做一次预热
    print("\nWarming up model (first generation)...")
    try:
        _ = mad_engine.generate_response("Test prompt", max_new_tokens=10)
        print("✅ Model warm-up completed")
    except Exception as e:
        print(f"⚠️  Warm-up warning: {str(e)}")

    # 运行评估
    print("\nStarting evaluation...")
    results = []

    for idx, item in enumerate(tqdm(data, desc="Evaluating", mininterval=1.0)):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        true_answer = item.get('output', '')

        # 添加详细日志（仅前3个样本）
        if idx < 3:
            print(f"\n[Sample {idx+1}] Processing...")
            mad_engine.verbose = True
        else:
            mad_engine.verbose = False

        try:
            result = mad_engine.run_debate(instruction, input_text, true_answer)
            results.append(result)

            if idx < 3:
                print(f"[Sample {idx+1}] ✅ Completed")

        except Exception as e:
            print(f"\n⚠️  Error processing sample {idx}: {str(e)}")
            # 添加一个失败的结果 (保持结构一致性)
            results.append({
                'question': instruction,
                'context': input_text,
                'true_answer': true_answer,
                'final_answer': '1',
                'correct': False,
                'error': str(e),
                'debate_history': {
                    'round_0': {'agent_a': {}, 'agent_b': {}},
                    'round_1': {'agent_a': {}, 'agent_b': {}},
                    'round_2': {'agent_a': {}, 'agent_b': {}},
                    'final_decision': {'answer': '1', 'confidence': 0, 'reasoning': 'Error occurred'}
                },
                'debate_metrics': {}
            })

    # 保存结果
    print("\nSaving results...")
    save_results(results, args.output_dir)

    # 打印统计
    print_statistics(results)

    print("\n✅ Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
