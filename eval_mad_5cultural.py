#!/usr/bin/env python3
"""
5-Agent Cultural Multi-Agent Debate (MAD) 评估脚本

实现基于5个文化视角的多代理辩论系统:
- Asia (亚洲)
- Western (西方: 北美+欧洲)
- South America (南美)
- Oceania (大洋洲)
- Africa (非洲)

使用方法:
    python eval_mad_5cultural.py \
        --model_type 1 \
        --data_file /path/to/data.json \
        --output_dir /path/to/output \
        [--max_samples 100]
"""

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import random
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Prompt模板 - Round 0 (初始回答)
# ============================================================================

PROMPT_ASIA_R0 = """You are a cultural expert representing Asian perspectives (East Asia, Southeast Asia, South Asia). Your task is to answer the following question based on Asian cultural values and understanding.

**Question:**
{instruction}

**Context:**
{input}

**Your Cultural Perspective:**
As an Asian cultural expert, consider values such as:
- Collectivism and community harmony
- Respect for tradition and elders
- Balance and moderation
- Indirect communication and context sensitivity

**Instructions:**
1. Analyze the question from an Asian cultural viewpoint
2. Consider how Asian cultures would approach this situation
3. Choose the answer (1-4) that best reflects Asian cultural understanding
4. Explain your reasoning clearly

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your reasoning in 2-3 sentences from Asian cultural perspective]
Confidence: [0-100]

Your answer:"""

PROMPT_WESTERN_R0 = """You are a cultural expert representing Western perspectives (North America and Europe). Your task is to answer the following question based on Western cultural values and understanding.

**Question:**
{instruction}

**Context:**
{input}

**Your Cultural Perspective:**
As a Western cultural expert, consider values such as:
- Individualism and personal freedom
- Direct communication and clarity
- Rational analysis and logic
- Innovation and progress

**Instructions:**
1. Analyze the question from a Western cultural viewpoint
2. Consider how Western cultures would approach this situation
3. Choose the answer (1-4) that best reflects Western cultural understanding
4. Explain your reasoning clearly

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your reasoning in 2-3 sentences from Western cultural perspective]
Confidence: [0-100]

Your answer:"""

PROMPT_SOUTH_AMERICA_R0 = """You are a cultural expert representing South American perspectives (Latin America). Your task is to answer the following question based on South American cultural values and understanding.

**Question:**
{instruction}

**Context:**
{input}

**Your Cultural Perspective:**
As a South American cultural expert, consider values such as:
- Strong community and family bonds
- Warmth and personal relationships
- Celebration and expressiveness
- Resilience and adaptability

**Instructions:**
1. Analyze the question from a South American cultural viewpoint
2. Consider how South American cultures would approach this situation
3. Choose the answer (1-4) that best reflects South American cultural understanding
4. Explain your reasoning clearly

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your reasoning in 2-3 sentences from South American cultural perspective]
Confidence: [0-100]

Your answer:"""

PROMPT_OCEANIA_R0 = """You are a cultural expert representing Oceanian perspectives (Australia, New Zealand, Pacific Islands). Your task is to answer the following question based on Oceanian cultural values and understanding.

**Question:**
{instruction}

**Context:**
{input}

**Your Cultural Perspective:**
As an Oceanian cultural expert, consider values such as:
- Connection to nature and land
- Multicultural harmony and diversity
- Egalitarianism and fairness
- Laid-back and practical approach

**Instructions:**
1. Analyze the question from an Oceanian cultural viewpoint
2. Consider how Oceanian cultures would approach this situation
3. Choose the answer (1-4) that best reflects Oceanian cultural understanding
4. Explain your reasoning clearly

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your reasoning in 2-3 sentences from Oceanian cultural perspective]
Confidence: [0-100]

Your answer:"""

PROMPT_AFRICA_R0 = """You are a cultural expert representing African perspectives. Your task is to answer the following question based on African cultural values and understanding.

**Question:**
{instruction}

**Context:**
{input}

**Your Cultural Perspective:**
As an African cultural expert, consider values such as:
- Ubuntu philosophy (I am because we are)
- Strong community and extended family
- Oral tradition and storytelling
- Respect for ancestors and wisdom

**Instructions:**
1. Analyze the question from an African cultural viewpoint
2. Consider how African cultures would approach this situation
3. Choose the answer (1-4) that best reflects African cultural understanding
4. Explain your reasoning clearly

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your reasoning in 2-3 sentences from African cultural perspective]
Confidence: [0-100]

Your answer:"""

# ============================================================================
# Prompt模板 - Round 1 (第一轮讨论)
# ============================================================================

PROMPT_ROUND1_TEMPLATE = """You are continuing the multicultural discussion. You have seen perspectives from other cultural experts.

**Question:**
{instruction}

**Context:**
{input}

**Your Previous Answer (Round 0):**
- Answer: {your_answer}
- Reasoning: {your_reasoning}
- Confidence: {your_confidence}

**Other Cultural Perspectives (Round 0):**
{other_perspectives}

**Instructions:**
1. Consider the diverse cultural perspectives shared
2. Reflect on whether other viewpoints reveal aspects you missed
3. Update your answer if you find stronger cultural evidence
4. Maintain your cultural perspective while being open to learning

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your updated reasoning considering multicultural input]
Confidence: [0-100]
Changed: [Yes/No]

Your response:"""

# ============================================================================
# Prompt模板 - Round 2 (第二轮讨论)
# ============================================================================

PROMPT_ROUND2_TEMPLATE = """This is the final round of multicultural discussion. Review all perspectives and provide your final answer.

**Question:**
{instruction}

**Context:**
{input}

**Discussion History:**
Round 0 - Your initial answer: {your_r0_answer}
Round 1 - Your updated answer: {your_r1_answer} (Changed: {changed_r1})

**Other Cultural Perspectives (Round 1):**
{other_perspectives_r1}

**Instructions:**
1. Review the full multicultural discussion
2. Make your final decision based on the strongest cultural evidence
3. Your answer should reflect the most culturally accurate response

**Output Format:**
Answer: [1/2/3/4]
Reasoning: [Your final reasoning]
Confidence: [0-100]
Changed: [Yes/No]

Your final answer:"""

# ============================================================================
# Prompt模板 - Summarizer (总结者)
# ============================================================================

PROMPT_SUMMARIZER = """You are a multicultural summarizer. Five cultural experts have discussed a question from different perspectives. Your task is to determine the most culturally accurate answer by synthesizing their viewpoints.

**Question:**
{instruction}

**Context:**
{input}

**Final Answers from Cultural Experts (Round 2):**

Asia Expert:
- Answer: {asia_answer}
- Reasoning: {asia_reasoning}
- Confidence: {asia_confidence}

Western Expert:
- Answer: {western_answer}
- Reasoning: {western_reasoning}
- Confidence: {western_confidence}

South America Expert:
- Answer: {south_america_answer}
- Reasoning: {south_america_reasoning}
- Confidence: {south_america_confidence}

Oceania Expert:
- Answer: {oceania_answer}
- Reasoning: {oceania_reasoning}
- Confidence: {oceania_confidence}

Africa Expert:
- Answer: {africa_answer}
- Reasoning: {africa_reasoning}
- Confidence: {africa_confidence}

**Your Task:**
1. Analyze the diverse cultural perspectives
2. Identify common ground and key differences
3. Determine which answer has the strongest support across cultures
4. Make a final decision that best represents multicultural understanding

**Decision Guidelines:**
- Consider the strength of reasoning from each perspective
- Look for convergence across multiple cultures
- Evaluate confidence levels
- Choose the most culturally accurate answer

**Output Format:**
Final_Answer: [1/2/3/4]
Reasoning: [2-3 sentences explaining your synthesis of multicultural perspectives]
Confidence: [0-100]

Your decision:"""

# ============================================================================
# 辅助函数 (从eval_mad.py复制)
# ============================================================================

def parse_model_output(output_text: str, expected_fields: List[str]) -> Dict[str, str]:
    """解析模型输出，提取结构化字段"""
    result = {}
    for field in expected_fields:
        pattern = rf"{field}:\s*(.+?)(?=\n[A-Z][a-z_]+:|$)"
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            result[field] = content
        else:
            pattern_loose = rf"{field}[:\s]+(.+?)(?=\n\n|$)"
            match_loose = re.search(pattern_loose, output_text, re.DOTALL | re.IGNORECASE)
            if match_loose:
                result[field] = match_loose.group(1).strip()
            else:
                result[field] = ""
    return result


def extract_answer(text: str) -> str:
    """从文本中提取答案（1-4）"""
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
    return '1'


def extract_confidence(text: str) -> int:
    """从文本中提取置信度（0-100）"""
    pattern = r'Confidence:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        conf = int(match.group(1))
        return max(0, min(100, conf))
    else:
        return 50


def extract_yes_no(text: str) -> bool:
    """从文本中提取Yes/No答案"""
    text_lower = text.lower()
    if 'yes' in text_lower:
        return True
    elif 'no' in text_lower:
        return False
    else:
        return False


# ============================================================================
# 5-Agent Cultural MAD Engine
# ============================================================================

class CulturalMADEngine:
    """5-Agent Cultural Multi-Agent Debate Engine"""

    CULTURES = ['asia', 'western', 'south_america', 'oceania', 'africa']

    def __init__(self, model, tokenizer, device: str = 'cuda', verbose: bool = False):
        """初始化5-Agent Cultural MAD引擎"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose
        self.model.eval()
        self._generation_lock = threading.Lock()

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False
    ) -> str:
        """通用生成函数"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

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
                        use_cache=True
                    )
                except Exception as e:
                    print(f"\n⚠️  Generation error: {str(e)}")
                    raise

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated_text

    def generate_initial_answer(
        self,
        instruction: str,
        input_text: str,
        culture: str
    ) -> Dict:
        """生成初始回答 (Round 0)"""
        # 选择对应文化的prompt
        prompt_map = {
            'asia': PROMPT_ASIA_R0,
            'western': PROMPT_WESTERN_R0,
            'south_america': PROMPT_SOUTH_AMERICA_R0,
            'oceania': PROMPT_OCEANIA_R0,
            'africa': PROMPT_AFRICA_R0
        }

        prompt_template = prompt_map[culture]
        prompt = prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else "(No additional context)"
        )

        output_text = self.generate_response(prompt, max_new_tokens=256)

        parsed = parse_model_output(
            output_text,
            ['Answer', 'Reasoning', 'Confidence']
        )

        result = {
            'culture': culture,
            'answer': extract_answer(parsed.get('Answer', output_text)),
            'reasoning': parsed.get('Reasoning', output_text[:200]),
            'confidence': extract_confidence(parsed.get('Confidence', '50')),
            'raw_output': output_text
        }

        return result

    def format_other_perspectives(
        self,
        all_answers: Dict[str, Dict],
        exclude_culture: str
    ) -> str:
        """格式化其他文化的观点"""
        perspectives = []
        culture_names = {
            'asia': 'Asia',
            'western': 'Western',
            'south_america': 'South America',
            'oceania': 'Oceania',
            'africa': 'Africa'
        }

        for culture in self.CULTURES:
            if culture != exclude_culture:
                answer_dict = all_answers[culture]
                perspectives.append(
                    f"{culture_names[culture]} Expert:\n"
                    f"- Answer: {answer_dict['answer']}\n"
                    f"- Reasoning: {answer_dict['reasoning']}\n"
                    f"- Confidence: {answer_dict['confidence']}"
                )

        return "\n\n".join(perspectives)

    def generate_round1_answer(
        self,
        instruction: str,
        input_text: str,
        culture: str,
        r0_self: Dict,
        r0_all: Dict[str, Dict]
    ) -> Dict:
        """生成Round 1回答"""
        other_perspectives = self.format_other_perspectives(r0_all, culture)

        prompt = PROMPT_ROUND1_TEMPLATE.format(
            instruction=instruction,
            input=input_text if input_text else "(No additional context)",
            your_answer=r0_self['answer'],
            your_reasoning=r0_self['reasoning'],
            your_confidence=r0_self['confidence'],
            other_perspectives=other_perspectives
        )

        output_text = self.generate_response(prompt, max_new_tokens=256)

        parsed = parse_model_output(
            output_text,
            ['Answer', 'Reasoning', 'Confidence', 'Changed']
        )

        result = {
            'culture': culture,
            'answer': extract_answer(parsed.get('Answer', output_text)),
            'reasoning': parsed.get('Reasoning', output_text[:200]),
            'confidence': extract_confidence(parsed.get('Confidence', '50')),
            'changed': extract_yes_no(parsed.get('Changed', 'No')),
            'raw_output': output_text
        }

        return result

    def generate_round2_answer(
        self,
        instruction: str,
        input_text: str,
        culture: str,
        r0_self: Dict,
        r1_self: Dict,
        r1_all: Dict[str, Dict]
    ) -> Dict:
        """生成Round 2回答"""
        other_perspectives = self.format_other_perspectives(r1_all, culture)

        prompt = PROMPT_ROUND2_TEMPLATE.format(
            instruction=instruction,
            input=input_text if input_text else "(No additional context)",
            your_r0_answer=r0_self['answer'],
            your_r1_answer=r1_self['answer'],
            changed_r1='Yes' if r1_self.get('changed', False) else 'No',
            other_perspectives_r1=other_perspectives
        )

        output_text = self.generate_response(prompt, max_new_tokens=256)

        parsed = parse_model_output(
            output_text,
            ['Answer', 'Reasoning', 'Confidence', 'Changed']
        )

        result = {
            'culture': culture,
            'answer': extract_answer(parsed.get('Answer', output_text)),
            'reasoning': parsed.get('Reasoning', output_text[:200]),
            'confidence': extract_confidence(parsed.get('Confidence', '50')),
            'changed': extract_yes_no(parsed.get('Changed', 'No')),
            'raw_output': output_text
        }

        return result

    def generate_summarizer_decision(
        self,
        instruction: str,
        input_text: str,
        r2_all: Dict[str, Dict]
    ) -> Dict:
        """生成Summarizer决策"""
        prompt = PROMPT_SUMMARIZER.format(
            instruction=instruction,
            input=input_text if input_text else "(No additional context)",
            asia_answer=r2_all['asia']['answer'],
            asia_reasoning=r2_all['asia']['reasoning'],
            asia_confidence=r2_all['asia']['confidence'],
            western_answer=r2_all['western']['answer'],
            western_reasoning=r2_all['western']['reasoning'],
            western_confidence=r2_all['western']['confidence'],
            south_america_answer=r2_all['south_america']['answer'],
            south_america_reasoning=r2_all['south_america']['reasoning'],
            south_america_confidence=r2_all['south_america']['confidence'],
            oceania_answer=r2_all['oceania']['answer'],
            oceania_reasoning=r2_all['oceania']['reasoning'],
            oceania_confidence=r2_all['oceania']['confidence'],
            africa_answer=r2_all['africa']['answer'],
            africa_reasoning=r2_all['africa']['reasoning'],
            africa_confidence=r2_all['africa']['confidence']
        )

        output_text = self.generate_response(prompt, max_new_tokens=256)

        parsed = parse_model_output(
            output_text,
            ['Final_Answer', 'Reasoning', 'Confidence']
        )

        result = {
            'answer': extract_answer(parsed.get('Final_Answer', output_text)),
            'reasoning': parsed.get('Reasoning', output_text[:200]),
            'confidence': extract_confidence(parsed.get('Confidence', '50')),
            'raw_output': output_text
        }

        return result

    def check_consensus(self, r2_all: Dict[str, Dict]) -> Tuple[bool, Optional[str]]:
        """检查是否达成多数共识 (≥3/5)"""
        answers = [r2_all[culture]['answer'] for culture in self.CULTURES]
        vote_counts = Counter(answers)
        most_common = vote_counts.most_common(1)[0]

        if most_common[1] >= 3:  # 多数共识
            return True, most_common[0]
        else:
            return False, None

    def run_debate(
        self,
        instruction: str,
        input_text: str,
        true_answer: str
    ) -> Dict:
        """运行完整的5-Agent Cultural MAD流程"""
        # Round 0: 初始回答 (并行)
        if self.verbose:
            print("  [R0] All 5 cultural agents generating in parallel...")

        r0_all = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                culture: executor.submit(
                    self.generate_initial_answer,
                    instruction, input_text, culture
                )
                for culture in self.CULTURES
            }
            for culture, future in futures.items():
                r0_all[culture] = future.result()

        # Round 1: 第一轮讨论 (并行)
        if self.verbose:
            print("  [R1] All 5 cultural agents generating in parallel...")

        r1_all = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                culture: executor.submit(
                    self.generate_round1_answer,
                    instruction, input_text, culture,
                    r0_all[culture], r0_all
                )
                for culture in self.CULTURES
            }
            for culture, future in futures.items():
                r1_all[culture] = future.result()

        # Round 2: 第二轮讨论 (并行)
        if self.verbose:
            print("  [R2] All 5 cultural agents generating in parallel...")

        r2_all = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                culture: executor.submit(
                    self.generate_round2_answer,
                    instruction, input_text, culture,
                    r0_all[culture], r1_all[culture], r1_all
                )
                for culture in self.CULTURES
            }
            for culture, future in futures.items():
                r2_all[culture] = future.result()

        # 检查一致性
        has_consensus, consensus_answer = self.check_consensus(r2_all)

        if has_consensus:
            if self.verbose:
                print(f"  [Consensus] Majority agreement on answer {consensus_answer}")
            final_answer = consensus_answer
            final_reasoning = f"Majority consensus (≥3/5 agents) on answer {consensus_answer}"
            decision_method = 'consensus'
        else:
            if self.verbose:
                print("  [Summarizer] No consensus, using summarizer...")
            summarizer_result = self.generate_summarizer_decision(
                instruction, input_text, r2_all
            )
            final_answer = summarizer_result['answer']
            final_reasoning = summarizer_result['reasoning']
            decision_method = 'summarizer'

        # 构建结果
        result = {
            'question': instruction,
            'context': input_text,
            'true_answer': true_answer,
            'debate_history': {
                'round_0': r0_all,
                'round_1': r1_all,
                'round_2': r2_all
            },
            'final_answer': final_answer,
            'final_reasoning': final_reasoning,
            'decision_method': decision_method,
            'has_consensus': has_consensus,
            'correct': final_answer == true_answer
        }

        return result


# ============================================================================
# 数据加载和结果保存
# ============================================================================

def load_data(
    data_file: str,
    max_samples: Optional[int] = None,
    random_p: Optional[float] = None,
    random_seed: int = 42
) -> List[Dict]:
    """加载和采样数据"""
    print(f"Loading data from: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_size = len(data)
    print(f"Loaded {original_size} samples")

    # 采样逻辑
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
        print(f"Using max_samples mode: taking first {max_samples} samples (random_p ignored)")
    else:
        if random_p is not None:
            if random_p == 1.0:
                print(f"Using all data (random_p=1.0): {original_size} samples")
            else:
                random.seed(random_seed)
                sample_size = int(original_size * random_p)
                data = random.sample(data, sample_size)
                print(f"Random sampling: {sample_size} samples ({random_p:.1%} of dataset)")

    print(f"Final dataset size: {len(data)} samples")
    return data


def save_results(results: List[Dict], output_dir: str):
    """保存评估结果"""
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

    # 保存生成的答案
    generated_answers = generate_answers_report(results)
    answers_path = os.path.join(output_dir, 'generated_answers.json')
    with open(answers_path, 'w', encoding='utf-8') as f:
        json.dump(generated_answers, f, indent=2, ensure_ascii=False)
    print(f"Generated answers saved to: {answers_path}")

    # 保存评估指标
    eval_metrics = compute_eval_metrics(results)
    metrics_path = os.path.join(output_dir, 'eval_results.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    print(f"Evaluation metrics saved to: {metrics_path}")


def generate_answers_report(results: List[Dict]) -> List[Dict]:
    """生成答案报告"""
    report = []
    for r in results:
        if 'error' in r:
            continue

        try:
            r2 = r['debate_history']['round_2']
            report.append({
                'question': r['question'],
                'context': r.get('context', ''),
                'cultural_answers': {
                    culture: {
                        'round_0': r['debate_history']['round_0'][culture]['answer'],
                        'round_1': r['debate_history']['round_1'][culture]['answer'],
                        'round_2': r2[culture]['answer']
                    }
                    for culture in CulturalMADEngine.CULTURES
                },
                'final_answer': r.get('final_answer', 'N/A'),
                'decision_method': r.get('decision_method', 'N/A'),
                'has_consensus': r.get('has_consensus', False),
                'true_answer': r.get('true_answer', 'N/A'),
                'is_correct': r.get('correct', False)
            })
        except (KeyError, TypeError):
            continue

    return report


def compute_eval_metrics(results: List[Dict]) -> Dict:
    """计算评估指标"""
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

    correct = sum(1 for r in valid_results if r.get('correct', False))
    accuracy = correct / total
    precision = accuracy
    recall = accuracy

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


def compute_summary_statistics(results: List[Dict]) -> Dict:
    """计算汇总统计"""
    valid_results = [r for r in results if 'error' not in r]
    total = len(results)
    valid_total = len(valid_results)

    if valid_total == 0:
        return {
            'accuracy': 0.0,
            'total_samples': total,
            'valid_samples': 0,
            'failed_samples': total,
            'correct_predictions': 0,
            'consensus_rate': 0.0,
            'summarizer_usage_rate': 0.0
        }

    # 最终准确率
    correct = sum(1 for r in valid_results if r.get('correct', False))
    accuracy = correct / valid_total

    # 每个文化Agent的Round 0准确率
    culture_r0_accuracy = {}
    for culture in CulturalMADEngine.CULTURES:
        culture_correct = 0
        for r in valid_results:
            try:
                r0_answer = r['debate_history']['round_0'][culture]['answer']
                if r0_answer == r['true_answer']:
                    culture_correct += 1
            except (KeyError, TypeError):
                pass
        culture_r0_accuracy[f'{culture}_r0_accuracy'] = culture_correct / valid_total if valid_total > 0 else 0

    # 共识率
    consensus_count = sum(1 for r in valid_results if r.get('has_consensus', False))
    consensus_rate = consensus_count / valid_total if valid_total > 0 else 0

    # Summarizer使用率
    summarizer_count = sum(1 for r in valid_results if r.get('decision_method') == 'summarizer')
    summarizer_usage_rate = summarizer_count / valid_total if valid_total > 0 else 0

    # MAD增益 (相对于最好的单个Agent)
    best_agent_accuracy = max(culture_r0_accuracy.values()) if culture_r0_accuracy else 0
    mad_gain = accuracy - best_agent_accuracy

    result = {
        'accuracy': accuracy,
        'total_samples': total,
        'valid_samples': valid_total,
        'failed_samples': total - valid_total,
        'correct_predictions': correct,
        'consensus_rate': consensus_rate,
        'summarizer_usage_rate': summarizer_usage_rate,
        'mad_gain': mad_gain,
        **culture_r0_accuracy
    }

    return result


def print_statistics(results: List[Dict]):
    """打印统计信息"""
    summary = compute_summary_statistics(results)

    print("\n" + "="*80)
    print("5-Agent Cultural MAD Evaluation Report")
    print("="*80)

    print("\n--- Basic Metrics ---")
    print(f"Final Accuracy: {summary['accuracy']:.2%}")
    print(f"Correct Predictions: {summary['correct_predictions']}/{summary['valid_samples']}")
    print(f"Consensus Rate: {summary['consensus_rate']:.2%}")
    print(f"Summarizer Usage: {summary['summarizer_usage_rate']:.2%}")

    print("\n--- Cultural Agents (Round 0) ---")
    for culture in CulturalMADEngine.CULTURES:
        key = f'{culture}_r0_accuracy'
        if key in summary:
            culture_name = culture.replace('_', ' ').title()
            print(f"{culture_name}: {summary[key]:.2%}")

    print("\n--- MAD Performance ---")
    print(f"Best Agent (R0): {max([summary[f'{c}_r0_accuracy'] for c in CulturalMADEngine.CULTURES]):.2%}")
    print(f"MAD Final: {summary['accuracy']:.2%}")
    print(f"MAD Gain: {summary['mad_gain']:+.2%}")

    print("\n" + "="*80)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='5-Agent Cultural MAD Evaluation')
    parser.add_argument('--model_type', type=int, required=True,
                        help='Model type: 1=LLaMA, 2=Qwen')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model (optional, will use default if not provided)')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to evaluation data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--random_p', type=float, default=None,
                        help='Random sampling ratio (0-1), only used if max_samples=0')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for sampling')

    args = parser.parse_args()

    # 设置默认模型路径
    if args.model_path is None:
        if args.model_type == 1:
            args.model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        elif args.model_type == 2:
            args.model_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        else:
            print(f"❌ Invalid model_type: {args.model_type}")
            sys.exit(1)

    print("="*80)
    print("5-Agent Cultural Multi-Agent Debate (MAD) Evaluation")
    print("="*80)
    print(f"Model Type: {args.model_type} ({'LLaMA' if args.model_type == 1 else 'Qwen'})")
    print(f"Model Path: {args.model_path}")
    print(f"Data File: {args.data_file}")
    print(f"Output Dir: {args.output_dir}")
    print("="*80)

    # 环境变量设置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # 加载模型
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
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
        args.model_path,
        trust_remote_code=True,
        use_fast=False
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

    # 初始化Cultural MAD引擎
    print("\nInitializing 5-Agent Cultural MAD Engine...")
    mad_engine = CulturalMADEngine(model, tokenizer, device, verbose=True)
    print("✅ Cultural MAD Engine initialized")

    # Warm-up
    print("\nWarming up model...")
    try:
        _ = mad_engine.generate_response("Test prompt", max_new_tokens=10)
        print("✅ Model warm-up completed")
    except Exception as e:
        print(f"⚠️  Warm-up warning: {str(e)}")

    # 运行评估
    print("\nStarting 5-Agent Cultural MAD evaluation...")
    print(f"Evaluating {len(data)} samples with 5 cultural perspectives...\n")

    results = []

    for idx, item in enumerate(tqdm(data, desc="Evaluating", mininterval=1.0)):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        true_answer = item.get('output', '')

        # 详细日志(仅前3个样本)
        if idx < 3:
            print(f"\n[Sample {idx+1}] Processing with 5 cultural agents...")
            mad_engine.verbose = True
        else:
            mad_engine.verbose = False

        try:
            result = mad_engine.run_debate(instruction, input_text, true_answer)
            results.append(result)

            if idx < 3:
                print(f"[Sample {idx+1}] ✅ Completed")
                print(f"  Decision: {result['decision_method']}")
                print(f"  Final Answer: {result['final_answer']} (Correct: {result['correct']})")

        except Exception as e:
            print(f"\n⚠️  Error processing sample {idx}: {str(e)}")
            results.append({
                'question': instruction,
                'context': input_text,
                'true_answer': true_answer,
                'final_answer': '1',
                'correct': False,
                'error': str(e),
                'debate_history': {},
                'decision_method': 'error'
            })

    # 保存结果
    print("\nSaving results...")
    save_results(results, args.output_dir)

    # 打印统计
    print_statistics(results)

    print(f"\n✅ Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
