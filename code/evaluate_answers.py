"""答案评估脚本

支持三种数据集的答案评估：
1. WikiTableQuestions: 使用denotation matching（值匹配）
2. HiTab: 数值精确匹配
3. TAT-QA: 使用TAT-QA官方的EM和F1评估

用法：
python evaluate_answers.py --pred_file predictions.jsonl --dataset wtq
"""

import argparse
import json
import re
from typing import Any, Dict, List, Union


def normalize_value(value: Any) -> str:
    """标准化值：去除空格、标点、统一大小写"""
    if value is None:
        return ""
    s = str(value).strip().lower()
    # 移除常见的标点符号
    s = re.sub(r'[,\$]', '', s)
    return s


def check_denotation_wtq(target_values: List[str], predicted_values: List[str]) -> bool:
    """WikiTableQuestions的denotation matching
    
    检查预测值是否与目标值在语义上等价（例如"1000"和"1,000"应该匹配）
    """
    # 标准化所有值
    target_normalized = [normalize_value(v) for v in target_values]
    predicted_normalized = [normalize_value(v) for v in predicted_values]
    
    # 检查集合是否相等
    return set(target_normalized) == set(predicted_normalized)


def check_exact_match_hitab(target_value: Union[float, int], predicted_value: Union[float, int], tolerance: float = 1e-6) -> bool:
    """HiTab的精确数值匹配"""
    try:
        target = float(target_value)
        predicted = float(predicted_value)
        return abs(target - predicted) < tolerance
    except (ValueError, TypeError):
        return False


def check_exact_match_tatqa(target: Union[str, List[str]], predicted: Union[str, List[str]]) -> bool:
    """TAT-QA的精确匹配（简化版）"""
    # 转换为列表
    if isinstance(target, str):
        target = [target]
    if isinstance(predicted, str):
        predicted = [predicted]
    
    # 标准化并比较
    target_normalized = [normalize_value(v) for v in target]
    predicted_normalized = [normalize_value(v) for v in predicted]
    
    return set(target_normalized) == set(predicted_normalized)


def evaluate_wtq(pred_file: str) -> Dict[str, float]:
    """评估WikiTableQuestions预测结果"""
    correct = 0
    total = 0
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            target_value = item.get('answer') or item.get('targetValue', '')
            predicted_value = item.get('final_answer', '') or item.get('final_value', '')
            
            if not target_value or not predicted_value:
                total += 1
                continue
            
            # 转换为列表
            if isinstance(target_value, str):
                target_values = [target_value]
            else:
                target_values = target_value if isinstance(target_value, list) else [str(target_value)]
            
            if isinstance(predicted_value, str):
                predicted_values = [predicted_value]
            else:
                predicted_values = predicted_value if isinstance(predicted_value, list) else [str(predicted_value)]
            
            if check_denotation_wtq(target_values, predicted_values):
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def evaluate_hitab(pred_file: str) -> Dict[str, float]:
    """评估HiTab预测结果"""
    correct = 0
    total = 0
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            target_answer = item.get('answer', [])
            predicted_value = item.get('final_answer', '') or item.get('final_value', '')
            
            if not target_answer or not predicted_value:
                total += 1
                continue
            
            # HiTab的答案通常是数值列表
            if isinstance(target_answer, list) and len(target_answer) > 0:
                target_value = target_answer[0]
            else:
                target_value = target_answer
            
            # 尝试提取数值
            try:
                if isinstance(predicted_value, (int, float)):
                    pred_num = float(predicted_value)
                else:
                    # 从字符串中提取数值
                    pred_str = str(predicted_value)
                    numbers = re.findall(r'-?\d+\.?\d*', pred_str)
                    if numbers:
                        pred_num = float(numbers[0])
                    else:
                        total += 1
                        continue
                
                if check_exact_match_hitab(target_value, pred_num):
                    correct += 1
            except (ValueError, TypeError):
                pass
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def evaluate_tatqa(pred_file: str) -> Dict[str, float]:
    """评估TAT-QA预测结果（简化版，完整版需要使用官方评估器）"""
    correct = 0
    total = 0
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            target_answer = item.get('answer', [])
            predicted_value = item.get('final_answer', '') or item.get('final_value', '')
            
            if not target_answer or not predicted_value:
                total += 1
                continue
            
            if check_exact_match_tatqa(target_answer, predicted_value):
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def main():
    parser = argparse.ArgumentParser(description="评估答案正确性")
    parser.add_argument("--pred_file", type=str, required=True, help="预测结果文件（JSONL格式）")
    parser.add_argument("--dataset", type=str, choices=["wtq", "hitab", "tatqa"], required=True, help="数据集类型")
    args = parser.parse_args()
    
    print(f"评估数据集: {args.dataset}")
    print(f"预测文件: {args.pred_file}")
    print("=" * 80)
    
    if args.dataset == "wtq":
        results = evaluate_wtq(args.pred_file)
    elif args.dataset == "hitab":
        results = evaluate_hitab(args.pred_file)
    elif args.dataset == "tatqa":
        results = evaluate_tatqa(args.pred_file)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    print(f"\n评估结果:")
    print(f"  正确数: {results['correct']}")
    print(f"  总数: {results['total']}")
    print(f"  准确率: {results['accuracy']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
