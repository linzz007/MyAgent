"""运行官方评估器评估预测结果

用法：
python run_evaluation.py --dataset wtq --pred_file wtq_predictions.tsv --gold_file dataset/test_samples_wtq.tsv
python run_evaluation.py --dataset tat --pred_file tatqa_predictions.json --gold_file dataset/test_samples_tatqa.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dataset/TAT-QA'))


def evaluate_wtq_simple(pred_file: str, gold_file: str):
    """简单的WTQ评估（用于测试数据，不使用tagged数据）"""
    print(f"\n{'='*80}")
    print("WTQ 评估（简化版）")
    print(f"{'='*80}")
    
    # 读取预测结果
    predictions = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                ex_id = parts[0]
                pred_answer = parts[1]  # 只取第一个答案
                predictions[ex_id] = pred_answer
    
    # 读取标准答案
    gold_answers = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
        if len(lines) > 1:
            header = lines[0].split("\t")
            for line in lines[1:]:
                parts = line.split("\t")
                row = {k: v for k, v in zip(header, parts)}
                ex_id = row.get('id', '')
                gold_answer = row.get('answer', '') or row.get('targetValue', '')
                if ex_id and gold_answer:
                    gold_answers[ex_id] = gold_answer
    
    # 评估
    correct = 0
    total = 0
    
    for ex_id, gold_answer in gold_answers.items():
        if ex_id not in predictions:
            print(f"⚠️  预测结果中缺少样本: {ex_id}")
            continue
        
        pred_answer = predictions[ex_id]
        total += 1
        
        # 简单的字符串匹配（标准化后比较）
        def normalize(s):
            s = str(s).lower().strip()
            # 移除标点符号
            import string
            s = ''.join(c for c in s if c not in string.punctuation)
            return s
        
        pred_norm = normalize(pred_answer)
        gold_norm = normalize(gold_answer)
        
        # 检查是否包含标准答案（因为预测答案可能包含额外文本）
        is_correct = gold_norm in pred_norm or pred_norm in gold_norm or pred_norm == gold_norm
        
        if is_correct:
            correct += 1
            print(f"✓ {ex_id}: 预测='{pred_answer[:50]}...' vs 标准='{gold_answer}'")
        else:
            print(f"✗ {ex_id}: 预测='{pred_answer[:50]}...' vs 标准='{gold_answer}'")
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    print(f"\n准确率: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


def evaluate_tatqa(pred_file: str, gold_file: str):
    """使用TAT-QA官方评估器"""
    print(f"\n{'='*80}")
    print("TAT-QA 评估（使用官方评估器）")
    print(f"{'='*80}")
    
    try:
        from tatqa_eval import evaluate_json
        
        # 读取预测结果
        with open(pred_file, 'r', encoding='utf-8') as f:
            predicted_answers = json.load(f)
        
        # 读取标准答案
        with open(gold_file, 'r', encoding='utf-8') as f:
            golden_answers = json.load(f)
        
        # 运行评估
        evaluate_json(golden_answers, predicted_answers)
        
    except ImportError as e:
        print(f"❌ 无法导入TAT-QA评估器: {e}")
        print("尝试简化评估...")
        evaluate_tatqa_simple(pred_file, gold_file)
    except Exception as e:
        print(f"❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()


def evaluate_tatqa_simple(pred_file: str, gold_file: str):
    """简化的TAT-QA评估"""
    print(f"\n{'='*80}")
    print("TAT-QA 评估（简化版）")
    print(f"{'='*80}")
    
    # 读取预测结果
    with open(pred_file, 'r', encoding='utf-8') as f:
        predicted_answers = json.load(f)
    
    # 读取标准答案
    with open(gold_file, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    
    # 提取标准答案
    gold_answers = {}
    for table_item in golden_data:
        for q_item in table_item.get('questions', []):
            uid = q_item.get('uid', '')
            answer = q_item.get('answer', [])
            if uid and answer:
                gold_answers[uid] = answer[0] if isinstance(answer, list) else answer
    
    # 评估
    correct = 0
    total = 0
    
    for uid, gold_answer in gold_answers.items():
        if uid not in predicted_answers:
            print(f"⚠️  预测结果中缺少问题: {uid}")
            continue
        
        pred_data = predicted_answers[uid]
        if isinstance(pred_data, list) and len(pred_data) > 0:
            pred_answer = pred_data[0] if isinstance(pred_data[0], list) else pred_data[0]
        else:
            pred_answer = str(pred_data)
        
        total += 1
        
        # 简单的字符串匹配（标准化后比较）
        def normalize(s):
            s = str(s).lower().strip()
            # 移除标点符号和空格
            import string
            s = ''.join(c for c in s if c not in string.punctuation and not c.isspace())
            return s
        
        pred_norm = normalize(pred_answer)
        gold_norm = normalize(gold_answer)
        
        # 检查是否包含标准答案
        is_correct = gold_norm in pred_norm or pred_norm in gold_norm or pred_norm == gold_norm
        
        if is_correct:
            correct += 1
            print(f"✓ {uid}: 预测='{pred_answer[:50]}...' vs 标准='{gold_answer}'")
        else:
            print(f"✗ {uid}: 预测='{pred_answer[:50]}...' vs 标准='{gold_answer}'")
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    print(f"\n准确率: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="运行官方评估器评估预测结果")
    parser.add_argument("--dataset", type=str, choices=["wtq", "tat"], required=True, help="数据集类型")
    parser.add_argument("--pred_file", type=str, required=True, help="预测结果文件")
    parser.add_argument("--gold_file", type=str, required=True, help="标准答案文件")
    args = parser.parse_args()
    
    if args.dataset == "wtq":
        evaluate_wtq_simple(args.pred_file, args.gold_file)
    elif args.dataset == "tat":
        evaluate_tatqa(args.pred_file, args.gold_file)


if __name__ == "__main__":
    main()
