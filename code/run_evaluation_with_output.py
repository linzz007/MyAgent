"""运行官方评估器并保存评估结果到文件

用法：
python run_evaluation_with_output.py --dataset tat --pred_file tatqa_predictions_for_eval.json --gold_file ../dataset/test_samples_tatqa.json --output_file evaluation_results.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any
from io import StringIO
from contextlib import redirect_stdout

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dataset/TAT-QA'))


def evaluate_tatqa_with_output(pred_file: str, gold_file: str, output_file: str = None):
    """使用TAT-QA官方评估器，并保存详细结果"""
    print(f"\n{'='*80}")
    print("TAT-QA 评估（使用官方评估器）")
    print(f"{'='*80}")
    
    try:
        from tatqa_eval import evaluate_json
        from tatqa_metric import TaTQAEmAndF1
        
        # 读取预测结果
        with open(pred_file, 'r', encoding='utf-8') as f:
            predicted_answers = json.load(f)
        
        # 读取标准答案
        with open(gold_file, 'r', encoding='utf-8') as f:
            golden_answers = json.load(f)
        
        # 创建评估器并逐个评估
        em_and_f1 = TaTQAEmAndF1()
        detailed_results = []
        
        for qas in golden_answers:
            for qa in qas["questions"]:
                query_id = qa["uid"]
                pred_answer, pred_scale = None, None
                if query_id in predicted_answers:
                    pred_answer, pred_scale = predicted_answers[query_id]
                
                # 评估单个问题
                em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)
                
                # 获取单个问题的结果（需要手动计算）
                gold_answer = qa.get("answer", [])
                gold_scale = qa.get("scale", "")
                
                # 计算EM和F1
                is_em = _check_exact_match(gold_answer, pred_answer, gold_scale, pred_scale)
                f1_score = _compute_f1_score(gold_answer, pred_answer)
                scale_match = (gold_scale == pred_scale) if gold_scale and pred_scale else (not gold_scale and not pred_scale)
                
                detailed_results.append({
                    "uid": query_id,
                    "question": qa.get("question", ""),
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer if isinstance(pred_answer, list) else [pred_answer] if pred_answer else [],
                    "gold_scale": gold_scale,
                    "pred_scale": pred_scale or "",
                    "exact_match": is_em,
                    "f1_score": f1_score,
                    "scale_match": scale_match
                })
        
        # 获取总体指标
        global_em, global_f1, global_scale, global_op = em_and_f1.get_overall_metric()
        
        # 获取详细指标
        detail_raw = em_and_f1.get_raw_pivot_table()
        detail_em, detail_f1 = em_and_f1.get_detail_metric()
        
        # 将DataFrame转换为可序列化的格式
        def df_to_serializable(obj):
            """将DataFrame或其他对象转换为可序列化的格式"""
            if hasattr(obj, 'to_dict'):
                try:
                    # 先尝试转换为records格式
                    records = obj.to_dict(orient='records')
                    # 检查是否有tuple key，如果有则转换为字符串
                    def clean_dict(d):
                        if isinstance(d, dict):
                            return {str(k) if not isinstance(k, (str, int, float, bool)) or k is None else k: clean_dict(v) for k, v in d.items()}
                        elif isinstance(d, list):
                            return [clean_dict(item) for item in d]
                        else:
                            return d
                    return clean_dict(records)
                except:
                    # 如果失败，转换为字符串表示
                    return obj.to_string() if hasattr(obj, 'to_string') else str(obj)
            return str(obj)
        
        # 构建结果字典
        results = {
            "overall_metrics": {
                "exact_match_accuracy": float(global_em * 100),
                "f1_score": float(global_f1 * 100),
                "scale_score": float(global_scale * 100),
                "operation_score": float(global_op * 100) if global_op else None
            },
            "detail_metrics": {
                "raw_pivot_table": df_to_serializable(detail_raw),
                "em_detail": df_to_serializable(detail_em),
                "f1_detail": df_to_serializable(detail_f1)
            },
            "per_sample_results": detailed_results,
            "summary": {
                "total_samples": len(detailed_results),
                "exact_match_count": sum(1 for r in detailed_results if r["exact_match"]),
                "scale_match_count": sum(1 for r in detailed_results if r["scale_match"]),
                "average_f1": sum(r["f1_score"] for r in detailed_results) / len(detailed_results) if detailed_results else 0.0
            }
        }
        
        # 打印结果
        print("----")
        print("Exact-match accuracy {0:.2f}".format(global_em * 100))
        print("F1 score {0:.2f}".format(global_f1 * 100))
        print("Scale score {0:.2f}".format(global_scale * 100))
        print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
        print("----")
        print("\n详细结果（每个样本）：")
        for result in detailed_results:
            status = "✓" if result["exact_match"] else "✗"
            print(f"{status} {result['uid']}: EM={result['exact_match']}, F1={result['f1_score']:.3f}, "
                  f"Scale={result['scale_match']}")
            print(f"   预测: {result['pred_answer']} (scale: {result['pred_scale']})")
            print(f"   标准: {result['gold_answer']} (scale: {result['gold_scale']})")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 评估结果已保存到: {output_file}")
        
        return results
        
    except ImportError as e:
        print(f"❌ 无法导入TAT-QA评估器: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def _check_exact_match(gold_answer: List[str], pred_answer: Any, gold_scale: str, pred_scale: str) -> bool:
    """检查答案是否完全匹配"""
    from tatqa_utils import normalize_answer
    
    if not pred_answer:
        return False
    
    # 处理pred_answer格式
    if isinstance(pred_answer, list):
        pred_list = pred_answer
    else:
        pred_list = [pred_answer]
    
    # 标准化答案
    gold_normalized = [normalize_answer(str(a)) for a in gold_answer]
    pred_normalized = [normalize_answer(str(a)) for a in pred_list]
    
    # 检查答案是否匹配
    answer_match = set(gold_normalized) == set(pred_normalized)
    
    # 检查scale是否匹配
    scale_match = (gold_scale == pred_scale) if gold_scale and pred_scale else (not gold_scale and not pred_scale)
    
    return answer_match and scale_match


def _compute_f1_score(gold_answer: List[str], pred_answer: Any) -> float:
    """计算F1分数"""
    from tatqa_metric import _answer_to_bags, _align_bags
    
    if not pred_answer:
        return 0.0
    
    # 处理pred_answer格式
    if isinstance(pred_answer, list):
        pred_list = pred_answer
    else:
        pred_list = [pred_answer]
    
    try:
        _, pred_bags = _answer_to_bags(pred_list)
        _, gold_bags = _answer_to_bags(gold_answer)
        
        if not gold_bags:
            return 1.0 if not pred_bags else 0.0
        
        aligned_scores = _align_bags(pred_bags, gold_bags)
        return float(aligned_scores.mean()) if len(aligned_scores) > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_wtq_with_output(pred_file: str, gold_file: str, output_file: str = None):
    """WTQ评估并保存结果"""
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
    detailed_results = []
    correct = 0
    total = 0
    
    def normalize(s):
        s = str(s).lower().strip()
        import string
        s = ''.join(c for c in s if c not in string.punctuation)
        return s
    
    for ex_id, gold_answer in gold_answers.items():
        if ex_id not in predictions:
            print(f"⚠️  预测结果中缺少样本: {ex_id}")
            continue
        
        pred_answer = predictions[ex_id]
        total += 1
        
        pred_norm = normalize(pred_answer)
        gold_norm = normalize(gold_answer)
        
        is_correct = gold_norm in pred_norm or pred_norm in gold_norm or pred_norm == gold_norm
        
        if is_correct:
            correct += 1
        
        detailed_results.append({
            "id": ex_id,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "exact_match": is_correct
        })
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {ex_id}: 预测='{pred_answer[:50]}...' vs 标准='{gold_answer}'")
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    print(f"\n准确率: {correct}/{total} = {accuracy:.2f}%")
    
    # 构建结果字典
    results = {
        "overall_metrics": {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_count": total
        },
        "per_sample_results": detailed_results
    }
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 评估结果已保存到: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="运行官方评估器并保存评估结果")
    parser.add_argument("--dataset", type=str, choices=["wtq", "tat"], required=True, help="数据集类型")
    parser.add_argument("--pred_file", type=str, required=True, help="预测结果文件")
    parser.add_argument("--gold_file", type=str, required=True, help="标准答案文件")
    parser.add_argument("--output_file", type=str, default=None, help="输出结果文件（JSON格式）")
    args = parser.parse_args()
    
    # 如果没有指定输出文件，自动生成
    if not args.output_file:
        base_name = os.path.splitext(os.path.basename(args.pred_file))[0]
        args.output_file = f"{base_name}_evaluation_results.json"
    
    if args.dataset == "wtq":
        evaluate_wtq_with_output(args.pred_file, args.gold_file, args.output_file)
    elif args.dataset == "tat":
        evaluate_tatqa_with_output(args.pred_file, args.gold_file, args.output_file)


if __name__ == "__main__":
    main()
