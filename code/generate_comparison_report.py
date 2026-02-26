"""生成MyAgent与MACT的对比报告

用法：
python generate_comparison_report.py --eval_file tatqa_evaluation_results.json --output_file comparison_report.md
"""

import argparse
import json
import os
from datetime import datetime


def load_evaluation_results(eval_file: str):
    """加载评估结果"""
    with open(eval_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_comparison_report(eval_file: str, dataset: str = "tat", output_file: str = None):
    """生成对比报告"""
    
    # 加载评估结果
    results = load_evaluation_results(eval_file)
    
    # MACT论文中报告的结果（需要根据实际情况填写）
    # 注意：这些数字是示例，需要根据MACT论文的实际数据填写
    mact_results = {
        "tat": {
            "exact_match": None,  # 需要填写MACT论文中的EM分数
            "f1_score": None,      # 需要填写MACT论文中的F1分数
            "scale_score": None,   # 如果有的话
            "note": "请根据MACT论文填写实际数值"
        },
        "wtq": {
            "exact_match": None,
            "denotation_match": None,
            "note": "请根据MACT论文填写实际数值"
        }
    }
    
    # 获取MyAgent的结果
    myagent_metrics = results.get("overall_metrics", {})
    myagent_em = myagent_metrics.get("exact_match_accuracy", 0)
    myagent_f1 = myagent_metrics.get("f1_score", 0)
    myagent_scale = myagent_metrics.get("scale_score", 0)
    
    # 生成报告
    report_lines = []
    report_lines.append("# MyAgent vs MACT 对比报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**数据集**: {dataset.upper()}")
    report_lines.append(f"**评估结果文件**: {eval_file}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 总体指标对比
    report_lines.append("## 📊 总体指标对比")
    report_lines.append("")
    
    if dataset == "tat":
        report_lines.append("| 指标 | MyAgent | MACT | 差异 |")
        report_lines.append("|------|---------|------|------|")
        
        # EM对比
        mact_em = mact_results["tat"]["exact_match"]
        if mact_em is not None:
            diff_em = myagent_em - mact_em
            diff_str = f"{diff_em:+.2f}%" if diff_em >= 0 else f"{diff_em:.2f}%"
            report_lines.append(f"| **Exact Match (EM)** | {myagent_em:.2f}% | {mact_em:.2f}% | {diff_str} |")
        else:
            report_lines.append(f"| **Exact Match (EM)** | {myagent_em:.2f}% | *待填写* | - |")
        
        # F1对比
        mact_f1 = mact_results["tat"]["f1_score"]
        if mact_f1 is not None:
            diff_f1 = myagent_f1 - mact_f1
            diff_str = f"{diff_f1:+.2f}%" if diff_f1 >= 0 else f"{diff_f1:.2f}%"
            report_lines.append(f"| **F1 Score** | {myagent_em:.2f}% | {mact_f1:.2f}% | {diff_str} |")
        else:
            report_lines.append(f"| **F1 Score** | {myagent_f1:.2f}% | *待填写* | - |")
        
        # Scale对比
        mact_scale = mact_results["tat"]["scale_score"]
        if mact_scale is not None:
            diff_scale = myagent_scale - mact_scale
            diff_str = f"{diff_scale:+.2f}%" if diff_scale >= 0 else f"{diff_scale:.2f}%"
            report_lines.append(f"| **Scale Score** | {myagent_scale:.2f}% | {mact_scale:.2f}% | {diff_str} |")
        else:
            report_lines.append(f"| **Scale Score** | {myagent_scale:.2f}% | *待填写* | - |")
    
    elif dataset == "wtq":
        report_lines.append("| 指标 | MyAgent | MACT | 差异 |")
        report_lines.append("|------|---------|------|------|")
        
        mact_em = mact_results["wtq"]["exact_match"]
        if mact_em is not None:
            diff_em = myagent_em - mact_em
            diff_str = f"{diff_em:+.2f}%" if diff_em >= 0 else f"{diff_em:.2f}%"
            report_lines.append(f"| **Exact Match (EM)** | {myagent_em:.2f}% | {mact_em:.2f}% | {diff_str} |")
        else:
            report_lines.append(f"| **Exact Match (EM)** | {myagent_em:.2f}% | *待填写* | - |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # MyAgent详细结果
    report_lines.append("## 📈 MyAgent详细结果")
    report_lines.append("")
    
    summary = results.get("summary", {})
    report_lines.append(f"- **总样本数**: {summary.get('total_samples', 0)}")
    report_lines.append(f"- **完全匹配数**: {summary.get('exact_match_count', 0)}")
    report_lines.append(f"- **平均F1分数**: {summary.get('average_f1', 0):.4f}")
    report_lines.append("")
    
    # 每个样本的结果
    report_lines.append("### 每个样本的详细结果")
    report_lines.append("")
    
    per_sample = results.get("per_sample_results", [])
    report_lines.append("| UID | 问题 | 预测答案 | 标准答案 | EM | F1 |")
    report_lines.append("|-----|------|----------|----------|----|----|")
    
    for sample in per_sample:
        uid = sample.get("uid", "")
        question = sample.get("question", "")[:50] + "..." if len(sample.get("question", "")) > 50 else sample.get("question", "")
        pred_answer = str(sample.get("pred_answer", []))[:30] + "..." if len(str(sample.get("pred_answer", []))) > 30 else str(sample.get("pred_answer", []))
        gold_answer = str(sample.get("gold_answer", []))[:30] + "..." if len(str(sample.get("gold_answer", []))) > 30 else str(sample.get("gold_answer", []))
        em = "✓" if sample.get("exact_match", False) else "✗"
        f1 = f"{sample.get('f1_score', 0):.3f}"
        
        report_lines.append(f"| {uid} | {question} | {pred_answer} | {gold_answer} | {em} | {f1} |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 错误分析
    report_lines.append("## 🔍 错误分析")
    report_lines.append("")
    
    error_samples = [s for s in per_sample if not s.get("exact_match", False)]
    if error_samples:
        report_lines.append(f"共 {len(error_samples)} 个错误样本：")
        report_lines.append("")
        for sample in error_samples:
            report_lines.append(f"### {sample.get('uid', '')}")
            report_lines.append(f"- **问题**: {sample.get('question', '')}")
            report_lines.append(f"- **预测答案**: {sample.get('pred_answer', [])}")
            report_lines.append(f"- **标准答案**: {sample.get('gold_answer', [])}")
            report_lines.append(f"- **F1分数**: {sample.get('f1_score', 0):.3f}")
            report_lines.append("")
    else:
        report_lines.append("✅ 所有样本都预测正确！")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # 说明
    report_lines.append("## 📝 说明")
    report_lines.append("")
    report_lines.append("1. **MACT结果**: 请根据MACT论文中的实际数据填写上述表格中的MACT分数")
    report_lines.append("2. **评估方式**: 使用官方评估器（TAT-QA官方评估脚本）")
    report_lines.append("3. **数据子集**: 当前使用的是测试样本，如需与MACT论文对比，建议使用相同的数据子集")
    report_lines.append("")
    
    # 生成报告
    report_content = "\n".join(report_lines)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✓ 对比报告已保存到: {output_file}")
    else:
        print(report_content)
    
    return report_content


def main():
    parser = argparse.ArgumentParser(description="生成MyAgent与MACT的对比报告")
    parser.add_argument("--eval_file", type=str, required=True, help="评估结果文件（JSON格式）")
    parser.add_argument("--dataset", type=str, choices=["tat", "wtq"], default="tat", help="数据集类型")
    parser.add_argument("--output_file", type=str, default=None, help="输出报告文件（Markdown格式）")
    parser.add_argument("--mact_em", type=float, default=None, help="MACT的EM分数（可选）")
    parser.add_argument("--mact_f1", type=float, default=None, help="MACT的F1分数（可选）")
    parser.add_argument("--mact_scale", type=float, default=None, help="MACT的Scale分数（可选）")
    
    args = parser.parse_args()
    
    # 如果提供了MACT分数，使用它们
    if args.mact_em is not None or args.mact_f1 is not None:
        # 这里可以扩展以支持自定义MACT分数
        pass
    
    generate_comparison_report(args.eval_file, args.dataset, args.output_file)


if __name__ == "__main__":
    main()
