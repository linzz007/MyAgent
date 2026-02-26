"""快速查看MyAgent评估结果和与MACT的对比

用法:
python view_comparison.py --eval_file tatqa_evaluation_results.json
python view_comparison.py --eval_file tatqa_evaluation_results.json --mact_em 70.5 --mact_f1 72.3
"""

import argparse
import json
import sys


def print_header(title: str, width: int = 60):
    """打印标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_metrics_bar(label: str, value: float, max_value: float = 100.0, width: int = 40):
    """打印指标条形图"""
    bar_length = int((value / max_value) * width)
    bar = "█" * bar_length + "░" * (width - bar_length)
    print(f"{label:20s} {bar} {value:6.2f}%")


def view_comparison(eval_file: str, mact_em: float = None, mact_f1: float = None, mact_scale: float = None):
    """查看对比结果"""
    
    # 加载评估结果
    with open(eval_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 获取MyAgent的结果
    metrics = results.get("overall_metrics", {})
    myagent_em = metrics.get("exact_match_accuracy", 0)
    myagent_f1 = metrics.get("f1_score", 0)
    myagent_scale = metrics.get("scale_score", 0)
    
    summary = results.get("summary", {})
    per_sample = results.get("per_sample_results", [])
    
    # 打印标题
    print_header("🎯 MyAgent vs MACT 性能对比", 70)
    
    # 打印总体指标对比
    print("\n📊 总体指标对比")
    print("-" * 70)
    print(f"{'指标':<20s} {'MyAgent':<15s} {'MACT':<15s} {'差异':<15s}")
    print("-" * 70)
    
    # EM对比
    if mact_em is not None:
        diff_em = myagent_em - mact_em
        diff_str = f"{diff_em:+.2f}%" if diff_em >= 0 else f"{diff_em:.2f}%"
        status = "✅" if diff_em >= 0 else "❌"
        print(f"{'Exact Match (EM)':<20s} {myagent_em:>6.2f}%{'':<7s} {mact_em:>6.2f}%{'':<7s} {diff_str:<15s} {status}")
    else:
        print(f"{'Exact Match (EM)':<20s} {myagent_em:>6.2f}%{'':<7s} {'待填写':<15s} {'-':<15s} ⏳")
    
    # F1对比
    if mact_f1 is not None:
        diff_f1 = myagent_f1 - mact_f1
        diff_str = f"{diff_f1:+.2f}%" if diff_f1 >= 0 else f"{diff_f1:.2f}%"
        status = "✅" if diff_f1 >= 0 else "❌"
        print(f"{'F1 Score':<20s} {myagent_f1:>6.2f}%{'':<7s} {mact_f1:>6.2f}%{'':<7s} {diff_str:<15s} {status}")
    else:
        print(f"{'F1 Score':<20s} {myagent_f1:>6.2f}%{'':<7s} {'待填写':<15s} {'-':<15s} ⏳")
    
    # Scale对比
    if mact_scale is not None:
        diff_scale = myagent_scale - mact_scale
        diff_str = f"{diff_scale:+.2f}%" if diff_scale >= 0 else f"{diff_scale:.2f}%"
        status = "✅" if diff_scale >= 0 else "❌"
        print(f"{'Scale Score':<20s} {myagent_scale:>6.2f}%{'':<7s} {mact_scale:>6.2f}%{'':<7s} {diff_str:<15s} {status}")
    else:
        print(f"{'Scale Score':<20s} {myagent_scale:>6.2f}%{'':<7s} {'待填写':<15s} {'-':<15s} ⏳")
    
    print("-" * 70)
    
    # 打印可视化对比
    print("\n📈 可视化对比")
    print("-" * 70)
    
    if mact_em is not None:
        print("\nExact Match (EM):")
        print_metrics_bar("MyAgent", myagent_em)
        print_metrics_bar("MACT", mact_em)
        if myagent_em > mact_em:
            print("  ✅ MyAgent表现更好！")
        elif myagent_em < mact_em:
            print("  ❌ MACT表现更好")
        else:
            print("  ➡️  表现相同")
    
    if mact_f1 is not None:
        print("\nF1 Score:")
        print_metrics_bar("MyAgent", myagent_f1)
        print_metrics_bar("MACT", mact_f1)
        if myagent_f1 > mact_f1:
            print("  ✅ MyAgent表现更好！")
        elif myagent_f1 < mact_f1:
            print("  ❌ MACT表现更好")
        else:
            print("  ➡️  表现相同")
    
    # 打印详细统计
    print_header("📊 MyAgent详细统计", 70)
    print(f"总样本数:     {summary.get('total_samples', 0)}")
    print(f"完全匹配数:   {summary.get('exact_match_count', 0)}")
    print(f"Scale匹配数:  {summary.get('scale_match_count', 0)}")
    print(f"平均F1分数:   {summary.get('average_f1', 0):.4f}")
    
    # 打印每个样本的结果
    print_header("📋 每个样本的详细结果", 70)
    print(f"{'UID':<8s} {'状态':<6s} {'问题':<40s} {'F1':<8s}")
    print("-" * 70)
    
    for sample in per_sample:
        uid = sample.get("uid", "")
        question = sample.get("question", "")[:38] + ".." if len(sample.get("question", "")) > 40 else sample.get("question", "")
        em = "✅" if sample.get("exact_match", False) else "❌"
        f1 = f"{sample.get('f1_score', 0):.3f}"
        print(f"{uid:<8s} {em:<6s} {question:<40s} {f1:<8s}")
    
    # 打印错误分析
    error_samples = [s for s in per_sample if not s.get("exact_match", False)]
    if error_samples:
        print_header("🔍 错误分析", 70)
        for sample in error_samples:
            print(f"\n❌ {sample.get('uid', '')}: {sample.get('question', '')}")
            print(f"   预测: {sample.get('pred_answer', [])}")
            print(f"   标准: {sample.get('gold_answer', [])}")
            print(f"   F1: {sample.get('f1_score', 0):.3f}")
    
    print("\n" + "=" * 70)
    
    # 提示信息
    if mact_em is None or mact_f1 is None:
        print("\n💡 提示: 要查看完整对比，请提供MACT的结果:")
        print("   python view_comparison.py --eval_file <file> --mact_em <em> --mact_f1 <f1>")
        print("\n   或者编辑 MyAgent_vs_MACT_对比报告.md 文件填写MACT数据")


def main():
    parser = argparse.ArgumentParser(description="查看MyAgent评估结果和与MACT的对比")
    parser.add_argument("--eval_file", type=str, required=True, help="评估结果文件（JSON格式）")
    parser.add_argument("--mact_em", type=float, default=None, help="MACT的EM分数（可选）")
    parser.add_argument("--mact_f1", type=float, default=None, help="MACT的F1分数（可选）")
    parser.add_argument("--mact_scale", type=float, default=None, help="MACT的Scale分数（可选）")
    
    args = parser.parse_args()
    
    view_comparison(args.eval_file, args.mact_em, args.mact_f1, args.mact_scale)


if __name__ == "__main__":
    main()
