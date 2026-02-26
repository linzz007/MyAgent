"""将MyAgent的输出转换为官方评估器需要的格式

支持：
1. WTQ: 转换为TSV格式（id <tab> answer1 <tab> answer2 ...）
2. TAT-QA: 转换为JSON格式（{"uid": ["answer"], ...}）
"""

import argparse
import json
import sys
from typing import Any, Dict, List


def extract_answer_text(answer_str: str) -> str:
    """从答案字符串中提取纯答案文本
    
    优先提取：
    1. "最终答案："或"Final answer:"后的内容
    2. 货币格式（如 $1,496.5）
    3. 数字格式（如 210, 420.0）
    4. 最后一句中的数值
    """
    if not isinstance(answer_str, str):
        return str(answer_str)
    
    import re
    
    # 1. 尝试提取"最终答案："或"Final answer:"后的内容
    patterns_explicit = [
        r'最终答案[：:]\s*(.+?)(?:\n|$)',
        r'Final answer[：:]\s*(.+?)(?:\n|$)',
        r'答案[：:]\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns_explicit:
        match = re.search(pattern, answer_str, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
    
    # 2. 提取货币格式（如 $1,496.5, $1,146.2）
    currency_pattern = r'\$[\d,]+\.?\d*'
    currency_matches = re.findall(currency_pattern, answer_str)
    if currency_matches:
        # 取最后一个（通常是最准确的答案）
        return currency_matches[-1]
    
    # 3. 提取数字格式（整数或小数）
    # 优先匹配带逗号的数字（如 1,496.5）或普通数字（如 210, 420.0）
    # 注意：需要按长度排序，优先匹配更长的数字（避免"420.0"被拆分成"2"）
    number_patterns = [
        r'\d{1,3}(?:,\d{3})+(?:\.\d+)?',  # 带逗号的数字：1,496.5（至少4位）
        r'\d{3,}(?:\.\d+)?',  # 3位以上的数字：210, 420.0
        r'\d+\.\d+',  # 小数：420.0
        r'\d{2,}',  # 2位以上的整数：210
        r'\d+',  # 任意整数（最后匹配）
    ]
    
    all_matches = []
    for pattern in number_patterns:
        matches = re.findall(pattern, answer_str)
        if matches:
            all_matches.extend(matches)
    
    if all_matches:
        # 按长度和位置排序，取最长的数字（通常是答案）
        all_matches.sort(key=lambda x: (len(x), answer_str.rfind(x)), reverse=True)
        return all_matches[0]
    
    # 4. 如果都没找到，返回原答案（去除前后空白）
    return answer_str.strip()


def convert_to_wtq_format(pred_file: str, output_file: str):
    """将MyAgent的JSONL输出转换为WTQ评估器需要的TSV格式
    
    WTQ评估器需要的格式：
    id <tab> answer1 <tab> answer2 ...
    
    例如：
    nt-0	2004
    nt-1	Wolfe Tones
    """
    results = []
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # 获取ID和答案
            sample_id = item.get('id', '')
            answer = item.get('final_answer', '') or item.get('final_value', '')
            
            if not sample_id:
                print(f"警告: 样本缺少id字段，跳过", file=sys.stderr)
                continue
            
            # 提取纯答案文本
            answer_text = extract_answer_text(answer)
            
            # 将答案转换为列表（WTQ评估器支持多个答案）
            if isinstance(answer_text, list):
                answers = answer_text
            elif isinstance(answer_text, str):
                # 如果答案包含多个句子，尝试提取最后一个
                # 但通常WTQ答案应该是单个值
                answers = [answer_text]
            else:
                answers = [str(answer_text)]
            
            # 写入TSV格式：id <tab> answer1 <tab> answer2 ...
            results.append(f"{sample_id}\t" + "\t".join(str(a) for a in answers))
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(results))
    
    print(f"✓ 已转换 {len(results)} 个样本到WTQ格式: {output_file}")


def convert_to_tat_format(pred_file: str, output_file: str, gold_file: str = None):
    """将MyAgent的JSONL输出转换为TAT-QA评估器需要的JSON格式
    
    TAT-QA评估器需要的格式：
    {
      "uid1": [["answer1"], "scale1"],
      "uid2": [["answer2"], "scale2"],
      ...
    }
    
    参数:
        pred_file: MyAgent预测结果文件（JSONL格式）
        output_file: 输出文件路径
        gold_file: （可选）gold文件路径，用于提取scale信息
    """
    predictions = {}
    
    # 从gold文件中读取scale信息（如果提供）
    scale_map = {}
    if gold_file:
        try:
            with open(gold_file, 'r', encoding='utf-8') as f:
                gold_data = json.load(f)
                for table_entry in gold_data:
                    for qa in table_entry.get('questions', []):
                        uid = qa.get('uid', '')
                        scale = qa.get('scale', '')
                        if uid:
                            scale_map[uid] = scale
        except Exception as e:
            print(f"警告: 无法读取gold文件获取scale信息: {e}", file=sys.stderr)
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # TAT-QA的格式：每个问题有uid
            uid = item.get('uid') or item.get('question_uid') or item.get('id', '')
            answer = item.get('final_answer', '') or item.get('final_value', '')
            
            # 优先使用预测中的scale，否则从gold文件获取
            scale = item.get('scale', '') or scale_map.get(uid, '')
            
            if not uid:
                print(f"警告: 样本缺少uid字段，跳过", file=sys.stderr)
                continue
            
            # 提取纯答案文本
            answer_text = extract_answer_text(answer)
            
            # 将答案转换为列表
            if isinstance(answer_text, list):
                answer_list = answer_text
            elif isinstance(answer_text, str):
                answer_list = [answer_text]
            else:
                answer_list = [str(answer_text)]
            
            # TAT-QA评估器格式：[answer_list, scale]
            predictions[uid] = [answer_list, scale]
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(predictions, fout, ensure_ascii=False, indent=2)
    
    print(f"✓ 已转换 {len(predictions)} 个预测到TAT-QA格式: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="将MyAgent输出转换为官方评估器格式")
    parser.add_argument("--pred_file", type=str, required=True, help="MyAgent预测结果文件（JSONL格式）")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument("--dataset", type=str, choices=["wtq", "tat"], required=True, help="数据集类型")
    parser.add_argument("--gold_file", type=str, default=None, help="（TAT-QA）gold文件路径，用于提取scale信息")
    args = parser.parse_args()
    
    if args.dataset == "wtq":
        convert_to_wtq_format(args.pred_file, args.output_file)
    elif args.dataset == "tat":
        convert_to_tat_format(args.pred_file, args.output_file, gold_file=args.gold_file)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")


if __name__ == "__main__":
    main()
