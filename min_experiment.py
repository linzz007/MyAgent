"""
最小实验：在 WTQ 小样本上跑当前 my_agents 流水线，并统计 token 与正确率。

用法示例：
python min_experiment.py --num_questions 5 --api_base http://localhost:8000/v1 --model qwen
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


def _load_my_agents_module() -> Any:
    repo_root = Path(__file__).resolve().parent
    code_dir = repo_root / "MyAgent" / "code"
    import sys

    sys.path.insert(0, str(code_dir))
    from my_agents import (  # noqa: WPS433
        Calculator,
        CriticAgent,
        FinalAnswerAgent,
        InputHandler,
        PlannerAgent,
        RouterAgent,
        SimplePathAgent,
        TableQAPipeline,
        TQASessionState,
    )

    class MyAgentsModule:
        Calculator = Calculator
        CriticAgent = CriticAgent
        FinalAnswerAgent = FinalAnswerAgent
        InputHandler = InputHandler
        PlannerAgent = PlannerAgent
        RouterAgent = RouterAgent
        SimplePathAgent = SimplePathAgent
        TableQAPipeline = TableQAPipeline
        TQASessionState = TQASessionState

    return MyAgentsModule


def _read_wtq_samples(path: Path, n: int) -> List[Dict[str, str]]:
    df = pd.read_csv(path, sep="\t")
    out = []
    for _, row in df.head(n).iterrows():
        out.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "table_path": str(row["table_path"]),
                "answer": str(row["answer"]),
            }
        )
    return out


def _approx_token_len(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback heuristic: ~4 chars per token
        return max(1, int(len(text) / 4))


def build_llm_fn(
    api_base: str,
    model: Optional[str],
    timeout: float = 120.0,
    usage_counter: Optional[Dict[str, int]] = None,
) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=api_base)

    if not model:
        models = client.models.list()
        model = models.data[0].id if models.data else ""
        if not model:
            raise RuntimeError("无法从 /v1/models 获取 model id，请手动传 --model")

    system_rules = (
        "You are a strict formatter.\n"
        "- NEVER output <think> blocks, chain-of-thought, or hidden reasoning.\n"
        "- If the user asks for JSON only, output ONLY valid JSON (no extra text).\n"
        "- If the user asks for tagged blocks like [PLAN]/[CODE], output exactly those blocks.\n"
        "- If the user asks to answer concisely, output ONLY the final answer.\n"
    )

    def _strip_think(text: str) -> str:
        import re

        t = re.sub(r"<think>[\\s\\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        t = re.sub(r"^<think>[\\s\\S]*$", "", t, flags=re.IGNORECASE).strip() or t
        return t

    def llm_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=800,
            timeout=timeout,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = _strip_think(raw)
        out_text = cleaned or raw

        if usage_counter is not None:
            # Prefer API usage if present
            usage = getattr(resp, "usage", None)
            if usage and hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
                usage_counter["prompt_tokens"] += int(usage.prompt_tokens or 0)
                usage_counter["completion_tokens"] += int(usage.completion_tokens or 0)
            else:
                usage_counter["prompt_tokens"] += _approx_token_len(prompt)
                usage_counter["completion_tokens"] += _approx_token_len(out_text)

        return out_text

    return llm_fn


@dataclass
class RunResult:
    qid: str
    question: str
    gold: str
    pred: str
    route: str
    exec_success: Optional[bool]
    is_correct: bool


def _normalize(s: str) -> str:
    return "".join(str(s).strip().lower().split())


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def _extract_pred_from_state(state: Any) -> str:
    for key in ["final_value", "final_answer_value", "simple_answer_value", "simple_value"]:
        if hasattr(state, key):
            val = getattr(state, key)
            if val is None:
                continue
            try:
                if hasattr(val, "item"):
                    val = val.item()
            except Exception:
                pass
            s = str(val).strip()
            if s:
                if s.endswith(".0") and s[:-2].replace("-", "", 1).isdigit():
                    s = s[:-2]
                return s

    for key in ["final_answer", "answer", "output_answer"]:
        if hasattr(state, key):
            val = getattr(state, key)
            if isinstance(val, str):
                import re

                t = re.sub(r"<think>[\\s\\S]*?</think>", "", val, flags=re.IGNORECASE).strip()
                return (t or val).strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description="MyAgent 最小实验（WTQ 小样本）")
    parser.add_argument("--num_questions", type=int, default=5)
    parser.add_argument("--api_base", type=str, default=os.getenv("LOCAL_LLM_API_BASE", "http://localhost:8000/v1"))
    parser.add_argument("--model", type=str, default=os.getenv("LOCAL_LLM_MODEL", None))
    parser.add_argument("--max_replan", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    usage_counter = {"prompt_tokens": 0, "completion_tokens": 0}
    llm_fn = build_llm_fn(args.api_base, args.model, usage_counter=usage_counter)
    my_agents = _load_my_agents_module()

    router = my_agents.RouterAgent(llm_fn)
    planner = my_agents.PlannerAgent(llm_fn)
    calculator = my_agents.Calculator()
    critic = my_agents.CriticAgent(llm_fn)
    final_answer_agent = my_agents.FinalAnswerAgent(llm_fn)
    simple_path_agent = my_agents.SimplePathAgent(llm_fn)

    pipeline = my_agents.TableQAPipeline(
        router=router,
        planner=planner,
        calculator=calculator,
        critic=critic,
        final_answer_agent=final_answer_agent,
        simple_agent=simple_path_agent,
        max_replan=args.max_replan,
    )

    repo_root = Path(__file__).resolve().parent
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wtq_path = repo_root / "MyAgent" / "dataset" / "test_samples_wtq_5.tsv"
    wtq_base = repo_root / "MyAgent" / "dataset" / "WikiTableQuestions"
    samples = _read_wtq_samples(wtq_path, n=args.num_questions)

    results: List[RunResult] = []

    for s in samples:
        csv_path = wtq_base / s["table_path"]
        df = pd.read_csv(csv_path)
        schema = {
            "columns": [str(c) for c in df.columns.tolist()],
            "preview_text": df.head(5).to_string(index=False),
            "num_rows": int(df.shape[0]),
            "num_cols": int(df.shape[1]),
        }
        state = my_agents.TQASessionState(question=s["question"], df=df, table_schema=schema)
        state = pipeline.run(state)
        pred = _extract_pred_from_state(state)
        route = getattr(state, "route_type", "") or ""
        exec_success = getattr(state, "exec_success", None)
        is_correct = _exact_match(pred, s["answer"])
        results.append(
            RunResult(
                qid=s["id"],
                question=s["question"],
                gold=str(s["answer"]),
                pred=pred,
                route=str(route),
                exec_success=exec_success,
                is_correct=is_correct,
            )
        )

        print("\n" + "=" * 80)
        print(f"[{s['id']}] {s['question']}")
        print("-" * 80)
        print(f"route: {route} | exec_success: {exec_success}")
        print(f"pred: {pred}")
        print(f"gold: {s['answer']}")

    acc = sum(1 for r in results if r.is_correct) / max(1, len(results))
    total_tokens = usage_counter["prompt_tokens"] + usage_counter["completion_tokens"]
    avg_tokens = total_tokens / max(1, len(results))

    summary = {
        "num_samples": len(results),
        "accuracy_em": acc,
        "prompt_tokens": usage_counter["prompt_tokens"],
        "completion_tokens": usage_counter["completion_tokens"],
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens,
    }

    out_path = out_dir / f"min_experiment_wtq_{args.num_questions}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": [r.__dict__ for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 80)
    print(f"✓ 完成，共 {len(results)} 条")
    print(f"EM: {acc:.3f} | avg_tokens: {avg_tokens:.1f}")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
