"""Custom table QA pipeline for myAgent.

This module defines a simplified agent architecture:
- TQASessionState: per-sample state container
- RouterAgent: decides SIMPLE vs COMPLEX based on semantic + structural scores
- PlannerAgent: generates [PLAN] + [CODE] from question + table schema
- Calculator: safely executes generated code on a pandas DataFrame
- CriticAgent: lightweight checker; can request REPLAN via feedback
- FinalAnswerAgent: formats final natural language answer
- TableQAPipeline: orchestrator that wires everything together
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from utils import table2df


class TQASessionState:
    """Unified state container for one table QA session.

    One question + one table => one state object.
    All intermediate information (routing scores, plan, code, execution
    results, critic feedback, final answer) is stored here.
    """

    def __init__(self, question: str, df: pd.DataFrame, table_schema: Dict[str, Any]):
        self.question: str = question
        self.df: pd.DataFrame = df
        self.table_schema: Dict[str, Any] = table_schema

        # Routing / 难度与路径信息
        self.route_type: Optional[str] = None  # "SIMPLE" or "COMPLEX"
        self.sem_score: Optional[float] = None
        self.cell_score: Optional[float] = None
        self.coarse_intent: Optional[str] = None
        self.selected_columns: List[str] = []
        self.row_filter: Dict[str, Any] = {}
        self.table_reduced: bool = False
        self.original_shape: Optional[Tuple[int, int]] = None
        self.reduced_shape: Optional[Tuple[int, int]] = None
        self.semantic_features: Dict[str, Any] = {}
        self.structural_features: Dict[str, Any] = {}
        self.difficulty_score: Optional[float] = None  # 0~1 综合难度分
        self.difficulty_level: Optional[str] = None  # "easy" / "hard"
        self.routing_context: Dict[str, Any] = {}

        # Planner
        self.planner_raw_output: str = ""
        self.plan_steps: List[str] = []
        self.code_str: str = ""

        # Calculator
        self.exec_success: bool = False
        self.exec_error: Optional[str] = None
        self.exec_locals: Dict[str, Any] = {}
        self.final_value: Any = None  # scalar or small object

        # Critic
        self.critic_raw_output: str = ""
        self.critic_verdict: Optional[str] = None  # "PASS" or "REPLAN"
        self.critic_feedback: str = ""

        # Final answer
        self.final_answer: Optional[str] = None
        self.simple_answer: Optional[str] = None


# ------------------------ helpers ------------------------


def _build_table_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a lightweight schema from a DataFrame.

    Currently includes:
    - columns: list of column names
    - preview_text: a short textual preview of the first few rows
    - num_rows / num_cols
    - row_names_str / col_names_str
    """
    columns = list(df.columns)
    preview_df = df.head(5)
    preview_text = preview_df.to_string(index=False)
    num_rows, num_cols = df.shape

    col_labels = [str(c).strip() for c in df.columns.tolist()]
    col_names_str: Optional[str] = "##".join(col_labels) if any(col_labels) else None

    if not isinstance(df.index, pd.RangeIndex):
        idx_labels = [str(i).strip() for i in df.index.tolist()]
        row_names_str: Optional[str] = "##".join(idx_labels) if any(idx_labels) else None
    else:
        row_names_str = None
    return {
        "columns": columns,
        "preview_text": preview_text,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "row_names_str": row_names_str,
        "col_names_str": col_names_str,
    }


def build_df_from_table(table: Any) -> pd.DataFrame:
    """Convert MACT-style table_text (list-of-lists) to a DataFrame.

    Uses utils.table2df to get executable code that reconstructs df.
    """
    if isinstance(table, pd.DataFrame):
        return table
    if not isinstance(table, list):
        raise ValueError("Unsupported table format for DataFrame construction.")
    df_code = table2df(table)
    local_env: Dict[str, Any] = {}
    exec(df_code, {}, local_env)  # defines `df` in local_env
    df = local_env.get("df", None)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Failed to construct DataFrame from table.")
    return df


def load_csv_row_col_names(csv_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read a CSV file and return row/column names as two strings (or None).

    - Column names: if there is at least one non-empty column label, join
      them with '##' into col_names_str.
    - Row names: only if df.index is not a RangeIndex and has at least one
      non-empty label; join them with '##' into row_names_str.

    Returns:
        row_names_str, col_names_str
    """
    df = pd.read_csv(csv_path)

    col_labels = [str(c).strip() for c in df.columns.tolist()]
    col_names_str: Optional[str] = "##".join(col_labels) if any(col_labels) else None

    if not isinstance(df.index, pd.RangeIndex):
        idx_labels = [str(i).strip() for i in df.index.tolist()]
        row_names_str: Optional[str] = "##".join(idx_labels) if any(idx_labels) else None
    else:
        row_names_str = None

    return row_names_str, col_names_str


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    sub = text.split(start, 1)[1]
    if end in sub:
        sub = sub.split(end, 1)[0]
    return sub.strip()


def _find_line_startswith(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return ""


# ------------------------ Router ------------------------


ROUTER_PROMPT_TEMPLATE = """你是一个表格问答路由器。\n\n请基于问题语义和表格结构，输出以下 JSON：\n{\n  \"sem_score\": 0-1,\n  \"coarse_intent\": \"lookup|filter|aggregation|comparison|multi_step|other\",\n  \"selected_columns\": [\"col1\", \"col2\"],\n  \"row_filter\": {\"column\": \"列名\", \"values\": [\"值1\", \"值2\"]},\n  \"semantic_flags\": {\"has_aggregation\": bool, \"has_comparison\": bool, \"has_temporal_reasoning\": bool, \"has_multi_step\": bool, \"has_ranking\": bool, \"num_constraints\": int}\n}\n\n要求：\n- 必须输出 JSON，不能附加其他文本。\n- selected_columns 为空表示不确定。\n- row_filter 为空对象表示不确定。\n\n[Question]\n{question}\n\n[Table Columns]\n{col_names}\n\n[Answer JSON]\n"""


class InputHandler:
    """输入管理模块：规范化问题与表格，构建轻量 schema。"""

    @staticmethod
    def process_input(question: str, table: Any) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
        normalized_question = (question or "").strip()
        df = build_df_from_table(table)
        table_schema = _build_table_schema(df)
        return normalized_question, df, table_schema


class FeatureExtractor:
    """语义特征抽取：生成 sem_score + 意图 + 行列提示。"""

    def __init__(self, llm_fn: Callable[[str], str], prompt_template: str = ROUTER_PROMPT_TEMPLATE):
        self.llm_fn = llm_fn
        self.prompt_tmpl = prompt_template

    def _fallback(self, question: str) -> Dict[str, Any]:
        complex_keywords = [
            "增长", "增幅", "占比", "比例", "同比", "环比", "平均",
            "总和", "合计", "总计", "变化", "差值", "增速", "下降",
        ]
        has_complex = any(k in question for k in complex_keywords)
        sem_score = 0.7 if has_complex else 0.3
        return {
            "sem_score": sem_score,
            "coarse_intent": "aggregation" if has_complex else "lookup",
            "selected_columns": [],
            "row_filter": {},
            "semantic_flags": {
                "has_aggregation": has_complex,
                "has_comparison": False,
                "has_temporal_reasoning": "年" in question,
                "has_multi_step": has_complex,
                "has_ranking": False,
                "num_constraints": 1 if "年" in question else 0,
            },
        }

    def extract(self, question: str, table_schema: Dict[str, Any]) -> Tuple[float, Dict[str, Any], Dict[str, Any], str, List[str], Dict[str, Any]]:
        col_names = "##".join([str(c) for c in table_schema.get("columns", [])])
        prompt = self.prompt_tmpl.format(question=question, col_names=col_names)
        raw = (self.llm_fn(prompt) or "").strip()
        data: Dict[str, Any]
        try:
            # 允许输出被包裹在代码块里
            if raw.startswith("```"):
                raw = raw.strip("` \n")
                raw = raw.split("\n", 1)[-1]
            data = json.loads(raw)
        except Exception:
            data = self._fallback(question)

        sem_score = float(data.get("sem_score", 0.5))
        sem_score = max(0.0, min(1.0, sem_score))
        coarse_intent = data.get("coarse_intent", "other")
        selected_columns = data.get("selected_columns", []) or []
        row_filter = data.get("row_filter", {}) or {}
        semantic_flags = data.get("semantic_flags", {}) or {}

        semantic_features = {
            "sem_score": sem_score,
            "semantic_flags": semantic_flags,
        }
        structural_hints = {
            "selected_columns": selected_columns,
            "row_filter": row_filter,
        }
        return sem_score, semantic_features, structural_hints, coarse_intent, selected_columns, row_filter


class DifficultyScorer:
    """综合难度评分器。"""

    def __init__(self, w_sem: float = 0.6, w_cell: float = 0.4, threshold: float = 0.5):
        self.w_sem = w_sem
        self.w_cell = w_cell
        self.threshold = threshold

    def score(self, sem_score: float, cell_score: float) -> Tuple[float, str]:
        total_score = self.w_sem * sem_score + self.w_cell * cell_score
        total_score = max(0.0, min(1.0, total_score))
        difficulty_level = "easy" if total_score < self.threshold else "hard"
        return total_score, difficulty_level


class RouterAgent:
    """Router 模块：抽取特征、压缩表格、计算分数并做路径决策。"""

    def __init__(self, llm_fn: Callable[[str], str]):
        self.feature_extractor = FeatureExtractor(llm_fn=llm_fn)
        self.difficulty_scorer = DifficultyScorer()

    @staticmethod
    def _filter_rows_by_column_values(df: pd.DataFrame, column: str, values: List[Any]) -> pd.DataFrame:
        if column not in df.columns:
            return df
        if not values:
            return df
        return df[df[column].isin(values)]

    def _reduce_df(self, df: pd.DataFrame, selected_columns: List[str], row_filter: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        original_shape = df.shape

        # 列裁剪
        if selected_columns:
            kept_cols = [c for c in df.columns if str(c) in [str(x) for x in selected_columns]]
            reduced_df = df[kept_cols] if kept_cols else df
        else:
            reduced_df = df

        # 行裁剪（支持 {"column": "...", "values": [...] }）
        col = row_filter.get("column")
        values = row_filter.get("values") or []
        if col and values:
            reduced_df = self._filter_rows_by_column_values(reduced_df, col, values)

        reduced_shape = reduced_df.shape
        table_reduced = reduced_shape != original_shape

        reduce_info = {
            "table_reduced": table_reduced,
            "original_shape": list(original_shape),
            "reduced_shape": list(reduced_shape),
        }
        return reduced_df, reduce_info

    @staticmethod
    def _compute_cell_score(original_df: pd.DataFrame, reduced_df: pd.DataFrame) -> float:
        original_rows, original_cols = original_df.shape
        reduced_rows, reduced_cols = reduced_df.shape
        original_size = max(1, original_rows * max(1, original_cols))
        reduced_size = max(1, reduced_rows * max(1, reduced_cols))
        cell_score = reduced_size / original_size
        return max(0.0, min(1.0, cell_score))

    def route(self, state: TQASessionState) -> TQASessionState:
        question = state.question or ""
        original_df = state.df

        # 1) 特征抽取：sem_score + 结构提示
        sem_score, semantic_features, structural_hints, coarse_intent, selected_columns, row_filter = (
            self.feature_extractor.extract(question, state.table_schema)
        )
        state.sem_score = sem_score
        state.coarse_intent = coarse_intent
        state.selected_columns = selected_columns
        state.row_filter = row_filter
        state.semantic_features = semantic_features

        # 2) 压缩表格
        reduced_df, reduce_info = self._reduce_df(state.df, selected_columns, row_filter)
        state.original_shape = tuple(reduce_info["original_shape"])
        state.reduced_shape = tuple(reduce_info["reduced_shape"])
        state.table_reduced = reduce_info["table_reduced"]
        if state.table_reduced:
            state.df = reduced_df

        # 3) cell_score
        reduced_rows, reduced_cols = state.df.shape
        estimated_cells = max(1, reduced_rows * max(1, reduced_cols))
        cell_score = self._compute_cell_score(
            original_df=original_df,
            reduced_df=state.df,
        )
        state.cell_score = cell_score
        state.structural_features = {
            "cell_score": cell_score,
            "estimated_cells_touched": estimated_cells,
        }

        # 4) 综合难度
        total_score, difficulty_level = self.difficulty_scorer.score(sem_score, cell_score)
        state.difficulty_score = total_score
        state.difficulty_level = difficulty_level

        # 5) 路由决策
        state.route_type = "SIMPLE" if difficulty_level == "easy" else "COMPLEX"
        state.routing_context = {
            "sem_score": sem_score,
            "cell_score": cell_score,
            "estimated_cells_touched": estimated_cells,
            "total_score": total_score,
            "difficulty_level": difficulty_level,
            "coarse_intent": coarse_intent,
            "selected_columns": selected_columns,
            "row_filter": row_filter,
            "table_reduced": state.table_reduced,
            "original_shape": list(state.original_shape) if state.original_shape else None,
            "reduced_shape": list(state.reduced_shape) if state.reduced_shape else None,
        }
        return state


# ------------------------ Simple Path ------------------------


SIMPLE_PATH_PROMPT = """你是一个表格问答助手。\n\n【问题】\n{question}\n\n【表格预览】\n{table_preview}\n\n请直接给出答案，1-2 句中文即可。\n如果可以，说明涉及的年份/地区/指标名称。\n"""


class SimplePathAgent:
    """简单路径：一次 LLM 直接回答。"""

    def __init__(self, llm_fn: Callable[[str], str], prompt_template: str = SIMPLE_PATH_PROMPT):
        self.llm_fn = llm_fn
        self.prompt_tmpl = prompt_template

    def run_simple_path(self, state: TQASessionState) -> TQASessionState:
        table_preview = state.table_schema.get("preview_text", "")
        prompt = self.prompt_tmpl.format(
            question=state.question,
            table_preview=table_preview,
        )
        answer = self.llm_fn(prompt)
        state.simple_answer = (answer or "").strip()
        state.final_answer = state.simple_answer
        return state



# ------------------------ Planner ------------------------


PLANNER_PROMPT_TEMPLATE = """You are a table reasoning planner and programmer.

You are given:
- A Chinese question about a governmental statistics table.
- A table represented as a pandas DataFrame `df`.

Your job:
1. First, write a step-by-step plan in natural language to solve the question.
2. Then, write executable Python code that uses ONLY the given DataFrame `df`
   (and standard Python/pandas operations) to compute the final answer.

Requirements:
- In the [PLAN] section, number the steps: Step1, Step2, ...
- In the [CODE] section, write valid Python code.
- Assume `df` is already defined as a pandas DataFrame with the following columns:
  {col_names}
- Use column names exactly as shown.
- At the end of your code, assign the final scalar answer to a variable named:
  final_answer_value

Question:
{question}

Table Schema (first few rows):
{table_preview}

If there was a previous critic feedback, consider it:
{critic_feedback}

Now produce your reasoning plan and code with the following format:

[PLAN]
Step1: ...
Step2: ...
...

[CODE]
# your python code here
...
final_answer_value = ...
"""


class PlannerAgent:
    def __init__(self, llm_fn: Callable[[str], str], prompt_template: str = PLANNER_PROMPT_TEMPLATE):
        self.llm_fn = llm_fn
        self.prompt_tmpl = prompt_template

    def plan(self, state: TQASessionState) -> TQASessionState:
        # Build prompt from question + table schema + critic feedback
        columns = state.table_schema.get("columns", [])
        col_names = ", ".join(str(c) for c in columns)
        table_preview = state.table_schema.get("preview_text", "")
        critic_feedback = state.critic_feedback or "无"

        prompt = self.prompt_tmpl.format(
            col_names=col_names,
            question=state.question,
            table_preview=table_preview,
            critic_feedback=critic_feedback,
        )
        raw = self.llm_fn(prompt)
        state.planner_raw_output = raw

        plan_text = _between(raw, "[PLAN]", "[CODE]")
        code_text = raw.split("[CODE]", 1)[-1] if "[CODE]" in raw else ""
        state.code_str = code_text.strip()

        plan_lines = []
        for line in plan_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("step"):
                plan_lines.append(stripped)
        state.plan_steps = plan_lines

        return state


# ------------------------ Calculator ------------------------


class Calculator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _safe_execute(code_str: str, df: pd.DataFrame) -> Dict[str, Any]:
        import pandas as pd  # local import for sandbox globals

        allowed_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "round": round,
            "sorted": sorted,
        }
        global_env: Dict[str, Any] = {
            "__builtins__": allowed_builtins,
            "pd": pd,
        }
        local_env: Dict[str, Any] = {"df": df}
        exec(code_str, global_env, local_env)
        return local_env

    def execute(self, state: TQASessionState) -> TQASessionState:
        """Execute planner-generated code on state.df and update execution fields."""
        code = state.code_str or ""
        if not code.strip():
            state.exec_success = False
            state.exec_error = "Empty code_str from planner."
            state.final_value = None
            state.exec_locals = {}
            return state

        try:
            local_env = self._safe_execute(code, state.df)
            state.exec_success = True
            state.exec_locals = local_env
            state.final_value = local_env.get("final_answer_value", None)
            state.exec_error = None
        except Exception as e:  # noqa: BLE001
            state.exec_success = False
            state.exec_error = str(e)
            state.final_value = None
            state.exec_locals = {}
        return state


# ------------------------ Critic ------------------------


CRITIC_PROMPT_TEMPLATE = """You are a careful auditor for table-based numerical reasoning.

You will be given:
- A question about a governmental statistics table.
- A reasoning plan (PLAN) written in steps.
- A Python code snippet (CODE) that was executed on the table DataFrame `df`.
- The execution result (RESULT), or an error if execution failed.

Your job:
1. Check whether the plan and the code correctly answer the question.
2. Check whether the result is reasonable and matches the question.
3. If you find serious issues, ask to REPLAN and describe what to fix.
4. Otherwise, PASS and briefly confirm correctness.

Output in the following format:

[VERDICT] PASS or REPLAN

[COMMENT]
(1-3 sentences explanation, in Chinese)
[/COMMENT]

[HINT_FOR_PLANNER]
(if REPLAN, give concrete suggestions how to modify PLAN or CODE; if PASS, you can say "保持当前方案")
[/HINT_FOR_PLANNER]


[QUESTION]
{question}

[PLAN]
{plan_text}

[CODE]
{code_str}

[RESULT]
success = {exec_success}
final_answer_value = {final_value}
error = {exec_error}
"""


class CriticAgent:
    def __init__(self, llm_fn: Callable[[str], str], prompt_template: str = CRITIC_PROMPT_TEMPLATE):
        self.llm_fn = llm_fn
        self.prompt_tmpl = prompt_template

    def review(self, state: TQASessionState) -> TQASessionState:
        # Check PLAN / CODE / execution result and decide PASS / REPLAN + feedback
        plan_text = "\n".join(state.plan_steps) if state.plan_steps else state.planner_raw_output
        code_str = state.code_str
        prompt = self.prompt_tmpl.format(
            question=state.question,
            plan_text=plan_text,
            code_str=code_str,
            exec_success=state.exec_success,
            final_value=state.final_value,
            exec_error=state.exec_error,
        )
        raw = self.llm_fn(prompt)
        state.critic_raw_output = raw

        verdict_line = _find_line_startswith(raw, "[VERDICT]")
        if "REPLAN" in verdict_line.upper():
            state.critic_verdict = "REPLAN"
        else:
            state.critic_verdict = "PASS"

        comment = _between(raw, "[COMMENT]", "[/COMMENT]")
        hint = _between(raw, "[HINT_FOR_PLANNER]", "[/HINT_FOR_PLANNER]")
        state.critic_feedback = hint or comment

        return state


# ------------------------ Final answer ------------------------


COMPLEX_ANSWER_PROMPT = """你是一名政务统计分析助手，请根据【问题】【推理步骤】【最终数值结果】生成简洁、正式的中文回答。

【问题】
{question}

【推理步骤】
{plan_steps}

【最终计算结果】
final_answer_value = {final_value}

要求：
1. 用 2–3 句话回答。
2. 首句直接给出数值答案，并说明单位（如果题目中能看出）。
3. 第二句简要说明是基于哪几年/哪些指标进行计算的（可参考推理步骤）。
4. 不要暴露 Python 代码。

现在给出回答：
"""

SIMPLE_ANSWER_PROMPT = """你是一个政务统计表格问答助手。

给定一个简单问题和表格（已在系统中解析），系统已经为你定位了目标单元格的数值：
value = {simple_answer_value}

【问题】
{question}

请用 1–2 句中文给出直接答案，并简单提及年份/地区/指标名称（如果题目中能看出）。
"""


class FinalAnswerAgent:
    def __init__(
        self,
        llm_fn: Callable[[str], str],
        complex_prompt: str = COMPLEX_ANSWER_PROMPT,
        simple_prompt: str = SIMPLE_ANSWER_PROMPT,
    ):
        self.llm_fn = llm_fn
        self.complex_prompt = complex_prompt
        self.simple_prompt = simple_prompt

    def respond(self, state: TQASessionState) -> TQASessionState:
        # SIMPLE: directly answer based on question + table preview
        # COMPLEX: use PLAN + final_value to format a more formal answer
        if state.route_type == "SIMPLE":
            table_preview = state.table_schema.get("preview_text", "")
            prompt = (
                "你是一个政务统计表格问答助手。\n\n"
                "【问题】\n"
                f"{state.question}\n\n"
                "【表格预览】\n"
                f"{table_preview}\n\n"
                "请根据表格内容，用 1–2 句中文给出直接答案，"
                "并尽量提及涉及的年份、地区和指标名称（如果题目中能看出）。"
            )
        else:
            plan_steps_text = "\n".join(state.plan_steps)
            prompt = self.complex_prompt.format(
                question=state.question,
                plan_steps=plan_steps_text,
                final_value=state.final_value,
            )
        answer = self.llm_fn(prompt)
        state.final_answer = answer.strip()
        return state


# ------------------------ Orchestrator ------------------------


class TableQAPipeline:
    """High-level orchestrator that wires all agents together.

    External interface: run(state) → update state in-place.
    Internally calls Router / Planner / Calculator / Critic / FinalAnswer.
    """

    def __init__(
        self,
        router: RouterAgent,
        planner: PlannerAgent,
        calculator: Calculator,
        critic: CriticAgent,
        final_answer_agent: FinalAnswerAgent,
        simple_agent: Optional[SimplePathAgent] = None,
        max_replan: int = 2,
    ) -> None:
        self.router = router
        self.planner = planner
        self.calculator = calculator
        self.critic = critic
        self.final_answer_agent = final_answer_agent
        self.simple_agent = simple_agent
        self.max_replan = max_replan

    def build_state_from_table(self, question: str, table: Any) -> TQASessionState:
        df = build_df_from_table(table)
        schema = _build_table_schema(df)
        return TQASessionState(question=question, df=df, table_schema=schema)

    def run(self, state: TQASessionState) -> TQASessionState:
        # 1) routing
        state = self.router.route(state)

        # 2) simple path
        if state.route_type == "SIMPLE":
            if self.simple_agent:
                state = self.simple_agent.run_simple_path(state)
            else:
                state = self.final_answer_agent.respond(state)
            return state

        # 3) complex path with planner / calculator / critic (allowing limited REPLAN)
        for _ in range(self.max_replan):
            state = self.planner.plan(state)
            state = self.calculator.execute(state)
            state = self.critic.review(state)
            if state.critic_verdict == "PASS":
                break

        state = self.final_answer_agent.respond(state)
        return state
