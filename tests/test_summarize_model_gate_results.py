from pathlib import Path
import json
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "server"))

from summarize_model_gate_results import summarize_gate_results, render_markdown  # noqa: E402


def write_eval(path: Path, *, samples: int, accuracy: float, tokens: float, failed: int = 0, missing: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "result_schema": "myagent",
                "num_samples": samples,
                "num_with_gold": samples,
                "primary_accuracy": accuracy,
                "num_failed_exec": failed,
                "num_missing_answer": missing,
                "avg_total_tokens": tokens,
            }
        ),
        encoding="utf-8",
    )


class SummarizeModelGateResultsTests(unittest.TestCase):
    def test_no_go_when_correct_below_reference(self):
        """Catches expanding a model whose Gate-50 accuracy is below the Qwen3-32B reference."""
        with tempfile.TemporaryDirectory() as tmp_name:
            gate_root = Path(tmp_name) / "myagent_gate50"
            write_eval(gate_root / "eval" / "wtq_model_eval.json", samples=50, accuracy=0.74, tokens=6300)
            write_eval(gate_root / "eval" / "tabfact_model_eval.json", samples=50, accuracy=0.88, tokens=2500)
            write_eval(gate_root / "eval" / "crt_model_eval.json", samples=50, accuracy=0.54, tokens=13200)

            summary = summarize_gate_results(
                gate_root=gate_root,
                model_tag="qwen3_14b_awq",
                reference_correct=124,
                mact_avg_tokens=11262.41,
            )

        self.assertEqual(summary["overall"]["correct"], 108)
        self.assertEqual(summary["overall"]["rows"], 150)
        self.assertEqual(summary["decision"], "no-go")
        self.assertIn("overall_correct_below_reference", summary["decision_reasons"])

    def test_gate150_when_accuracy_failures_and_tokens_pass(self):
        """Catches blocking a model that satisfies the Gate-50 expansion criteria."""
        with tempfile.TemporaryDirectory() as tmp_name:
            gate_root = Path(tmp_name) / "myagent_gate50"
            write_eval(gate_root / "eval" / "wtq_model_eval.json", samples=50, accuracy=0.70, tokens=6560.84)
            write_eval(gate_root / "eval" / "tabfact_model_eval.json", samples=50, accuracy=0.96, tokens=2100.0)
            write_eval(gate_root / "eval" / "crt_model_eval.json", samples=50, accuracy=0.82, tokens=13138.0)

            summary = summarize_gate_results(
                gate_root=gate_root,
                model_tag="qwen3_32b",
                reference_correct=124,
                mact_avg_tokens=11262.41,
            )
            markdown = render_markdown(summary)

        self.assertEqual(summary["overall"]["correct"], 124)
        self.assertEqual(summary["decision"], "gate150")
        self.assertLess(summary["overall"]["token_ratio_to_mact"], 0.75)
        self.assertIn("Gate-50 Summary", markdown)
        self.assertIn("qwen3_32b", markdown)
        self.assertIn("124/150", markdown)
        self.assertIn("gate150", markdown)

    def test_incomplete_when_any_dataset_eval_is_missing(self):
        """Catches silent decisions from partial WTQ/TabFact/CRT outputs."""
        with tempfile.TemporaryDirectory() as tmp_name:
            gate_root = Path(tmp_name) / "myagent_gate50"
            write_eval(gate_root / "eval" / "wtq_model_eval.json", samples=50, accuracy=0.70, tokens=6500)
            write_eval(gate_root / "eval" / "tabfact_model_eval.json", samples=50, accuracy=0.96, tokens=2100)

            summary = summarize_gate_results(
                gate_root=gate_root,
                model_tag="partial_model",
                reference_correct=124,
                mact_avg_tokens=11262.41,
            )

        self.assertEqual(summary["decision"], "incomplete")
        self.assertEqual(summary["missing_tasks"], ["crt"])


if __name__ == "__main__":
    unittest.main()
