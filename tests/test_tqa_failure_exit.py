import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import tqa  # noqa: E402


class TqaFailureExitTests(unittest.TestCase):
    def test_row_processing_exception_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "bad_tabfact.jsonl"
            output_path = tmp_path / "out.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "bad-1",
                        "statement": "the table has a missing table_text field",
                        "answer": ["true"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                dataset_path=str(dataset_path),
                output_path=str(output_path),
                append_output=False,
                task="tabfact",
                table_dir="",
                plan_model_name="fake-model",
                enable_multiview_validation=False,
                collaboration_mode="selective",
                disable_strong_verification=True,
                disable_deterministic_shortcuts=False,
                mact_avg_tokens=11460.0,
                max_replan=2,
            )

            with patch.object(tqa, "build_llm_fn", return_value=lambda prompt: "fake"):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(KeyError):
                        tqa.main(args)

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
