"""Tests for the autoresearch harness (safety, metric capture, determinism)."""

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_minimal_val_split(tmp_path: Path) -> str:
    """Create a tiny val split pickle for testing."""
    n_users, n_items = 10, 20

    rows = [0, 0, 1, 1, 2, 3]
    cols = [1, 3, 5, 7, 9, 11]
    data = [1.0] * len(rows)
    import pandas as pd
    train_df = pd.DataFrame({"user_idx": rows, "item_idx": cols, "value": data})

    val_interactions = {0: {2, 4}, 1: {6, 8}, 2: {10}, 3: {12}}

    payload = {
        "train_df": train_df,
        "val_interactions": val_interactions,
        "n_users": n_users,
        "n_items": n_items,
        "user_id_to_idx": {},
        "item_id_to_idx": {},
    }
    val_path = tmp_path / "val_split.pkl"
    with open(val_path, "wb") as f:
        pickle.dump(payload, f)
    return str(val_path)


def write_train_py(tmp_path: Path, code: str) -> Path:
    p = tmp_path / "train.py"
    p.write_text(code)
    return p


# ---------------------------------------------------------------------------
# check_imports
# ---------------------------------------------------------------------------

class TestCheckImports:
    def test_clean_script_passes(self, tmp_path):
        from autoresearch.harness import check_imports
        code = (
            "import sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from src.models.popularity import PopularityRecommender\n"
            "print('val_ndcg10: 0.0')\n"
        )
        p = write_train_py(tmp_path, code)
        assert check_imports(p) == []

    def test_banned_subprocess_caught(self, tmp_path):
        from autoresearch.harness import check_imports
        code = "import subprocess\nprint('val_ndcg10: 0.0')\n"
        p = write_train_py(tmp_path, code)
        violations = check_imports(p)
        assert "subprocess" in violations

    def test_banned_socket_caught(self, tmp_path):
        from autoresearch.harness import check_imports
        code = "from socket import gethostname\nprint('val_ndcg10: 0.0')\n"
        p = write_train_py(tmp_path, code)
        violations = check_imports(p)
        assert any("socket" in v for v in violations)

    def test_syntax_error_caught(self, tmp_path):
        from autoresearch.harness import check_imports
        code = "def broken(:\n    pass\n"
        p = write_train_py(tmp_path, code)
        violations = check_imports(p)
        assert any("SyntaxError" in v for v in violations)


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------

class TestParseOutput:
    def test_parses_all_fields(self):
        from autoresearch.harness import parse_output
        stdout = (
            "notes: trying ALS with factors=64\n"
            "model: ALSRecommender\n"
            "val_ndcg10: 0.123456\n"
        )
        score, model, notes = parse_output(stdout)
        assert abs(score - 0.123456) < 1e-8
        assert model == "ALSRecommender"
        assert "ALS" in notes

    def test_missing_metric_returns_neg_inf(self):
        from autoresearch.harness import parse_output
        score, _, _ = parse_output("some output without the metric line")
        assert score == float("-inf")

    def test_scientific_notation(self):
        from autoresearch.harness import parse_output
        score, _, _ = parse_output("val_ndcg10: 1.23e-2\n")
        assert abs(score - 0.0123) < 1e-8


# ---------------------------------------------------------------------------
# harness.run() — integration
# ---------------------------------------------------------------------------

class TestHarnessRun:
    def test_baseline_produces_valid_score(self, tmp_path):
        """A well-formed train.py should return a finite score."""
        from autoresearch.harness import run, parse_output, run_train_py

        val_path = make_minimal_val_split(tmp_path)
        code = f"""
import os, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scipy.sparse import csr_matrix
from src.models.popularity import PopularityRecommender
from src.evaluation.metrics import evaluate_model

val_path = os.environ["MAPLE_VAL_PATH"]
with open(val_path, "rb") as f:
    d = pickle.load(f)

import pandas as pd
train_df = d["train_df"]
n_users, n_items = d["n_users"], d["n_items"]
mat = csr_matrix(
    (train_df["value"].values, (train_df["user_idx"].values, train_df["item_idx"].values)),
    shape=(n_users, n_items)
)
model = PopularityRecommender()
model.fit(mat)
results = evaluate_model(model, d["val_interactions"], k_values=[10], n_items=n_items)
print("notes: test run")
print("model: PopularityRecommender")
print(f"val_ndcg10: {{results.get('ndcg@10', 0.0):.6f}}")
"""
        train_py = write_train_py(tmp_path, code)
        stdout, stderr, rc, elapsed = run_train_py(train_py, val_path)
        assert rc == 0, f"train.py failed:\n{stderr}"
        score, model_name, _ = parse_output(stdout)
        assert score > float("-inf")
        assert score >= 0.0
        assert model_name == "PopularityRecommender"

    def test_timeout_handled_gracefully(self, tmp_path, monkeypatch):
        """A hanging train.py should return -inf without crashing."""
        from autoresearch import harness

        val_path = make_minimal_val_split(tmp_path)
        code = "import time\ntime.sleep(9999)\nprint('val_ndcg10: 1.0')\n"
        train_py = write_train_py(tmp_path, code)

        monkeypatch.setattr(harness, "EXPERIMENT_TIMEOUT", 1)  # 1-second timeout for test
        stdout, stderr, rc, elapsed = harness.run_train_py(train_py, val_path)
        assert rc != 0
        assert "Timeout" in stderr or elapsed < 5  # should bail quickly

    def test_import_violation_blocked(self, tmp_path):
        """train.py with banned import must be blocked before execution."""
        from autoresearch.harness import check_imports
        code = "import requests\nprint('val_ndcg10: 1.0')\n"
        train_py = write_train_py(tmp_path, code)
        violations = check_imports(train_py)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# metric_adapter determinism
# ---------------------------------------------------------------------------

class TestMetricAdapterDeterminism:
    def test_evaluate_same_model_twice_gives_same_score(self):
        """evaluate_model must be deterministic for the same inputs."""
        from src.autoresearch.metric_adapter import compute_val_ndcg10
        from src.models.popularity import PopularityRecommender

        n_users, n_items = 20, 50
        rng = np.random.default_rng(42)
        rows = rng.integers(0, n_users, size=200)
        cols = rng.integers(0, n_items, size=200)
        data = rng.random(size=200) + 0.1

        mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        model = PopularityRecommender()
        model.fit(mat)

        val_interactions = {i: {(i * 3 + 1) % n_items, (i * 7 + 5) % n_items} for i in range(10)}

        score1, _ = compute_val_ndcg10(model, val_interactions, n_items)
        score2, _ = compute_val_ndcg10(model, val_interactions, n_items)
        assert score1 == score2
