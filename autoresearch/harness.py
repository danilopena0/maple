"""
Immutable execution harness for autoresearch.

Responsibilities:
  1. Ensure the val split exists (create it from the data CSV if not).
  2. AST-check train.py for banned imports before executing.
  3. Run train.py as a subprocess with a 5-minute timeout.
  4. Parse val_ndcg10, model name, and notes from stdout.
  5. Log the result to history.jsonl and optionally MLflow.
  6. Snapshot the best train.py to autoresearch/best/.

Do NOT modify this file during an autoresearch session.
"""

import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = PROJECT_ROOT / "autoresearch"
BEST_DIR = AUTORESEARCH_DIR / "best"
HISTORY_PATH = AUTORESEARCH_DIR / "history.jsonl"
TRAIN_PY_PATH = AUTORESEARCH_DIR / "train.py"

# Default data/val paths (override with env vars)
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "sample" / "interactions.csv")
DEFAULT_VAL_PATH = str(AUTORESEARCH_DIR / "data" / "val_split.pkl")

EXPERIMENT_TIMEOUT = int(os.environ.get("MAPLE_TIMEOUT", "300"))  # seconds
STDOUT_MAX_BYTES = 64 * 1024  # 64 KB cap to prevent runaway output


# ---------------------------------------------------------------------------
# Safety: AST-based import check
# ---------------------------------------------------------------------------

def check_imports(train_py_path: Path) -> list[str]:
    """
    Parse train.py with the AST module and return a list of banned imports found.
    Returns an empty list if the file is safe.
    """
    # Add project root to path for the import check
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.autoresearch.bridge import BANNED_IMPORTS

    source = train_py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BANNED_IMPORTS:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BANNED_IMPORTS:
                    violations.append(node.module)
    return violations


# ---------------------------------------------------------------------------
# Val split setup
# ---------------------------------------------------------------------------

def ensure_val_split(data_path: str, val_path: str) -> None:
    if Path(val_path).exists():
        return

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run:  python scripts/generate_sample_data.py\n"
            "Or set MAPLE_DATA_PATH to a valid interactions CSV."
        )

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.autoresearch.metric_adapter import create_val_split
    create_val_split(data_path, val_path)


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def run_train_py(
    train_py_path: Path,
    val_path: str,
) -> tuple[str, str, int, float]:
    """
    Run train.py as a subprocess.

    Returns:
        (stdout, stderr, returncode, elapsed_seconds)
    """
    env = os.environ.copy()
    env["MAPLE_VAL_PATH"] = val_path
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(train_py_path)],
            capture_output=True,
            timeout=EXPERIMENT_TIMEOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.monotonic() - start
        stdout = result.stdout.decode("utf-8", errors="replace")[:STDOUT_MAX_BYTES]
        stderr = result.stderr.decode("utf-8", errors="replace")[:STDOUT_MAX_BYTES]
        return stdout, stderr, result.returncode, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return "", f"TimeoutExpired after {EXPERIMENT_TIMEOUT}s", 1, elapsed


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(stdout: str) -> tuple[float, str, str]:
    """
    Extract val_ndcg10, model name, and notes from subprocess stdout.

    Returns:
        (val_ndcg10, model_name, notes)
    """
    val_ndcg10 = float("-inf")
    model_name = "unknown"
    notes = ""

    for line in stdout.splitlines():
        m = re.match(r"val_ndcg10:\s*([-\d.eE+]+)", line)
        if m:
            try:
                val_ndcg10 = float(m.group(1))
            except ValueError:
                pass

        m = re.match(r"model:\s*(.+)", line)
        if m:
            model_name = m.group(1).strip()

        m = re.match(r"notes:\s*(.+)", line)
        if m:
            notes = m.group(1).strip()

    return val_ndcg10, model_name, notes


# ---------------------------------------------------------------------------
# Best snapshot
# ---------------------------------------------------------------------------

def update_best(
    record,
    train_py_path: Path,
    best_dir: Path,
) -> None:
    """Copy train.py to best/ if this is a new high score."""
    store_sys_path = sys.path[:]
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.autoresearch.experiment_store import ExperimentStore
    sys.path = store_sys_path

    store = ExperimentStore(str(HISTORY_PATH))
    current_best = store.get_best()

    # current_best may be this record we just appended; compare to previous
    all_records = store.get_history(last_n=10_000)
    previous_best_score = max(
        (r.val_ndcg10 for r in all_records if r.iteration < record.iteration),
        default=float("-inf"),
    )

    if record.val_ndcg10 > previous_best_score:
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(train_py_path), str(best_dir / "train.py"))
        # Write a small metadata file alongside
        meta_path = best_dir / "best_meta.txt"
        meta_path.write_text(
            f"iteration: {record.iteration}\n"
            f"val_ndcg10: {record.val_ndcg10:.6f}\n"
            f"model: {record.model_name}\n"
            f"notes: {record.notes}\n"
            f"timestamp: {record.timestamp}\n"
        )
        print(f"  *** New best: {record.val_ndcg10:.6f} (was {previous_best_score:.6f}) ***")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    train_py_path: Path = TRAIN_PY_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    val_path: str = DEFAULT_VAL_PATH,
    log_mlflow: bool = True,
) -> float:
    """
    Execute one autoresearch iteration.

    Returns val_ndcg10 (float("-inf") on failure).
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.autoresearch.bridge import RunRecord
    from src.autoresearch.experiment_store import ExperimentStore

    store = ExperimentStore(str(HISTORY_PATH))
    iteration = store.next_iteration()

    print(f"\n── Iteration {iteration} ──────────────────────────────────────")

    # 1. Ensure val split exists
    data_path = os.environ.get("MAPLE_DATA_PATH", data_path)
    val_path = os.environ.get("MAPLE_VAL_PATH", val_path)
    ensure_val_split(data_path, val_path)

    # 2. Safety check
    violations = check_imports(train_py_path)
    if violations:
        error_msg = f"Banned imports detected: {violations}"
        print(f"  BLOCKED: {error_msg}")
        record = RunRecord(
            iteration=iteration,
            val_ndcg10=float("-inf"),
            model_name="blocked",
            notes="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            train_py_hash=hashlib.sha256(train_py_path.read_bytes()).hexdigest(),
            error=error_msg,
        )
        store.record_run(record)
        return float("-inf")

    train_py_hash = hashlib.sha256(train_py_path.read_bytes()).hexdigest()

    # 3. Run subprocess
    print(f"  Running {train_py_path.name} (timeout={EXPERIMENT_TIMEOUT}s)...")
    stdout, stderr, returncode, elapsed = run_train_py(train_py_path, val_path)

    print(f"  Elapsed: {elapsed:.1f}s  returncode: {returncode}")
    if stdout:
        print("  --- stdout ---")
        for line in stdout.strip().splitlines()[-10:]:  # last 10 lines
            print(f"  {line}")

    # 4. Parse output
    val_ndcg10, model_name, notes = parse_output(stdout)

    error = ""
    if returncode != 0:
        error = stderr[:500] if stderr else "non-zero exit code"
        val_ndcg10 = float("-inf")
        print(f"  FAILED: {error[:200]}")

    # 5. Record
    record = RunRecord(
        iteration=iteration,
        val_ndcg10=val_ndcg10,
        model_name=model_name,
        notes=notes,
        timestamp=datetime.now(timezone.utc).isoformat(),
        train_py_hash=train_py_hash,
        duration_seconds=elapsed,
        error=error,
    )
    store.record_run(record)

    # 6. Update best snapshot
    if val_ndcg10 > float("-inf"):
        update_best(record, train_py_path, BEST_DIR)

    # 7. Optional MLflow
    if log_mlflow and val_ndcg10 > float("-inf"):
        store.log_to_mlflow(record, str(train_py_path))

    print(f"  val_ndcg10 = {val_ndcg10:.6f}")
    return val_ndcg10


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one autoresearch iteration")
    parser.add_argument("--train-py", default=str(TRAIN_PY_PATH))
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    score = run(
        train_py_path=Path(args.train_py),
        data_path=args.data_path,
        val_path=args.val_path,
        log_mlflow=not args.no_mlflow,
    )
    sys.exit(0 if score > float("-inf") else 1)
