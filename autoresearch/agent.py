"""
Autoresearch agent — the outer LLM-driven iteration loop.

Usage:
    python autoresearch/agent.py --n-iterations 20
    python autoresearch/agent.py --n-iterations 5 --no-mlflow

Environment variables:
    ANTHROPIC_API_KEY   required
    MAPLE_DATA_PATH     path to interactions CSV (default: data/sample/interactions.csv)
    MAPLE_VAL_PATH      path to val split pickle (default: autoresearch/data/val_split.pkl)
    MAPLE_TIMEOUT       per-experiment timeout in seconds (default: 300)
"""

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = PROJECT_ROOT / "autoresearch"
TRAIN_PY_PATH = AUTORESEARCH_DIR / "train.py"
PROGRAM_MD_PATH = AUTORESEARCH_DIR / "program.md"
HISTORY_PATH = AUTORESEARCH_DIR / "history.jsonl"
BEST_DIR = AUTORESEARCH_DIR / "best"

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    program_md: str,
    history_summary: str,
    current_best_code: str,
    current_best_score: float,
) -> str:
    return textwrap.dedent(f"""
    You are an expert machine-learning researcher autonomously improving a recommendation system.

    ## Research goal
    {program_md}

    ## Current best result
    Score (val_ndcg10): {current_best_score:.6f}

    Best train.py so far:
    ```python
    {current_best_code}
    ```

    ## Experiment history (most recent last)
    {history_summary}

    ## Your task
    Write a new `train.py` that achieves a higher `val_ndcg10` than {current_best_score:.6f}.

    Think step-by-step:
    1. Review what has been tried and what the results suggest.
    2. Form a clear hypothesis for why your change will help.
    3. Write the new `train.py` that implements your hypothesis.
    4. Include a concise `notes:` line explaining your hypothesis.

    Output ONLY the raw Python source code of the new `train.py`.
    Do NOT wrap it in markdown fences. Do NOT add any explanation outside the code.
    The code must end with a line that prints:  val_ndcg10: <float>
    """).strip()


def summarise_history(records) -> str:
    if not records:
        return "(no history yet)"
    lines = []
    for r in records:
        status = f"{r.val_ndcg10:.6f}" if r.val_ndcg10 > float("-inf") else "FAILED"
        lines.append(
            f"  iter {r.iteration:3d}  {status:>12}  {r.model_name:<30}  {r.notes[:80]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def call_claude(prompt: str, model: str = "claude-opus-4-6") -> str:
    """Call the Anthropic API and return the assistant response text."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run:  pip install 'maple[autoresearch]'"
        )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(response: str) -> str:
    """
    Extract Python source from the LLM response.
    Handles cases where the model wraps the code in markdown fences despite instructions.
    """
    # Strip markdown fences if present
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start)
        return response[start:end].strip()
    if "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start)
        return response[start:end].strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_agent(
    n_iterations: int = 20,
    log_mlflow: bool = True,
    llm_model: str = "claude-opus-4-6",
    delay_between_iters: float = 2.0,
) -> None:
    from src.autoresearch.experiment_store import ExperimentStore
    from autoresearch.harness import run as harness_run, DEFAULT_DATA_PATH, DEFAULT_VAL_PATH

    store = ExperimentStore(str(HISTORY_PATH))
    program_md = PROGRAM_MD_PATH.read_text(encoding="utf-8")

    data_path = os.environ.get("MAPLE_DATA_PATH", DEFAULT_DATA_PATH)
    val_path = os.environ.get("MAPLE_VAL_PATH", DEFAULT_VAL_PATH)

    print(f"\n{'='*60}")
    print(f"  Maple Autoresearch Agent")
    print(f"  Iterations: {n_iterations}   Model: {llm_model}")
    print(f"{'='*60}\n")

    # Run iteration 0 with the baseline train.py (no LLM call)
    if store.next_iteration() == 0:
        print("Running baseline (iteration 0)...")
        harness_run(
            train_py_path=TRAIN_PY_PATH,
            data_path=data_path,
            val_path=val_path,
            log_mlflow=log_mlflow,
        )

    for i in range(n_iterations):
        print(f"\n[Agent] Calling LLM for iteration {store.next_iteration()}...")

        # Build context
        history = store.get_history(last_n=15)
        best_record = store.get_best()
        current_best_score = best_record.val_ndcg10 if best_record else 0.0

        # Load best train.py
        best_train_py = TRAIN_PY_PATH.read_text(encoding="utf-8")
        if (BEST_DIR / "train.py").exists():
            best_train_py = (BEST_DIR / "train.py").read_text(encoding="utf-8")

        history_summary = summarise_history(history)
        prompt = build_prompt(program_md, history_summary, best_train_py, current_best_score)

        # Call LLM
        try:
            response = call_claude(prompt, model=llm_model)
        except Exception as e:
            print(f"[Agent] LLM call failed: {e}")
            time.sleep(delay_between_iters)
            continue

        # Extract and write new train.py
        new_code = extract_code(response)
        if not new_code:
            print("[Agent] Empty response from LLM, skipping iteration.")
            continue

        TRAIN_PY_PATH.write_text(new_code, encoding="utf-8")
        print(f"[Agent] Wrote new train.py ({len(new_code)} chars)")

        # Run harness
        score = harness_run(
            train_py_path=TRAIN_PY_PATH,
            data_path=data_path,
            val_path=val_path,
            log_mlflow=log_mlflow,
        )

        if score > current_best_score:
            print(f"[Agent] New best: {score:.6f} (was {current_best_score:.6f})")
        else:
            print(f"[Agent] No improvement: {score:.6f} <= {current_best_score:.6f}")

        time.sleep(delay_between_iters)

    # Final summary
    print(f"\n{'='*60}")
    print("  Autoresearch complete")
    best = store.get_best()
    if best:
        print(f"  Best val_ndcg10 : {best.val_ndcg10:.6f}")
        print(f"  Best model      : {best.model_name}")
        print(f"  Best iteration  : {best.iteration}")
        print(f"  Notes           : {best.notes}")
        print(f"  Best train.py   : {BEST_DIR}/train.py")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maple autoresearch agent")
    parser.add_argument(
        "--n-iterations", type=int, default=20,
        help="Number of LLM-driven iterations (default: 20)"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging"
    )
    parser.add_argument(
        "--llm-model", default="claude-opus-4-6",
        help="Anthropic model to use (default: claude-opus-4-6)"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between iterations (default: 2)"
    )
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    run_agent(
        n_iterations=args.n_iterations,
        log_mlflow=not args.no_mlflow,
        llm_model=args.llm_model,
        delay_between_iters=args.delay,
    )
