"""Append-only run log (history.jsonl) with optional MLflow integration."""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.autoresearch.bridge import RunRecord


class ExperimentStore:
    """
    Manages the autoresearch run history.

    History is stored as newline-delimited JSON so it can be tailed,
    grepped, and inspected without any special tooling.
    """

    def __init__(self, history_path: str) -> None:
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self.history_path.touch()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_history(self, last_n: int = 20) -> list[RunRecord]:
        """Return the most recent N run records (oldest first)."""
        records = []
        with open(self.history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(RunRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        pass
        return records[-last_n:]

    def get_best(self) -> Optional[RunRecord]:
        """Return the record with the highest val_ndcg10."""
        all_records = self.get_history(last_n=10_000)
        if not all_records:
            return None
        return max(all_records, key=lambda r: r.val_ndcg10)

    def next_iteration(self) -> int:
        """Return the next iteration number (0-indexed)."""
        records = self.get_history(last_n=10_000)
        if not records:
            return 0
        return records[-1].iteration + 1

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record_run(self, record: RunRecord) -> None:
        """Append a run record to history.jsonl."""
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.info(
            f"Recorded iter {record.iteration}: "
            f"val_ndcg10={record.val_ndcg10:.6f}  model={record.model_name}"
        )

    # ------------------------------------------------------------------
    # MLflow integration (optional)
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        record: RunRecord,
        train_py_path: str,
        full_metrics: Optional[dict] = None,
    ) -> None:
        """Log a run record to MLflow if available."""
        try:
            from src.tracking.mlflow_tracker import ExperimentTracker
        except ImportError:
            return

        try:
            tracker = ExperimentTracker(experiment_name="maple_autoresearch")
            run_name = f"autoresearch_iter_{record.iteration}"

            with tracker.start_run(run_name=run_name, model_name=record.model_name):
                tracker.log_params({
                    "iteration": record.iteration,
                    "model_name": record.model_name,
                    "notes": record.notes[:500],
                    "train_py_hash": record.train_py_hash[:16],
                })
                metrics = {"val_ndcg10": record.val_ndcg10}
                if full_metrics:
                    metrics.update(full_metrics)
                tracker.log_metrics(metrics)
                tracker.log_artifact(train_py_path, artifact_path="train_scripts")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")
