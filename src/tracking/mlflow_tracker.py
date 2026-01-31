"""MLflow experiment tracking for recommendation models."""

import json
import os
import pickle
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

from src.models.base import BaseRecommender


@dataclass
class TrackedExperiment:
    """Container for experiment metadata and results."""

    experiment_id: str
    run_id: str
    run_name: str
    model_name: str
    parameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "RUNNING"

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "model_name": self.model_name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
        }


class ExperimentTracker:
    """
    MLflow-based experiment tracker for recommendation models.

    Provides:
    - Automatic parameter logging from model attributes
    - Metric logging with evaluation integration
    - Model artifact storage and versioning
    - Experiment comparison utilities
    - Model registry integration

    Example:
        tracker = ExperimentTracker(experiment_name="als_tuning")

        with tracker.start_run(run_name="als_factors_64") as run:
            # Train model
            model = ALSRecommender(factors=64)
            model.fit(interaction_matrix)

            # Log model parameters automatically
            tracker.log_model_params(model)

            # Evaluate and log metrics
            results = evaluate_model(model, test_df, loader)
            tracker.log_metrics(results)

            # Save model artifact
            tracker.log_model(model, "als_model")

        # Compare runs
        comparison = tracker.compare_runs(metric="ndcg@10")
    """

    def __init__(
        self,
        experiment_name: str = "maple_recommendations",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> None:
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Where to store artifacts (default: with tracking)
        """
        if not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow not installed. Tracking will be disabled. "
                "Install with: pip install mlflow"
            )
            self.enabled = False
            return

        self.enabled = True
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        self.artifact_location = artifact_location

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
            self.experiment = mlflow.get_experiment(self.experiment_id)
        else:
            self.experiment = experiment
            self.experiment_id = experiment.experiment_id
            # Set as active experiment
            mlflow.set_experiment(experiment_name)

        # Initialize client for advanced operations
        self.client = MlflowClient(self.tracking_uri)

        # Current run tracking
        self._current_run: Optional[TrackedExperiment] = None
        self._active_run = None

        logger.info(
            f"Experiment tracker initialized: {experiment_name} "
            f"(id: {self.experiment_id})"
        )

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        model_name: Optional[str] = None,
        tags: Optional[dict] = None,
        nested: bool = False,
    ):
        """
        Start a new tracked run.

        Args:
            run_name: Human-readable name for this run
            model_name: Name of the model being trained
            tags: Additional tags for the run
            nested: Whether this is a nested run

        Yields:
            TrackedExperiment: The current experiment context

        Example:
            with tracker.start_run(run_name="experiment_1") as run:
                # Training code here
                tracker.log_params({"lr": 0.01})
                tracker.log_metrics({"accuracy": 0.95})
        """
        if not self.enabled:
            # Yield a dummy experiment when MLflow is not available
            dummy = TrackedExperiment(
                experiment_id="disabled",
                run_id="disabled",
                run_name=run_name or "disabled",
                model_name=model_name or "unknown",
            )
            yield dummy
            return

        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        # Prepare tags
        run_tags = {
            "model_name": model_name or "unknown",
            "run_name": run_name,
        }
        if tags:
            run_tags.update(tags)

        # Start MLflow run
        self._active_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            tags=run_tags,
            nested=nested,
        )

        # Create tracked experiment
        self._current_run = TrackedExperiment(
            experiment_id=self.experiment_id,
            run_id=self._active_run.info.run_id,
            run_name=run_name,
            model_name=model_name or "unknown",
            tags=run_tags,
            start_time=datetime.now(),
        )

        logger.info(f"Started run: {run_name} (id: {self._current_run.run_id})")

        try:
            yield self._current_run
            self._current_run.status = "FINISHED"
        except Exception as e:
            self._current_run.status = "FAILED"
            mlflow.set_tag("error", str(e))
            raise
        finally:
            self._current_run.end_time = datetime.now()
            mlflow.end_run()

            logger.info(
                f"Ended run: {run_name} "
                f"(duration: {self._current_run.duration_seconds:.2f}s, "
                f"status: {self._current_run.status})"
            )

            self._active_run = None

    def log_params(self, params: dict) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self.enabled or not self._current_run:
            return

        # MLflow has a 500 char limit for param values
        sanitized = {}
        for key, value in params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            sanitized[key] = str_value

        mlflow.log_params(sanitized)
        self._current_run.parameters.update(params)

    def log_model_params(self, model: BaseRecommender) -> None:
        """
        Automatically extract and log parameters from a model.

        Args:
            model: Recommendation model to extract parameters from
        """
        if not self.enabled or not self._current_run:
            return

        params = {"model_type": model.__class__.__name__}

        # Extract common parameters
        param_attrs = [
            "name", "k", "factors", "n_factors", "regularization",
            "iterations", "n_epochs", "learning_rate", "embedding_dim",
            "mlp_layers", "dropout", "batch_size", "n_negatives",
            "min_similarity", "strategy", "cf_weight", "content_weight",
            "diversity_weight", "freshness_weight",
        ]

        for attr in param_attrs:
            if hasattr(model, attr):
                value = getattr(model, attr)
                if value is not None:
                    params[attr] = value

        # Log model shape if fitted
        if hasattr(model, "n_users") and model.n_users is not None:
            params["n_users"] = model.n_users
        if hasattr(model, "n_items") and model.n_items is not None:
            params["n_items"] = model.n_items

        self.log_params(params)
        logger.debug(f"Logged {len(params)} model parameters")

    def log_metrics(
        self,
        metrics: dict,
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
            prefix: Optional prefix for metric names
        """
        if not self.enabled or not self._current_run:
            return

        # Flatten nested dicts and add prefix
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    metric_name = f"{prefix}{key}_{subkey}" if prefix else f"{key}_{subkey}"
                    if isinstance(subvalue, (int, float)):
                        flat_metrics[metric_name] = float(subvalue)
            elif isinstance(value, (int, float)):
                metric_name = f"{prefix}{key}" if prefix else key
                flat_metrics[metric_name] = float(value)

        mlflow.log_metrics(flat_metrics, step=step)
        self._current_run.metrics.update(flat_metrics)

        logger.debug(f"Logged {len(flat_metrics)} metrics")

    def log_evaluation_results(self, results: dict) -> None:
        """
        Log results from evaluate_model().

        Args:
            results: Dictionary from evaluate_model() containing
                     precision, recall, ndcg, mrr, map, coverage
        """
        self.log_metrics(results)

    def log_model(
        self,
        model: BaseRecommender,
        artifact_name: str = "model",
        register_model: bool = False,
        registered_model_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log a model as an artifact.

        Args:
            model: Model to save
            artifact_name: Name for the artifact
            register_model: Whether to register in model registry
            registered_model_name: Name for registered model

        Returns:
            Model URI if successful
        """
        if not self.enabled or not self._current_run:
            return None

        # Save model to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / f"{artifact_name}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Log artifact
            mlflow.log_artifact(str(model_path), artifact_path="models")

            # Also save model metadata
            metadata = {
                "model_type": model.__class__.__name__,
                "name": model.name,
                "is_fitted": model.is_fitted,
                "n_users": getattr(model, "n_users", None),
                "n_items": getattr(model, "n_items", None),
            }
            metadata_path = Path(tmpdir) / f"{artifact_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(str(metadata_path), artifact_path="models")

        self._current_run.artifacts.append(f"models/{artifact_name}.pkl")

        model_uri = f"runs:/{self._current_run.run_id}/models/{artifact_name}.pkl"

        # Register model if requested
        if register_model:
            reg_name = registered_model_name or f"maple_{model.name}"
            try:
                mlflow.register_model(model_uri, reg_name)
                logger.info(f"Registered model: {reg_name}")
            except Exception as e:
                logger.warning(f"Failed to register model: {e}")

        logger.info(f"Logged model artifact: {artifact_name}")
        return model_uri

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log a file or directory as an artifact.

        Args:
            local_path: Path to file or directory
            artifact_path: Subdirectory in artifacts
        """
        if not self.enabled or not self._current_run:
            return

        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        self._current_run.artifacts.append(local_path)

    def log_figure(self, figure, artifact_name: str) -> None:
        """
        Log a matplotlib figure as an artifact.

        Args:
            figure: Matplotlib figure object
            artifact_name: Name for the saved figure
        """
        if not self.enabled or not self._current_run:
            return

        mlflow.log_figure(figure, f"figures/{artifact_name}.png")
        self._current_run.artifacts.append(f"figures/{artifact_name}.png")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        if not self.enabled or not self._current_run:
            return

        mlflow.set_tag(key, value)
        self._current_run.tags[key] = value

    def set_tags(self, tags: dict) -> None:
        """Set multiple tags on the current run."""
        if not self.enabled or not self._current_run:
            return

        mlflow.set_tags(tags)
        self._current_run.tags.update(tags)

    def get_run(self, run_id: str) -> Optional[dict]:
        """
        Get information about a specific run.

        Args:
            run_id: The run ID to retrieve

        Returns:
            Run information as a dictionary
        """
        if not self.enabled:
            return None

        run = self.client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
        }

    def list_runs(
        self,
        max_results: int = 100,
        filter_string: Optional[str] = None,
        order_by: Optional[list] = None,
    ) -> list[dict]:
        """
        List runs in the experiment.

        Args:
            max_results: Maximum number of runs to return
            filter_string: MLflow filter string (e.g., "params.k > 20")
            order_by: List of columns to order by

        Returns:
            List of run dictionaries
        """
        if not self.enabled:
            return []

        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string or "",
            max_results=max_results,
            order_by=order_by or ["start_time DESC"],
        )

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "params": run.data.params,
                "metrics": run.data.metrics,
            }
            for run in runs
        ]

    def compare_runs(
        self,
        run_ids: Optional[list[str]] = None,
        metric: str = "ndcg_10",
        top_n: int = 10,
    ) -> list[dict]:
        """
        Compare runs by a specific metric.

        Args:
            run_ids: Specific run IDs to compare (default: all runs)
            metric: Metric to compare by
            top_n: Number of top runs to return

        Returns:
            List of runs sorted by the metric
        """
        if not self.enabled:
            return []

        if run_ids:
            runs = [self.get_run(rid) for rid in run_ids]
            runs = [r for r in runs if r is not None]
        else:
            runs = self.list_runs(max_results=100)

        # Filter runs that have the metric
        runs_with_metric = [
            r for r in runs
            if metric in r.get("metrics", {})
        ]

        # Sort by metric (descending)
        sorted_runs = sorted(
            runs_with_metric,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=True,
        )

        return sorted_runs[:top_n]

    def get_best_run(self, metric: str = "ndcg_10") -> Optional[dict]:
        """
        Get the best run by a specific metric.

        Args:
            metric: Metric to optimize

        Returns:
            Best run information
        """
        comparison = self.compare_runs(metric=metric, top_n=1)
        return comparison[0] if comparison else None

    def load_model(self, run_id: str, artifact_name: str = "model") -> Optional[BaseRecommender]:
        """
        Load a model from a run.

        Args:
            run_id: The run ID containing the model
            artifact_name: Name of the model artifact

        Returns:
            Loaded model
        """
        if not self.enabled:
            return None

        artifact_path = self.client.download_artifacts(
            run_id,
            f"models/{artifact_name}.pkl",
        )

        with open(artifact_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Loaded model from run {run_id}")
        return model

    def register_best_model(
        self,
        metric: str = "ndcg_10",
        model_name: str = "maple_best_model",
        artifact_name: str = "model",
    ) -> Optional[str]:
        """
        Register the best model to the model registry.

        Args:
            metric: Metric to use for selecting best
            model_name: Name for the registered model
            artifact_name: Name of the model artifact

        Returns:
            Model version if successful
        """
        if not self.enabled:
            return None

        best_run = self.get_best_run(metric=metric)
        if not best_run:
            logger.warning("No runs found to register")
            return None

        model_uri = f"runs:/{best_run['run_id']}/models/{artifact_name}.pkl"

        try:
            result = mlflow.register_model(model_uri, model_name)
            logger.info(
                f"Registered best model: {model_name} "
                f"(version: {result.version}, {metric}: {best_run['metrics'].get(metric)})"
            )
            return result.version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> bool:
        """
        Transition a model version to a new stage.

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            return False


# Global tracker instance
_default_tracker: Optional[ExperimentTracker] = None


def get_tracker(
    experiment_name: str = "maple_recommendations",
    **kwargs,
) -> ExperimentTracker:
    """
    Get or create the default experiment tracker.

    Args:
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for ExperimentTracker

    Returns:
        ExperimentTracker instance
    """
    global _default_tracker

    if _default_tracker is None or _default_tracker.experiment_name != experiment_name:
        _default_tracker = ExperimentTracker(experiment_name=experiment_name, **kwargs)

    return _default_tracker
