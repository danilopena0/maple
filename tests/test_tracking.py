"""Tests for ML experiment tracking."""

import tempfile
from pathlib import Path

import pytest

from src.tracking.mlflow_tracker import (
    ExperimentTracker,
    TrackedExperiment,
    MLFLOW_AVAILABLE,
)


class TestTrackedExperiment:
    """Tests for TrackedExperiment dataclass."""

    def test_creation(self):
        """Test basic creation."""
        exp = TrackedExperiment(
            experiment_id="exp1",
            run_id="run1",
            run_name="test_run",
            model_name="test_model",
        )

        assert exp.experiment_id == "exp1"
        assert exp.run_id == "run1"
        assert exp.status == "RUNNING"
        assert exp.parameters == {}
        assert exp.metrics == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        exp = TrackedExperiment(
            experiment_id="exp1",
            run_id="run1",
            run_name="test_run",
            model_name="test_model",
            parameters={"k": 50},
            metrics={"ndcg_10": 0.5},
        )

        d = exp.to_dict()
        assert d["experiment_id"] == "exp1"
        assert d["parameters"]["k"] == 50
        assert d["metrics"]["ndcg_10"] == 0.5


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestExperimentTracker:
    """Tests for ExperimentTracker with MLflow."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a tracker with temp directory."""
        tracking_uri = str(tmp_path / "mlruns")
        return ExperimentTracker(
            experiment_name="test_experiment",
            tracking_uri=tracking_uri,
        )

    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        from src.models.popularity import PopularityRecommender
        return PopularityRecommender()

    @pytest.fixture
    def fitted_model(self, sample_model):
        """Create a fitted sample model."""
        import numpy as np
        from scipy.sparse import csr_matrix

        # Create sample interaction matrix
        data = np.random.rand(10, 5)
        data[data < 0.7] = 0
        matrix = csr_matrix(data)

        sample_model.fit(matrix)
        return sample_model

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker.enabled
        assert tracker.experiment_name == "test_experiment"
        assert tracker.experiment_id is not None

    def test_start_run(self, tracker):
        """Test starting a run."""
        with tracker.start_run(run_name="test_run", model_name="test_model") as run:
            assert run.run_name == "test_run"
            assert run.model_name == "test_model"
            assert run.status == "RUNNING"

        assert run.status == "FINISHED"
        assert run.end_time is not None
        assert run.duration_seconds is not None

    def test_log_params(self, tracker):
        """Test logging parameters."""
        with tracker.start_run(run_name="param_test") as run:
            tracker.log_params({"k": 50, "learning_rate": 0.01})

            assert run.parameters["k"] == 50
            assert run.parameters["learning_rate"] == 0.01

    def test_log_model_params(self, tracker, fitted_model):
        """Test automatic model parameter extraction."""
        with tracker.start_run(run_name="model_param_test") as run:
            tracker.log_model_params(fitted_model)

            assert "model_type" in run.parameters
            assert run.parameters["model_type"] == "PopularityRecommender"

    def test_log_metrics(self, tracker):
        """Test logging metrics."""
        with tracker.start_run(run_name="metric_test") as run:
            tracker.log_metrics({
                "precision_10": 0.15,
                "ndcg_10": 0.45,
                "mrr": 0.32,
            })

            assert run.metrics["precision_10"] == 0.15
            assert run.metrics["ndcg_10"] == 0.45

    def test_log_nested_metrics(self, tracker):
        """Test logging nested metric dictionaries."""
        with tracker.start_run(run_name="nested_metric_test") as run:
            tracker.log_metrics({
                "precision": {"5": 0.1, "10": 0.15, "20": 0.12},
                "coverage": 0.85,
            })

            assert "precision_5" in run.metrics
            assert "precision_10" in run.metrics
            assert run.metrics["coverage"] == 0.85

    def test_log_model_artifact(self, tracker, fitted_model):
        """Test logging model as artifact."""
        with tracker.start_run(run_name="artifact_test") as run:
            uri = tracker.log_model(fitted_model, artifact_name="test_model")

            assert uri is not None
            assert "models/test_model.pkl" in run.artifacts

    def test_set_tags(self, tracker):
        """Test setting tags."""
        with tracker.start_run(run_name="tag_test") as run:
            tracker.set_tag("version", "1.0")
            tracker.set_tags({"env": "test", "team": "ml"})

            assert run.tags["version"] == "1.0"
            assert run.tags["env"] == "test"

    def test_list_runs(self, tracker):
        """Test listing runs."""
        # Create a few runs
        for i in range(3):
            with tracker.start_run(run_name=f"run_{i}") as run:
                tracker.log_metrics({"score": i * 0.1})

        runs = tracker.list_runs()
        assert len(runs) >= 3

    def test_compare_runs(self, tracker):
        """Test comparing runs by metric."""
        # Create runs with different metrics
        for i, score in enumerate([0.3, 0.5, 0.4]):
            with tracker.start_run(run_name=f"compare_run_{i}") as run:
                tracker.log_metrics({"test_score": score})

        comparison = tracker.compare_runs(metric="test_score", top_n=3)
        assert len(comparison) >= 2

        # Best should be first
        if len(comparison) >= 2:
            assert comparison[0]["metrics"]["test_score"] >= comparison[1]["metrics"]["test_score"]

    def test_get_best_run(self, tracker):
        """Test getting best run."""
        # Create runs with different metrics
        for score in [0.3, 0.7, 0.5]:
            with tracker.start_run(run_name=f"best_run_test") as run:
                tracker.log_metrics({"best_metric": score})

        best = tracker.get_best_run(metric="best_metric")
        assert best is not None
        assert best["metrics"]["best_metric"] == 0.7

    def test_load_model(self, tracker, fitted_model):
        """Test saving and loading model."""
        with tracker.start_run(run_name="save_load_test") as run:
            tracker.log_model(fitted_model, artifact_name="saved_model")
            run_id = run.run_id

        # Load the model
        loaded = tracker.load_model(run_id, artifact_name="saved_model")

        assert loaded is not None
        assert loaded.__class__.__name__ == fitted_model.__class__.__name__
        assert loaded.is_fitted

    def test_failed_run(self, tracker):
        """Test that failed runs are marked correctly."""
        try:
            with tracker.start_run(run_name="failing_run") as run:
                tracker.log_params({"test": "value"})
                raise ValueError("Intentional failure")
        except ValueError:
            pass

        assert run.status == "FAILED"


class TestExperimentTrackerDisabled:
    """Tests for tracker when MLflow is not available."""

    def test_disabled_tracker(self):
        """Test tracker works (no-op) when disabled."""
        # Create tracker that pretends MLflow is not available
        tracker = ExperimentTracker(experiment_name="disabled_test")
        tracker.enabled = False

        # All operations should be no-ops
        with tracker.start_run(run_name="disabled_run") as run:
            tracker.log_params({"k": 50})
            tracker.log_metrics({"score": 0.5})

            assert run.run_id == "disabled"

        assert tracker.list_runs() == []
        assert tracker.get_best_run() is None


class TestIntegrationWithModels:
    """Integration tests with actual recommendation models."""

    @pytest.fixture
    def interaction_matrix(self):
        """Create sample interaction matrix."""
        import numpy as np
        from scipy.sparse import csr_matrix

        np.random.seed(42)
        data = np.random.rand(50, 20)
        data[data < 0.8] = 0
        return csr_matrix(data)

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_full_training_workflow(self, interaction_matrix, tmp_path):
        """Test complete training workflow with tracking."""
        from src.models.collaborative import ItemKNNRecommender

        tracker = ExperimentTracker(
            experiment_name="integration_test",
            tracking_uri=str(tmp_path / "mlruns"),
        )

        with tracker.start_run(run_name="item_knn_test", model_name="ItemKNN") as run:
            # Train model
            model = ItemKNNRecommender(k=10)
            model.fit(interaction_matrix)

            # Log params
            tracker.log_model_params(model)

            # Simulate evaluation metrics
            metrics = {
                "precision": {"5": 0.12, "10": 0.10},
                "recall": {"5": 0.08, "10": 0.15},
                "ndcg": {"5": 0.20, "10": 0.25},
                "mrr": 0.35,
                "coverage": 0.80,
            }
            tracker.log_evaluation_results(metrics)

            # Log model
            tracker.log_model(model, "item_knn_model")

        # Verify everything was logged
        assert run.parameters["k"] == 10
        assert run.parameters["model_type"] == "ItemKNNRecommender"
        assert "precision_10" in run.metrics
        assert "item_knn_model" in str(run.artifacts)
