"""Tests for model monitoring."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.monitor import (
    ModelMonitor,
    MonitoringReport,
    PredictionLogger,
    Alert,
    AlertLevel,
    create_alert_handler,
)


class TestAlert:
    """Tests for Alert class."""

    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            level=AlertLevel.WARNING,
            metric="ndcg@10",
            message="Metric degraded",
            value=0.35,
            threshold=0.40,
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.metric == "ndcg@10"
        assert alert.value == 0.35

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            metric="coverage",
            message="Coverage dropped",
            value=0.5,
            threshold=0.7,
        )

        d = alert.to_dict()

        assert d["level"] == "critical"
        assert d["metric"] == "coverage"
        assert "timestamp" in d


class TestMonitoringReport:
    """Tests for MonitoringReport class."""

    def test_report_creation(self):
        """Test report creation."""
        now = datetime.now()
        report = MonitoringReport(
            timestamp=now,
            period_start=now - timedelta(hours=24),
            period_end=now,
            metrics={"ndcg@10": 0.45},
            alerts=[],
        )

        assert report.metrics["ndcg@10"] == 0.45
        assert not report.drift_detected

    def test_report_to_dict(self):
        """Test report serialization."""
        now = datetime.now()
        report = MonitoringReport(
            timestamp=now,
            period_start=now - timedelta(hours=24),
            period_end=now,
        )

        d = report.to_dict()

        assert "timestamp" in d
        assert "metrics" in d
        assert "alerts" in d

    def test_report_to_json(self):
        """Test JSON serialization."""
        now = datetime.now()
        report = MonitoringReport(
            timestamp=now,
            period_start=now - timedelta(hours=24),
            period_end=now,
        )

        json_str = report.to_json()

        assert isinstance(json_str, str)
        assert "timestamp" in json_str


class TestPredictionLogger:
    """Tests for PredictionLogger class."""

    def test_logger_creation(self):
        """Test logger initialization."""
        logger = PredictionLogger(max_size=1000)

        assert logger.max_size == 1000
        assert len(logger.predictions) == 0

    def test_log_prediction(self):
        """Test logging predictions."""
        logger = PredictionLogger()

        recommendations = [(0, 0.9), (1, 0.8), (2, 0.7)]
        logger.log("user_001", recommendations, "test_model")

        assert len(logger.predictions) == 1
        assert logger.predictions[0]["user_id"] == "user_001"
        assert logger.predictions[0]["n_recommendations"] == 3

    def test_get_predictions(self):
        """Test getting predictions."""
        logger = PredictionLogger()

        # Log multiple predictions
        for i in range(5):
            logger.log(f"user_{i}", [(i, 0.5)], "model_a")

        predictions = logger.get_predictions()

        assert len(predictions) == 5

    def test_get_predictions_by_model(self):
        """Test filtering by model name."""
        logger = PredictionLogger()

        logger.log("user_1", [(0, 0.5)], "model_a")
        logger.log("user_2", [(1, 0.5)], "model_b")
        logger.log("user_3", [(2, 0.5)], "model_a")

        predictions = logger.get_predictions(model_name="model_a")

        assert len(predictions) == 2

    def test_get_score_distribution(self):
        """Test getting score distribution."""
        logger = PredictionLogger()

        logger.log("user_1", [(0, 0.9), (1, 0.8)], "model")
        logger.log("user_2", [(0, 0.7), (1, 0.6)], "model")

        scores = logger.get_score_distribution()

        assert len(scores) == 4
        assert 0.6 in scores
        assert 0.9 in scores

    def test_get_item_frequency(self):
        """Test getting item frequency."""
        logger = PredictionLogger()

        logger.log("user_1", [(0, 0.9), (1, 0.8)], "model")
        logger.log("user_2", [(0, 0.7), (2, 0.6)], "model")

        freq = logger.get_item_frequency()

        assert freq[0] == 2  # Item 0 recommended twice
        assert freq[1] == 1
        assert freq[2] == 1

    def test_max_size_limit(self):
        """Test that max size is respected."""
        logger = PredictionLogger(max_size=10)

        for i in range(20):
            logger.log(f"user_{i}", [(0, 0.5)], "model")

        assert len(logger.predictions) == 10

    def test_persist_and_load(self):
        """Test persisting and loading predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.json"
            logger = PredictionLogger(persist_path=str(path))

            logger.log("user_1", [(0, 0.5)], "model")
            logger.log("user_2", [(1, 0.6)], "model")
            logger.persist()

            # Create new logger and load
            new_logger = PredictionLogger(persist_path=str(path))

            assert len(new_logger.predictions) == 2


class TestModelMonitor:
    """Tests for ModelMonitor class."""

    def test_monitor_creation(self):
        """Test monitor initialization."""
        monitor = ModelMonitor(
            baseline_metrics={"ndcg@10": 0.45},
            alert_thresholds={"ndcg@10": 0.1},
        )

        assert monitor.baseline_metrics["ndcg@10"] == 0.45
        assert monitor.alert_thresholds["ndcg@10"] == 0.1

    def test_set_baseline(self):
        """Test setting baseline."""
        monitor = ModelMonitor()

        monitor.set_baseline(
            metrics={"ndcg@10": 0.5, "coverage": 0.8},
            score_distribution=np.random.normal(0.5, 0.1, 1000),
        )

        assert monitor.baseline_metrics["ndcg@10"] == 0.5
        assert monitor.reference_score_dist is not None

    def test_log_prediction(self):
        """Test logging predictions through monitor."""
        monitor = ModelMonitor()

        monitor.log_prediction("user_1", [(0, 0.9), (1, 0.8)], "test_model")

        predictions = monitor.prediction_logger.get_predictions()
        assert len(predictions) == 1

    def test_log_metrics(self):
        """Test logging metrics."""
        monitor = ModelMonitor(
            baseline_metrics={"ndcg@10": 0.5},
            alert_thresholds={"ndcg@10": 0.1},
        )

        alerts = monitor.log_metrics({"ndcg@10": 0.48})

        assert len(monitor.metric_history) == 1
        assert len(alerts) == 0  # No alert, degradation is small

    def test_log_metrics_with_alert(self):
        """Test that alerts are generated for degraded metrics."""
        monitor = ModelMonitor(
            baseline_metrics={"ndcg@10": 0.5},
            alert_thresholds={"ndcg@10": 0.1},
        )

        # Log metrics with significant degradation
        alerts = monitor.log_metrics({"ndcg@10": 0.35})  # 30% degradation

        assert len(alerts) == 1
        assert alerts[0].metric == "ndcg@10"
        assert alerts[0].level in [AlertLevel.WARNING, AlertLevel.CRITICAL]

    def test_alert_callback(self):
        """Test alert callbacks are called."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        monitor = ModelMonitor(
            baseline_metrics={"ndcg@10": 0.5},
            alert_thresholds={"ndcg@10": 0.1},
            alert_callbacks=[callback],
        )

        monitor.log_metrics({"ndcg@10": 0.35})

        assert len(alerts_received) == 1

    def test_check_prediction_drift_no_reference(self):
        """Test drift check without reference distribution."""
        monitor = ModelMonitor()

        has_drift, details = monitor.check_prediction_drift()

        assert has_drift is False
        assert "error" in details

    def test_check_prediction_drift(self):
        """Test drift detection."""
        monitor = ModelMonitor()

        # Set reference distribution
        np.random.seed(42)
        reference = np.random.normal(0.5, 0.1, 1000)
        monitor.reference_score_dist = reference

        # Log predictions from same distribution (no drift)
        for i in range(200):
            score = np.random.normal(0.5, 0.1)
            monitor.log_prediction(f"user_{i}", [(0, score)], "model")

        has_drift, details = monitor.check_prediction_drift()

        assert "psi" in details
        # Should have low drift with same distribution
        assert details["psi"] < 0.5

    def test_check_prediction_drift_with_drift(self):
        """Test drift detection when drift exists."""
        monitor = ModelMonitor()

        # Set reference distribution (mean 0.5)
        np.random.seed(42)
        reference = np.random.normal(0.5, 0.1, 1000)
        monitor.reference_score_dist = reference

        # Log predictions from different distribution (mean 0.8)
        for i in range(200):
            score = np.random.normal(0.8, 0.1)  # Different mean
            monitor.log_prediction(f"user_{i}", [(0, score)], "model")

        has_drift, details = monitor.check_prediction_drift(threshold=0.1)

        assert has_drift is True
        assert details["psi"] > 0.1

    def test_check_coverage_drift(self):
        """Test coverage drift detection."""
        monitor = ModelMonitor()

        # Set reference item frequency
        monitor.reference_item_freq = {i: 10 for i in range(100)}

        # Log predictions with similar coverage
        for i in range(100):
            item_idx = i % 100
            monitor.log_prediction(f"user_{i}", [(item_idx, 0.5)], "model")

        has_drift, details = monitor.check_coverage_drift()

        assert "reference_coverage" in details
        assert "current_coverage" in details

    def test_generate_report(self):
        """Test report generation."""
        monitor = ModelMonitor(
            baseline_metrics={"ndcg@10": 0.5},
        )

        # Log some predictions
        for i in range(10):
            monitor.log_prediction(f"user_{i}", [(0, 0.5)], "model")

        # Log metrics
        monitor.log_metrics({"ndcg@10": 0.48})

        report = monitor.generate_report(include_drift_check=False)

        assert isinstance(report, MonitoringReport)
        assert "n_predictions" in report.metrics

    def test_get_metric_history(self):
        """Test getting metric history."""
        monitor = ModelMonitor()

        monitor.log_metrics({"ndcg@10": 0.45})
        monitor.log_metrics({"ndcg@10": 0.46})
        monitor.log_metrics({"ndcg@10": 0.47})

        history = monitor.get_metric_history("ndcg@10")

        assert len(history) == 3
        assert history[0][1] == 0.45
        assert history[2][1] == 0.47

    def test_get_metric_history_with_limit(self):
        """Test getting limited metric history."""
        monitor = ModelMonitor()

        for i in range(10):
            monitor.log_metrics({"ndcg@10": 0.4 + i * 0.01})

        history = monitor.get_metric_history("ndcg@10", last_n=3)

        assert len(history) == 3

    def test_clear_history(self):
        """Test clearing history."""
        monitor = ModelMonitor()

        monitor.log_metrics({"ndcg@10": 0.45})
        monitor.clear_history()

        assert len(monitor.metric_history) == 0


class TestCreateAlertHandler:
    """Tests for alert handler creation."""

    def test_create_handler(self):
        """Test creating an alert handler."""
        handler = create_alert_handler()

        assert callable(handler)

    def test_handler_logs_alert(self):
        """Test that handler processes alerts."""
        handler = create_alert_handler()

        alert = Alert(
            level=AlertLevel.WARNING,
            metric="test",
            message="Test alert",
            value=0.5,
            threshold=0.6,
        )

        # Should not raise
        handler(alert)

    def test_handler_writes_to_file(self):
        """Test that handler writes to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        handler = create_alert_handler(log_file=log_path)

        alert = Alert(
            level=AlertLevel.WARNING,
            metric="test",
            message="Test alert",
            value=0.5,
            threshold=0.6,
        )

        handler(alert)

        # Check file was written
        with open(log_path, "r") as f:
            content = f.read()

        assert "test" in content
        assert "warning" in content.lower()

        # Cleanup
        Path(log_path).unlink()


class TestIntegration:
    """Integration tests for monitoring."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        # Setup
        np.random.seed(42)

        monitor = ModelMonitor(
            baseline_metrics={"ndcg@10": 0.50, "coverage": 0.80},
            alert_thresholds={"ndcg@10": 0.15, "coverage": 0.20},
        )

        # Set baseline distributions
        monitor.set_baseline(
            metrics={"ndcg@10": 0.50, "coverage": 0.80},
            score_distribution=np.random.normal(0.5, 0.1, 1000),
            item_frequency={i: 10 for i in range(100)},
        )

        # Simulate normal operation
        for i in range(50):
            score = np.random.normal(0.5, 0.1)
            monitor.log_prediction(f"user_{i}", [(i % 100, score)], "model")

        # Log good metrics
        alerts = monitor.log_metrics({"ndcg@10": 0.48, "coverage": 0.78})
        assert len(alerts) == 0  # No alerts for small degradation

        # Simulate degradation
        alerts = monitor.log_metrics({"ndcg@10": 0.35, "coverage": 0.55})
        assert len(alerts) > 0  # Should have alerts

        # Generate report
        report = monitor.generate_report(window_hours=1)

        assert isinstance(report, MonitoringReport)
        assert report.performance_degraded is True
        assert len(report.alerts) > 0
