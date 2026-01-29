"""Model monitoring for recommendation system."""

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    level: AlertLevel
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "metric": self.metric,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MonitoringReport:
    """Container for monitoring results."""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    metrics: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    drift_detected: bool = False
    performance_degraded: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "metrics": self.metrics,
            "alerts": [a.to_dict() if isinstance(a, Alert) else a for a in self.alerts],
            "drift_detected": self.drift_detected,
            "performance_degraded": self.performance_degraded,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PredictionLogger:
    """
    Logs predictions for monitoring.

    Maintains a rolling window of predictions and their metadata
    for drift detection and performance analysis.
    """

    def __init__(
        self,
        max_size: int = 100000,
        persist_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the prediction logger.

        Args:
            max_size: Maximum number of predictions to keep in memory
            persist_path: Optional path to persist predictions
        """
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None
        self.predictions: deque = deque(maxlen=max_size)

        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    def log(
        self,
        user_id: str,
        recommendations: list[tuple[int, float]],
        model_name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a prediction.

        Args:
            user_id: User identifier
            recommendations: List of (item_idx, score) tuples
            model_name: Name of the model that made the prediction
            metadata: Optional additional metadata
        """
        prediction = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "model_name": model_name,
            "n_recommendations": len(recommendations),
            "scores": [score for _, score in recommendations],
            "item_indices": [idx for idx, _ in recommendations],
            "metadata": metadata or {},
        }

        self.predictions.append(prediction)

    def get_predictions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        model_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Get predictions within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            model_name: Filter by model name

        Returns:
            List of prediction records
        """
        results = []

        for pred in self.predictions:
            pred_time = datetime.fromisoformat(pred["timestamp"])

            if start_time and pred_time < start_time:
                continue
            if end_time and pred_time > end_time:
                continue
            if model_name and pred["model_name"] != model_name:
                continue

            results.append(pred)

        return results

    def get_score_distribution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> np.ndarray:
        """Get distribution of prediction scores."""
        predictions = self.get_predictions(start_time, end_time)
        all_scores = []

        for pred in predictions:
            all_scores.extend(pred["scores"])

        return np.array(all_scores) if all_scores else np.array([])

    def get_item_frequency(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[int, int]:
        """Get frequency of recommended items."""
        predictions = self.get_predictions(start_time, end_time)
        item_counts: dict[int, int] = {}

        for pred in predictions:
            for idx in pred["item_indices"]:
                item_counts[idx] = item_counts.get(idx, 0) + 1

        return item_counts

    def persist(self) -> None:
        """Persist predictions to disk."""
        if not self.persist_path:
            logger.warning("No persist path configured")
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(list(self.predictions), f)

        logger.info(f"Persisted {len(self.predictions)} predictions")

    def _load_from_disk(self) -> None:
        """Load predictions from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        with open(self.persist_path, "r") as f:
            data = json.load(f)

        for pred in data[-self.max_size:]:
            self.predictions.append(pred)

        logger.info(f"Loaded {len(self.predictions)} predictions from disk")


class ModelMonitor:
    """
    Monitor recommendation model performance and drift.

    Features:
    - Prediction drift detection
    - Feature drift monitoring
    - Performance degradation alerts
    - Custom metric tracking
    - Alert generation

    Example:
        monitor = ModelMonitor(
            baseline_metrics={"ndcg@10": 0.45, "coverage": 0.8},
            alert_thresholds={"ndcg@10": 0.1, "coverage": 0.15},
        )

        # Log predictions
        monitor.log_prediction(user_id, recommendations, model_name)

        # Check for issues
        report = monitor.generate_report()
        if report.alerts:
            for alert in report.alerts:
                print(f"{alert.level}: {alert.message}")
    """

    def __init__(
        self,
        baseline_metrics: Optional[dict[str, float]] = None,
        alert_thresholds: Optional[dict[str, float]] = None,
        prediction_logger: Optional[PredictionLogger] = None,
        alert_callbacks: Optional[list[Callable[[Alert], None]]] = None,
    ) -> None:
        """
        Initialize the model monitor.

        Args:
            baseline_metrics: Baseline metrics to compare against
            alert_thresholds: Threshold for each metric (% degradation to alert)
            prediction_logger: Logger for predictions
            alert_callbacks: Functions to call when alerts are generated
        """
        self.baseline_metrics = baseline_metrics or {}
        self.alert_thresholds = alert_thresholds or {}
        self.prediction_logger = prediction_logger or PredictionLogger()
        self.alert_callbacks = alert_callbacks or []

        # Metric history
        self.metric_history: list[dict] = []

        # Reference distributions for drift detection
        self.reference_score_dist: Optional[np.ndarray] = None
        self.reference_item_freq: Optional[dict[int, int]] = None

        logger.info("ModelMonitor initialized")

    def set_baseline(
        self,
        metrics: dict[str, float],
        score_distribution: Optional[np.ndarray] = None,
        item_frequency: Optional[dict[int, int]] = None,
    ) -> None:
        """
        Set baseline for comparison.

        Args:
            metrics: Baseline metric values
            score_distribution: Reference score distribution
            item_frequency: Reference item recommendation frequency
        """
        self.baseline_metrics = metrics
        self.reference_score_dist = score_distribution
        self.reference_item_freq = item_frequency

        logger.info(f"Baseline set with {len(metrics)} metrics")

    def log_prediction(
        self,
        user_id: str,
        recommendations: list[tuple[int, float]],
        model_name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a prediction for monitoring.

        Args:
            user_id: User identifier
            recommendations: List of (item_idx, score) tuples
            model_name: Model that made the prediction
            metadata: Optional additional metadata
        """
        self.prediction_logger.log(user_id, recommendations, model_name, metadata)

    def log_metrics(self, metrics: dict[str, float], timestamp: Optional[datetime] = None) -> list[Alert]:
        """
        Log metrics and check for alerts.

        Args:
            metrics: Dictionary of metric names to values
            timestamp: Optional timestamp (default: now)

        Returns:
            List of alerts generated
        """
        timestamp = timestamp or datetime.now()

        record = {
            "timestamp": timestamp.isoformat(),
            **metrics,
        }
        self.metric_history.append(record)

        # Check for alerts
        alerts = self._check_metric_alerts(metrics)

        # Trigger callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return alerts

    def _check_metric_alerts(self, metrics: dict[str, float]) -> list[Alert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []

        for metric_name, current_value in metrics.items():
            baseline = self.baseline_metrics.get(metric_name)
            threshold = self.alert_thresholds.get(metric_name, 0.1)

            if baseline is None:
                continue

            # Calculate degradation
            if baseline != 0:
                degradation = (baseline - current_value) / baseline
            else:
                degradation = 0 if current_value == 0 else 1.0

            # Check if degradation exceeds threshold
            if degradation > threshold:
                level = AlertLevel.CRITICAL if degradation > threshold * 2 else AlertLevel.WARNING

                alert = Alert(
                    level=level,
                    metric=metric_name,
                    message=f"{metric_name} degraded by {degradation*100:.1f}% "
                            f"(baseline: {baseline:.4f}, current: {current_value:.4f})",
                    value=current_value,
                    threshold=baseline * (1 - threshold),
                )
                alerts.append(alert)

                logger.warning(f"Alert: {alert.message}")

        return alerts

    def check_prediction_drift(
        self,
        window_hours: int = 24,
        threshold: float = 0.1,
    ) -> tuple[bool, dict]:
        """
        Check for drift in prediction distribution.

        Args:
            window_hours: Hours of recent predictions to analyze
            threshold: PSI threshold for drift detection

        Returns:
            Tuple of (has_drift, drift_details)
        """
        if self.reference_score_dist is None:
            return False, {"error": "No reference distribution set"}

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)

        current_scores = self.prediction_logger.get_score_distribution(start_time, end_time)

        if len(current_scores) < 100:
            return False, {"error": "Insufficient predictions for drift detection"}

        # Calculate PSI
        psi = self._calculate_psi(self.reference_score_dist, current_scores)

        drift_details = {
            "psi": psi,
            "threshold": threshold,
            "n_reference": len(self.reference_score_dist),
            "n_current": len(current_scores),
            "reference_mean": float(np.mean(self.reference_score_dist)),
            "current_mean": float(np.mean(current_scores)),
        }

        has_drift = psi > threshold

        if has_drift:
            logger.warning(f"Prediction drift detected: PSI={psi:.4f}")

        return has_drift, drift_details

    def check_coverage_drift(
        self,
        window_hours: int = 24,
        threshold: float = 0.2,
    ) -> tuple[bool, dict]:
        """
        Check for drift in item coverage distribution.

        Args:
            window_hours: Hours of recent predictions to analyze
            threshold: Threshold for coverage drift

        Returns:
            Tuple of (has_drift, drift_details)
        """
        if self.reference_item_freq is None:
            return False, {"error": "No reference item frequency set"}

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)

        current_freq = self.prediction_logger.get_item_frequency(start_time, end_time)

        if not current_freq:
            return False, {"error": "No recent predictions"}

        # Calculate coverage metrics
        ref_items = set(self.reference_item_freq.keys())
        cur_items = set(current_freq.keys())

        ref_coverage = len(ref_items)
        cur_coverage = len(cur_items)

        # Items no longer being recommended
        dropped_items = ref_items - cur_items
        new_items = cur_items - ref_items

        coverage_change = abs(cur_coverage - ref_coverage) / ref_coverage if ref_coverage > 0 else 0

        drift_details = {
            "reference_coverage": ref_coverage,
            "current_coverage": cur_coverage,
            "coverage_change": coverage_change,
            "dropped_items": len(dropped_items),
            "new_items": len(new_items),
        }

        has_drift = coverage_change > threshold

        if has_drift:
            logger.warning(f"Coverage drift detected: {coverage_change*100:.1f}% change")

        return has_drift, drift_details

    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference
        _, bins = np.histogram(reference, bins=n_bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)

        # Avoid division by zero
        ref_props = np.clip(ref_props, 0.0001, 1)
        cur_props = np.clip(cur_props, 0.0001, 1)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(abs(psi))

    def generate_report(
        self,
        window_hours: int = 24,
        include_drift_check: bool = True,
    ) -> MonitoringReport:
        """
        Generate a monitoring report.

        Args:
            window_hours: Hours to analyze
            include_drift_check: Whether to check for drift

        Returns:
            MonitoringReport with metrics, alerts, and drift status
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)

        # Get recent metrics
        recent_metrics = {}
        if self.metric_history:
            # Get the most recent metrics
            latest = self.metric_history[-1]
            recent_metrics = {k: v for k, v in latest.items() if k != "timestamp"}

        # Collect alerts
        alerts = []
        if recent_metrics:
            alerts.extend(self._check_metric_alerts(recent_metrics))

        # Check for drift
        drift_detected = False
        if include_drift_check:
            pred_drift, pred_details = self.check_prediction_drift(window_hours)
            cov_drift, cov_details = self.check_coverage_drift(window_hours)

            drift_detected = pred_drift or cov_drift

            if pred_drift:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    metric="prediction_drift",
                    message=f"Prediction score drift detected (PSI={pred_details['psi']:.4f})",
                    value=pred_details["psi"],
                    threshold=0.1,
                ))

            if cov_drift:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    metric="coverage_drift",
                    message=f"Coverage drift detected ({cov_details['coverage_change']*100:.1f}% change)",
                    value=cov_details["coverage_change"],
                    threshold=0.2,
                ))

        # Check for performance degradation
        performance_degraded = any(
            a.metric in self.baseline_metrics and a.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
            for a in alerts
        )

        # Add prediction statistics
        predictions = self.prediction_logger.get_predictions(start_time, end_time)
        recent_metrics["n_predictions"] = len(predictions)

        if predictions:
            scores = self.prediction_logger.get_score_distribution(start_time, end_time)
            if len(scores) > 0:
                recent_metrics["avg_score"] = float(np.mean(scores))
                recent_metrics["score_std"] = float(np.std(scores))

            item_freq = self.prediction_logger.get_item_frequency(start_time, end_time)
            recent_metrics["unique_items_recommended"] = len(item_freq)

        report = MonitoringReport(
            timestamp=end_time,
            period_start=start_time,
            period_end=end_time,
            metrics=recent_metrics,
            alerts=alerts,
            drift_detected=drift_detected,
            performance_degraded=performance_degraded,
        )

        return report

    def get_metric_history(
        self,
        metric_name: str,
        last_n: Optional[int] = None,
    ) -> list[tuple[datetime, float]]:
        """
        Get history for a specific metric.

        Args:
            metric_name: Name of the metric
            last_n: Optional limit on number of records

        Returns:
            List of (timestamp, value) tuples
        """
        history = []

        for record in self.metric_history:
            if metric_name in record:
                timestamp = datetime.fromisoformat(record["timestamp"])
                history.append((timestamp, record[metric_name]))

        if last_n:
            history = history[-last_n:]

        return history

    def clear_history(self) -> None:
        """Clear metric history."""
        self.metric_history.clear()
        logger.info("Metric history cleared")


def create_alert_handler(log_file: Optional[str] = None) -> Callable[[Alert], None]:
    """
    Create an alert handler that logs alerts.

    Args:
        log_file: Optional file path to write alerts

    Returns:
        Alert handler function
    """
    def handler(alert: Alert) -> None:
        message = f"[{alert.level.value.upper()}] {alert.metric}: {alert.message}"

        if alert.level == AlertLevel.CRITICAL:
            logger.error(message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

        if log_file:
            with open(log_file, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")

    return handler
