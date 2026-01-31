"""Optuna-based hyperparameter tuning for recommendation models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    MedianPruner = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

from src.models.base import BaseRecommender
from src.evaluation.metrics import evaluate_model

# Optional MLflow integration
try:
    from src.tracking import ExperimentTracker
    TRACKING_AVAILABLE = True
except ImportError:
    ExperimentTracker = None
    TRACKING_AVAILABLE = False


# ============================================================================
# Search Space Definitions
# ============================================================================

SEARCH_SPACES = {
    "PopularityRecommender": {
        # Popularity has no tunable parameters
    },

    "ItemKNNRecommender": {
        "k": {"type": "int", "low": 10, "high": 200, "step": 10},
        "min_similarity": {"type": "float", "low": 0.0, "high": 0.3, "step": 0.05},
    },

    "UserKNNRecommender": {
        "k": {"type": "int", "low": 10, "high": 200, "step": 10},
        "min_similarity": {"type": "float", "low": 0.0, "high": 0.3, "step": 0.05},
    },

    "ALSRecommender": {
        "factors": {"type": "int", "low": 16, "high": 256, "log": True},
        "regularization": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "iterations": {"type": "int", "low": 5, "high": 50, "step": 5},
    },

    "ContentBasedRecommender": {
        "use_tfidf": {"type": "categorical", "choices": [True, False]},
    },

    "TFIDFRecommender": {
        "max_features": {"type": "int", "low": 1000, "high": 10000, "step": 1000},
    },

    "HybridRecommender": {
        "cf_weight": {"type": "float", "low": 0.1, "high": 0.9, "step": 0.1},
        "strategy": {"type": "categorical", "choices": ["weighted", "switching", "cascade"]},
        "cold_start_threshold": {"type": "int", "low": 1, "high": 20, "step": 1},
    },

    "FeatureAugmentedCF": {
        "n_factors": {"type": "int", "low": 16, "high": 128, "log": True},
        "feature_weight": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "regularization": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "n_iterations": {"type": "int", "low": 5, "high": 30, "step": 5},
    },

    "BPRRecommender": {
        "n_factors": {"type": "int", "low": 16, "high": 128, "log": True},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "regularization": {"type": "float", "low": 0.0001, "high": 0.1, "log": True},
        "n_epochs": {"type": "int", "low": 5, "high": 50, "step": 5},
    },

    "NeuralCFRecommender": {
        "embedding_dim": {"type": "int", "low": 8, "high": 64, "log": True},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1},
        "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "n_epochs": {"type": "int", "low": 5, "high": 30, "step": 5},
        "n_negatives": {"type": "int", "low": 1, "high": 10, "step": 1},
    },

    "EnsembleRecommender": {
        "strategy": {"type": "categorical", "choices": ["weighted_average", "rank_average", "voting"]},
    },

    "ReRanker": {
        "diversity_weight": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        "freshness_weight": {"type": "float", "low": 0.0, "high": 0.3, "step": 0.1},
    },
}


@dataclass
class TuningResult:
    """Container for tuning results."""

    best_params: dict
    best_value: float
    best_trial_number: int
    n_trials: int
    optimization_metric: str
    model_class: str
    all_trials: list = field(default_factory=list)
    study_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_trial_number": self.best_trial_number,
            "n_trials": self.n_trials,
            "optimization_metric": self.optimization_metric,
            "model_class": self.model_class,
            "study_name": self.study_name,
        }


def suggest_param(trial, name: str, config: dict) -> Any:
    """
    Suggest a parameter value based on configuration.

    Args:
        trial: Optuna trial object
        name: Parameter name
        config: Parameter configuration dict

    Returns:
        Suggested parameter value
    """
    param_type = config["type"]

    if param_type == "int":
        if config.get("log", False):
            return trial.suggest_int(name, config["low"], config["high"], log=True)
        elif "step" in config:
            return trial.suggest_int(name, config["low"], config["high"], step=config["step"])
        else:
            return trial.suggest_int(name, config["low"], config["high"])

    elif param_type == "float":
        if config.get("log", False):
            return trial.suggest_float(name, config["low"], config["high"], log=True)
        elif "step" in config:
            return trial.suggest_float(name, config["low"], config["high"], step=config["step"])
        else:
            return trial.suggest_float(name, config["low"], config["high"])

    elif param_type == "categorical":
        return trial.suggest_categorical(name, config["choices"])

    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def create_objective(
    model_class: Type[BaseRecommender],
    train_matrix: csr_matrix,
    val_interactions,
    data_loader,
    metric: str = "ndcg@10",
    search_space: Optional[dict] = None,
    fixed_params: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
) -> Callable:
    """
    Create an Optuna objective function for a model.

    Args:
        model_class: Model class to tune
        train_matrix: Training interaction matrix
        val_interactions: Validation interactions DataFrame
        data_loader: DataLoader for evaluation
        metric: Metric to optimize (e.g., "ndcg@10", "precision@5")
        search_space: Custom search space (default: use SEARCH_SPACES)
        fixed_params: Parameters to fix (not tuned)
        fit_kwargs: Additional kwargs for model.fit()

    Returns:
        Optuna objective function

    Example:
        objective = create_objective(
            model_class=ALSRecommender,
            train_matrix=train_matrix,
            val_interactions=val_df,
            data_loader=loader,
            metric="ndcg@10"
        )
        study.optimize(objective, n_trials=50)
    """
    # Get search space
    class_name = model_class.__name__
    if search_space is None:
        search_space = SEARCH_SPACES.get(class_name, {})

    if not search_space:
        raise ValueError(f"No search space defined for {class_name}")

    fixed_params = fixed_params or {}
    fit_kwargs = fit_kwargs or {}

    # Parse metric name
    metric_parts = metric.split("@")
    metric_name = metric_parts[0]
    k_value = int(metric_parts[1]) if len(metric_parts) > 1 else 10

    def objective(trial) -> float:
        # Suggest parameters
        params = {}
        for param_name, config in search_space.items():
            params[param_name] = suggest_param(trial, param_name, config)

        # Add fixed params
        params.update(fixed_params)

        try:
            # Create and fit model
            model = model_class(**params)
            model.fit(train_matrix, **fit_kwargs)

            # Evaluate
            results = evaluate_model(
                model=model,
                test_interactions=val_interactions,
                data_loader=data_loader,
                k_values=[k_value],
            )

            # Extract metric value
            if metric_name in results:
                if isinstance(results[metric_name], dict):
                    # Handle nested metrics like precision@10
                    value = results[metric_name].get(metric, 0.0)
                else:
                    value = results[metric_name]
            else:
                # Try direct key
                value = results.get(metric, 0.0)

            return value

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return 0.0

    return objective


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for recommendation models.

    Features:
    - Bayesian optimization with TPE sampler
    - Early stopping with median pruner
    - MLflow integration for tracking
    - Custom search spaces
    - Parallel optimization support

    Example:
        tuner = HyperparameterTuner(
            experiment_name="als_tuning",
            tracking_enabled=True,
        )

        result = tuner.tune(
            model_class=ALSRecommender,
            train_matrix=train_matrix,
            val_interactions=val_df,
            data_loader=loader,
            metric="ndcg@10",
            n_trials=50,
        )

        print(f"Best params: {result.best_params}")
        print(f"Best NDCG@10: {result.best_value}")
    """

    def __init__(
        self,
        experiment_name: str = "hyperparameter_tuning",
        tracking_enabled: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize the tuner.

        Args:
            experiment_name: Name for the tuning experiment
            tracking_enabled: Whether to log to MLflow
            seed: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Install with: pip install optuna"
            )

        self.experiment_name = experiment_name
        self.seed = seed
        self.tracking_enabled = tracking_enabled and TRACKING_AVAILABLE

        # Initialize tracker if enabled
        self.tracker: Optional[ExperimentTracker] = None
        if self.tracking_enabled:
            self.tracker = ExperimentTracker(experiment_name=experiment_name)

        logger.info(f"HyperparameterTuner initialized: {experiment_name}")

    def tune(
        self,
        model_class: Type[BaseRecommender],
        train_matrix: csr_matrix,
        val_interactions,
        data_loader,
        metric: str = "ndcg@10",
        n_trials: int = 50,
        timeout: Optional[int] = None,
        search_space: Optional[dict] = None,
        fixed_params: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> TuningResult:
        """
        Run hyperparameter tuning.

        Args:
            model_class: Model class to tune
            train_matrix: Training interaction matrix
            val_interactions: Validation interactions DataFrame
            data_loader: DataLoader for evaluation
            metric: Metric to optimize
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            search_space: Custom search space
            fixed_params: Parameters to keep fixed
            fit_kwargs: Additional kwargs for model.fit()
            n_jobs: Number of parallel jobs
            show_progress_bar: Show progress bar

        Returns:
            TuningResult with best parameters and value
        """
        class_name = model_class.__name__
        study_name = f"{self.experiment_name}_{class_name}"

        logger.info(f"Starting tuning for {class_name}")
        logger.info(f"Metric: {metric}, Trials: {n_trials}")

        # Create study
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Create objective
        objective = create_objective(
            model_class=model_class,
            train_matrix=train_matrix,
            val_interactions=val_interactions,
            data_loader=data_loader,
            metric=metric,
            search_space=search_space,
            fixed_params=fixed_params,
            fit_kwargs=fit_kwargs,
        )

        # Wrap objective with tracking if enabled
        if self.tracking_enabled and self.tracker:
            objective = self._wrap_with_tracking(objective, class_name, metric)

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )

        # Collect results
        all_trials = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ]

        result = TuningResult(
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial_number=study.best_trial.number,
            n_trials=len(study.trials),
            optimization_metric=metric,
            model_class=class_name,
            all_trials=all_trials,
            study_name=study_name,
        )

        logger.info(f"Tuning complete. Best {metric}: {result.best_value:.4f}")
        logger.info(f"Best params: {result.best_params}")

        return result

    def _wrap_with_tracking(
        self,
        objective: Callable,
        model_class: str,
        metric: str,
    ) -> Callable:
        """Wrap objective to log each trial to MLflow."""
        tracker = self.tracker

        def tracked_objective(trial) -> float:
            run_name = f"{model_class}_trial_{trial.number}"

            with tracker.start_run(run_name=run_name, model_name=model_class) as run:
                # Log trial params
                tracker.log_params(trial.params)
                tracker.set_tag("trial_number", str(trial.number))

                # Run objective
                value = objective(trial)

                # Log result
                tracker.log_metrics({metric: value})

            return value

        return tracked_objective

    def tune_multiple(
        self,
        model_configs: list[dict],
        train_matrix: csr_matrix,
        val_interactions,
        data_loader,
        metric: str = "ndcg@10",
        n_trials_per_model: int = 30,
    ) -> dict[str, TuningResult]:
        """
        Tune multiple models and compare results.

        Args:
            model_configs: List of dicts with 'class', 'fixed_params', 'fit_kwargs'
            train_matrix: Training interaction matrix
            val_interactions: Validation interactions
            data_loader: DataLoader for evaluation
            metric: Metric to optimize
            n_trials_per_model: Trials per model

        Returns:
            Dict mapping model names to TuningResults
        """
        results = {}

        for config in model_configs:
            model_class = config["class"]
            class_name = model_class.__name__

            logger.info(f"\n{'='*50}")
            logger.info(f"Tuning {class_name}")
            logger.info(f"{'='*50}")

            result = self.tune(
                model_class=model_class,
                train_matrix=train_matrix,
                val_interactions=val_interactions,
                data_loader=data_loader,
                metric=metric,
                n_trials=n_trials_per_model,
                fixed_params=config.get("fixed_params"),
                fit_kwargs=config.get("fit_kwargs"),
            )

            results[class_name] = result

        # Log summary
        logger.info("\n" + "=" * 50)
        logger.info("TUNING SUMMARY")
        logger.info("=" * 50)

        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].best_value,
            reverse=True,
        )

        for i, (name, result) in enumerate(sorted_results, 1):
            logger.info(f"{i}. {name}: {metric}={result.best_value:.4f}")

        return results

    def get_best_model(
        self,
        result: TuningResult,
        model_class: Type[BaseRecommender],
        train_matrix: csr_matrix,
        fit_kwargs: Optional[dict] = None,
    ) -> BaseRecommender:
        """
        Create and fit a model with the best parameters.

        Args:
            result: TuningResult from tune()
            model_class: Model class
            train_matrix: Training data
            fit_kwargs: Additional fit arguments

        Returns:
            Fitted model with best parameters
        """
        fit_kwargs = fit_kwargs or {}

        model = model_class(**result.best_params)
        model.fit(train_matrix, **fit_kwargs)

        logger.info(f"Created best model: {model_class.__name__}")
        logger.info(f"Params: {result.best_params}")

        return model


def quick_tune(
    model_class: Type[BaseRecommender],
    train_matrix: csr_matrix,
    val_interactions,
    data_loader,
    metric: str = "ndcg@10",
    n_trials: int = 20,
) -> tuple[dict, float]:
    """
    Quick tuning function for simple use cases.

    Args:
        model_class: Model class to tune
        train_matrix: Training matrix
        val_interactions: Validation data
        data_loader: DataLoader
        metric: Optimization metric
        n_trials: Number of trials

    Returns:
        Tuple of (best_params, best_value)

    Example:
        best_params, best_score = quick_tune(
            ALSRecommender, train_matrix, val_df, loader
        )
        model = ALSRecommender(**best_params)
    """
    tuner = HyperparameterTuner(tracking_enabled=False)

    result = tuner.tune(
        model_class=model_class,
        train_matrix=train_matrix,
        val_interactions=val_interactions,
        data_loader=data_loader,
        metric=metric,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return result.best_params, result.best_value
