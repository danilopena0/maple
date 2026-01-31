"""Tests for hyperparameter tuning."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.tuning.optuna_tuner import (
    HyperparameterTuner,
    TuningResult,
    create_objective,
    quick_tune,
    suggest_param,
    SEARCH_SPACES,
    OPTUNA_AVAILABLE,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)

    # Create interaction matrix
    n_users, n_items = 50, 30
    data = np.random.rand(n_users, n_items)
    data[data < 0.85] = 0
    interaction_matrix = csr_matrix(data)

    # Create interactions DataFrame with timestamp
    interactions = []
    base_time = pd.Timestamp("2024-01-01")
    idx = 0
    for user_idx in range(n_users):
        for item_idx in range(n_items):
            if data[user_idx, item_idx] > 0:
                interactions.append({
                    "user_id": f"user_{user_idx:03d}",
                    "product_id": f"prod_{item_idx:03d}",
                    "interaction_type": "purchase",
                    "timestamp": base_time + pd.Timedelta(hours=idx),
                })
                idx += 1

    interactions_df = pd.DataFrame(interactions)

    # Create data loader
    from src.data.loader import DataLoader
    loader = DataLoader()
    loader.load_interactions(df=interactions_df)

    return {
        "train_matrix": interaction_matrix,
        "interactions_df": interactions_df,
        "loader": loader,
    }


class TestTuningResult:
    """Tests for TuningResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = TuningResult(
            best_params={"k": 50},
            best_value=0.45,
            best_trial_number=5,
            n_trials=10,
            optimization_metric="ndcg@10",
            model_class="ItemKNNRecommender",
        )

        assert result.best_params == {"k": 50}
        assert result.best_value == 0.45
        assert result.best_trial_number == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TuningResult(
            best_params={"k": 50},
            best_value=0.45,
            best_trial_number=5,
            n_trials=10,
            optimization_metric="ndcg@10",
            model_class="ItemKNNRecommender",
        )

        d = result.to_dict()
        assert d["best_params"]["k"] == 50
        assert d["best_value"] == 0.45
        assert d["model_class"] == "ItemKNNRecommender"


class TestSearchSpaces:
    """Tests for search space definitions."""

    def test_search_spaces_defined(self):
        """Test that search spaces are defined for models."""
        expected_models = [
            "ItemKNNRecommender",
            "UserKNNRecommender",
            "ALSRecommender",
            "BPRRecommender",
        ]

        for model in expected_models:
            assert model in SEARCH_SPACES
            assert len(SEARCH_SPACES[model]) > 0

    def test_search_space_format(self):
        """Test search space configuration format."""
        for model_name, space in SEARCH_SPACES.items():
            for param_name, config in space.items():
                assert "type" in config, f"Missing type for {model_name}.{param_name}"

                if config["type"] == "int":
                    assert "low" in config
                    assert "high" in config
                elif config["type"] == "float":
                    assert "low" in config
                    assert "high" in config
                elif config["type"] == "categorical":
                    assert "choices" in config


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestSuggestParam:
    """Tests for parameter suggestion."""

    def test_suggest_int(self):
        """Test integer parameter suggestion."""
        import optuna

        def objective(trial):
            value = suggest_param(trial, "k", {"type": "int", "low": 10, "high": 100})
            assert isinstance(value, int)
            assert 10 <= value <= 100
            return value

        study = optuna.create_study()
        study.optimize(objective, n_trials=5, show_progress_bar=False)

    def test_suggest_int_with_step(self):
        """Test integer with step."""
        import optuna

        def objective(trial):
            value = suggest_param(trial, "k", {"type": "int", "low": 10, "high": 100, "step": 10})
            assert value % 10 == 0
            return value

        study = optuna.create_study()
        study.optimize(objective, n_trials=5, show_progress_bar=False)

    def test_suggest_float(self):
        """Test float parameter suggestion."""
        import optuna

        def objective(trial):
            value = suggest_param(trial, "lr", {"type": "float", "low": 0.001, "high": 0.1})
            assert isinstance(value, float)
            assert 0.001 <= value <= 0.1
            return value

        study = optuna.create_study()
        study.optimize(objective, n_trials=5, show_progress_bar=False)

    def test_suggest_categorical(self):
        """Test categorical parameter suggestion."""
        import optuna

        choices = ["a", "b", "c"]

        def objective(trial):
            value = suggest_param(trial, "strategy", {"type": "categorical", "choices": choices})
            assert value in choices
            return 1.0

        study = optuna.create_study()
        study.optimize(objective, n_trials=5, show_progress_bar=False)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestCreateObjective:
    """Tests for objective function creation."""

    def test_create_objective_item_knn(self, sample_data):
        """Test creating objective for ItemKNN."""
        from src.models.collaborative import ItemKNNRecommender

        objective = create_objective(
            model_class=ItemKNNRecommender,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
        )

        assert callable(objective)

    def test_create_objective_custom_space(self, sample_data):
        """Test creating objective with custom search space."""
        from src.models.collaborative import ItemKNNRecommender

        custom_space = {
            "k": {"type": "int", "low": 5, "high": 20, "step": 5},
        }

        objective = create_objective(
            model_class=ItemKNNRecommender,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
            search_space=custom_space,
        )

        assert callable(objective)

    def test_objective_returns_float(self, sample_data):
        """Test that objective returns a float."""
        import optuna
        from src.models.collaborative import ItemKNNRecommender

        objective = create_objective(
            model_class=ItemKNNRecommender,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
        )

        study = optuna.create_study()
        study.optimize(objective, n_trials=1, show_progress_bar=False)

        assert study.best_value is not None
        assert isinstance(study.best_value, float)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""

    def test_tuner_initialization(self):
        """Test tuner initializes correctly."""
        tuner = HyperparameterTuner(
            experiment_name="test_tuning",
            tracking_enabled=False,
        )

        assert tuner.experiment_name == "test_tuning"
        assert tuner.tracking_enabled is False

    def test_tune_item_knn(self, sample_data):
        """Test tuning ItemKNN model."""
        from src.models.collaborative import ItemKNNRecommender

        tuner = HyperparameterTuner(
            experiment_name="test_item_knn",
            tracking_enabled=False,
        )

        result = tuner.tune(
            model_class=ItemKNNRecommender,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
            n_trials=3,
            show_progress_bar=False,
        )

        assert isinstance(result, TuningResult)
        assert result.n_trials == 3
        assert "k" in result.best_params
        assert result.best_value >= 0

    def test_tune_als(self, sample_data):
        """Test tuning ALS model."""
        from src.models.collaborative import ALSRecommender

        tuner = HyperparameterTuner(
            experiment_name="test_als",
            tracking_enabled=False,
        )

        # Use smaller search space for faster test
        custom_space = {
            "factors": {"type": "categorical", "choices": [16, 32]},
            "iterations": {"type": "categorical", "choices": [5, 10]},
        }

        result = tuner.tune(
            model_class=ALSRecommender,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
            n_trials=2,
            search_space=custom_space,
            show_progress_bar=False,
        )

        assert result.best_params is not None
        assert "factors" in result.best_params

    def test_get_best_model(self, sample_data):
        """Test creating best model from results."""
        from src.models.collaborative import ItemKNNRecommender

        tuner = HyperparameterTuner(
            experiment_name="test_best_model",
            tracking_enabled=False,
        )

        result = tuner.tune(
            model_class=ItemKNNRecommender,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
            n_trials=2,
            show_progress_bar=False,
        )

        model = tuner.get_best_model(
            result=result,
            model_class=ItemKNNRecommender,
            train_matrix=sample_data["train_matrix"],
        )

        assert model.is_fitted
        assert model.k == result.best_params["k"]


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestQuickTune:
    """Tests for quick_tune function."""

    def test_quick_tune(self, sample_data):
        """Test quick tuning function."""
        from src.models.collaborative import ItemKNNRecommender

        best_params, best_value = quick_tune(
            model_class=ItemKNNRecommender,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
            n_trials=2,
        )

        assert isinstance(best_params, dict)
        assert "k" in best_params
        assert isinstance(best_value, float)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestTuneMultiple:
    """Tests for tuning multiple models."""

    def test_tune_multiple_models(self, sample_data):
        """Test tuning multiple models at once."""
        from src.models.collaborative import ItemKNNRecommender
        from src.models.popularity import PopularityRecommender

        tuner = HyperparameterTuner(
            experiment_name="test_multiple",
            tracking_enabled=False,
        )

        # Note: PopularityRecommender has no tunable params, so we skip it
        # and just test ItemKNN with minimal trials
        model_configs = [
            {
                "class": ItemKNNRecommender,
                "fixed_params": {},
            },
        ]

        results = tuner.tune_multiple(
            model_configs=model_configs,
            train_matrix=sample_data["train_matrix"],
            val_interactions=sample_data["interactions_df"],
            data_loader=sample_data["loader"],
            metric="ndcg@10",
            n_trials_per_model=2,
        )

        assert "ItemKNNRecommender" in results
        assert isinstance(results["ItemKNNRecommender"], TuningResult)
