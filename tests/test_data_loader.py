"""Tests for data loading and preprocessing."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.data.loader import DataLoader


@pytest.fixture
def sample_interactions_df():
    """Create sample interaction DataFrame."""
    base_time = datetime(2024, 1, 1)
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u3"],
        "product_id": ["p1", "p2", "p1", "p3", "p2", "p3", "p4"],
        "interaction_type": ["view", "purchase", "click", "view", "purchase", "click", "view"],
        "timestamp": [
            base_time + timedelta(days=i) for i in range(7)
        ],
    })


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_interactions_from_df(self, sample_interactions_df):
        """Test loading interactions from DataFrame."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        assert loader.interactions_df is not None
        assert len(loader.interactions_df) == 7
        assert loader.n_users == 3
        assert loader.n_products == 4

    def test_id_mappings(self, sample_interactions_df):
        """Test user and product ID mappings."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        # Check user mappings
        assert len(loader.user_id_to_idx) == 3
        assert len(loader.idx_to_user_id) == 3
        for uid, idx in loader.user_id_to_idx.items():
            assert loader.idx_to_user_id[idx] == uid

        # Check product mappings
        assert len(loader.product_id_to_idx) == 4
        assert len(loader.idx_to_product_id) == 4

    def test_interaction_matrix(self, sample_interactions_df):
        """Test interaction matrix creation."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        matrix = loader.get_interaction_matrix(weighted=False)

        assert matrix.shape == (3, 4)
        assert matrix.nnz > 0  # Should have non-zero entries

    def test_weighted_interaction_matrix(self, sample_interactions_df):
        """Test weighted interaction matrix."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        matrix = loader.get_interaction_matrix(weighted=True)

        # Purchases should have higher weight than views
        assert matrix.shape == (3, 4)

    def test_binary_interaction_matrix(self, sample_interactions_df):
        """Test binary interaction matrix."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        matrix = loader.get_interaction_matrix(binary=True)

        # All non-zero values should be 1
        assert all(v == 1.0 for v in matrix.data)

    def test_train_test_split(self, sample_interactions_df):
        """Test time-based train/test split."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        train_df, test_df = loader.train_test_split(test_ratio=0.3, by_time=True)

        assert len(train_df) + len(test_df) == len(loader.interactions_df)
        # Train should have earlier timestamps
        assert train_df["timestamp"].max() <= test_df["timestamp"].min()

    def test_get_user_history(self, sample_interactions_df):
        """Test getting user's interaction history."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        history = loader.get_user_history("u1")
        assert set(history) == {"p1", "p2"}

        history = loader.get_user_history("u3")
        assert set(history) == {"p2", "p3", "p4"}

    def test_get_popular_products(self, sample_interactions_df):
        """Test getting popular products."""
        loader = DataLoader()
        loader.load_interactions(df=sample_interactions_df)

        popular = loader.get_popular_products(n=2)

        assert len(popular) == 2
        # Should be sorted by count descending
        counts = [count for _, count in popular]
        assert counts == sorted(counts, reverse=True)

    def test_missing_required_columns(self):
        """Test error on missing required columns."""
        loader = DataLoader()
        df = pd.DataFrame({
            "user_id": ["u1"],
            # Missing product_id, interaction_type, timestamp
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_interactions(df=df)

    def test_no_data_provided(self):
        """Test error when no data source provided."""
        loader = DataLoader()

        with pytest.raises(ValueError, match="Either filepath or df must be provided"):
            loader.load_interactions()
