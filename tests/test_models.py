"""Tests for recommendation models."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemKNNRecommender, UserKNNRecommender


@pytest.fixture
def sample_interaction_matrix():
    """Create a simple interaction matrix for testing."""
    # 5 users x 10 items
    data = np.array([
        [5, 3, 0, 1, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 3, 4, 5, 0],
        [0, 0, 5, 4, 0, 0, 5, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
    ])
    return csr_matrix(data, dtype=np.float32)


class TestPopularityRecommender:
    """Tests for PopularityRecommender."""

    def test_fit(self, sample_interaction_matrix):
        """Test model fitting."""
        model = PopularityRecommender()
        model.fit(sample_interaction_matrix)

        assert model.is_fitted
        assert model.n_users == 5
        assert model.n_items == 10

    def test_recommend(self, sample_interaction_matrix):
        """Test recommendation generation."""
        model = PopularityRecommender()
        model.fit(sample_interaction_matrix)

        recs = model.recommend(user_idx=0, n=5, exclude_seen=True)

        assert len(recs) == 5
        assert all(isinstance(r, tuple) for r in recs)
        assert all(len(r) == 2 for r in recs)

        # Recommendations should be sorted by score descending
        scores = [score for _, score in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_excludes_seen(self, sample_interaction_matrix):
        """Test that seen items are excluded."""
        model = PopularityRecommender()
        model.fit(sample_interaction_matrix)

        # User 0 has interacted with items 0, 1, 3
        recs = model.recommend(user_idx=0, n=10, exclude_seen=True)
        rec_items = {item_idx for item_idx, _ in recs}

        # Items 0, 1, 3 should not be in recommendations
        assert 0 not in rec_items
        assert 1 not in rec_items
        assert 3 not in rec_items

    def test_not_fitted_error(self):
        """Test error when recommending without fitting."""
        model = PopularityRecommender()

        with pytest.raises(RuntimeError):
            model.recommend(user_idx=0, n=5)


class TestItemKNNRecommender:
    """Tests for ItemKNNRecommender."""

    def test_fit(self, sample_interaction_matrix):
        """Test model fitting."""
        model = ItemKNNRecommender(k=3)
        model.fit(sample_interaction_matrix)

        assert model.is_fitted
        assert model.item_similarity is not None
        assert model.item_similarity.shape == (10, 10)

    def test_recommend(self, sample_interaction_matrix):
        """Test recommendation generation."""
        model = ItemKNNRecommender(k=3)
        model.fit(sample_interaction_matrix)

        recs = model.recommend(user_idx=0, n=5, exclude_seen=True)

        # User 0 has interactions, should get recommendations
        assert len(recs) > 0
        assert all(isinstance(r, tuple) for r in recs)

    def test_similar_items(self, sample_interaction_matrix):
        """Test similar items functionality."""
        model = ItemKNNRecommender(k=3)
        model.fit(sample_interaction_matrix)

        similar = model.get_similar_items(item_idx=0, n=3)

        assert len(similar) <= 3
        # Item 0 should not be in its own similar items
        assert all(item_idx != 0 for item_idx, _ in similar)

    def test_cold_start_user(self, sample_interaction_matrix):
        """Test handling of users with no interactions."""
        # Create matrix with a user that has no interactions
        data = sample_interaction_matrix.toarray()
        data = np.vstack([data, np.zeros(10)])  # Add empty user
        matrix = csr_matrix(data)

        model = ItemKNNRecommender(k=3)
        model.fit(matrix)

        # Cold start user should return empty recommendations
        recs = model.recommend(user_idx=5, n=5)
        assert recs == []


class TestUserKNNRecommender:
    """Tests for UserKNNRecommender."""

    def test_fit(self, sample_interaction_matrix):
        """Test model fitting."""
        model = UserKNNRecommender(k=3)
        model.fit(sample_interaction_matrix)

        assert model.is_fitted
        assert model.user_similarity is not None
        assert model.user_similarity.shape == (5, 5)

    def test_recommend(self, sample_interaction_matrix):
        """Test recommendation generation."""
        model = UserKNNRecommender(k=3)
        model.fit(sample_interaction_matrix)

        recs = model.recommend(user_idx=0, n=5, exclude_seen=True)

        assert len(recs) > 0
        assert all(isinstance(r, tuple) for r in recs)


class TestModelComparison:
    """Tests comparing different models."""

    def test_all_models_produce_recommendations(self, sample_interaction_matrix):
        """Test that all models produce valid recommendations."""
        models = [
            PopularityRecommender(),
            ItemKNNRecommender(k=3),
            UserKNNRecommender(k=3),
        ]

        for model in models:
            model.fit(sample_interaction_matrix)
            recs = model.recommend(user_idx=0, n=5, exclude_seen=True)

            assert len(recs) > 0, f"{model.__class__.__name__} produced no recommendations"
            assert all(
                0 <= item_idx < 10 for item_idx, _ in recs
            ), f"{model.__class__.__name__} produced invalid item indices"
