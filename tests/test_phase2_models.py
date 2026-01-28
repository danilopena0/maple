"""Tests for Phase 2 recommendation models."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemKNNRecommender
from src.models.content_based import ContentBasedRecommender, TFIDFRecommender
from src.models.hybrid import HybridRecommender, FeatureAugmentedCF
from src.models.neural import BPRRecommender
from src.models.ensemble import EnsembleRecommender, ReRanker, BusinessRulesFilter


@pytest.fixture
def sample_interaction_matrix():
    """Create a simple interaction matrix for testing."""
    data = np.array([
        [5, 3, 0, 1, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 3, 4, 5, 0],
        [0, 0, 5, 4, 0, 0, 5, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
    ])
    return csr_matrix(data, dtype=np.float32)


@pytest.fixture
def sample_item_features():
    """Create sample item features DataFrame."""
    return pd.DataFrame({
        "name": [f"Product {i}" for i in range(10)],
        "description": [
            "Electronics gadget wireless bluetooth",
            "Electronics phone mobile smartphone",
            "Clothing shirt cotton casual",
            "Clothing pants denim jeans",
            "Home kitchen appliance blender",
            "Home furniture chair wood",
            "Sports fitness equipment weights",
            "Sports outdoor camping tent",
            "Books fiction novel bestseller",
            "Books technical programming python",
        ],
        "category": [
            "Electronics", "Electronics", "Clothing", "Clothing", "Home",
            "Home", "Sports", "Sports", "Books", "Books"
        ],
        "brand": [
            "BrandA", "BrandB", "BrandC", "BrandC", "BrandD",
            "BrandE", "BrandF", "BrandF", "BrandG", "BrandH"
        ],
    })


class TestContentBasedRecommender:
    """Tests for ContentBasedRecommender."""

    def test_fit(self, sample_interaction_matrix, sample_item_features):
        """Test model fitting with features."""
        model = ContentBasedRecommender(
            text_features=["name", "description"],
            categorical_features=["category", "brand"],
        )
        model.fit(sample_interaction_matrix, item_features_df=sample_item_features)

        assert model.is_fitted
        assert model.item_similarity is not None
        assert model.item_similarity.shape == (10, 10)

    def test_recommend(self, sample_interaction_matrix, sample_item_features):
        """Test recommendation generation."""
        model = ContentBasedRecommender(
            text_features=["description"],
            categorical_features=["category"],
        )
        model.fit(sample_interaction_matrix, item_features_df=sample_item_features)

        recs = model.recommend(user_idx=0, n=5, exclude_seen=True)

        assert len(recs) > 0
        assert all(isinstance(r, tuple) for r in recs)

    def test_similar_items(self, sample_interaction_matrix, sample_item_features):
        """Test similar items by content."""
        model = ContentBasedRecommender(
            text_features=["description"],
            categorical_features=["category"],
        )
        model.fit(sample_interaction_matrix, item_features_df=sample_item_features)

        # Items in same category should be similar
        similar = model.get_similar_items(item_idx=0, n=3)  # Electronics item

        assert len(similar) > 0
        # Item 1 is also Electronics, should be similar
        similar_indices = [idx for idx, _ in similar]
        assert 1 in similar_indices


class TestTFIDFRecommender:
    """Tests for TFIDFRecommender."""

    def test_fit(self, sample_interaction_matrix, sample_item_features):
        """Test TF-IDF model fitting."""
        texts = sample_item_features["description"].tolist()

        model = TFIDFRecommender()
        model.fit(sample_interaction_matrix, item_texts=texts)

        assert model.is_fitted
        assert model.item_vectors is not None

    def test_recommend(self, sample_interaction_matrix, sample_item_features):
        """Test TF-IDF recommendations."""
        texts = sample_item_features["description"].tolist()

        model = TFIDFRecommender()
        model.fit(sample_interaction_matrix, item_texts=texts)

        recs = model.recommend(user_idx=0, n=5, exclude_seen=True)
        assert len(recs) > 0


class TestHybridRecommender:
    """Tests for HybridRecommender."""

    def test_weighted_strategy(self, sample_interaction_matrix, sample_item_features):
        """Test weighted combination strategy."""
        # Fit sub-models
        cf_model = ItemKNNRecommender(k=3)
        cf_model.fit(sample_interaction_matrix)

        content_model = ContentBasedRecommender(
            text_features=["description"],
            categorical_features=["category"],
        )
        content_model.fit(sample_interaction_matrix, item_features_df=sample_item_features)

        # Create hybrid
        hybrid = HybridRecommender(
            cf_model=cf_model,
            content_model=content_model,
            strategy="weighted",
            cf_weight=0.6,
            content_weight=0.4,
        )
        hybrid.fit(sample_interaction_matrix)

        recs = hybrid.recommend(user_idx=0, n=5)

        assert len(recs) > 0
        assert hybrid.is_fitted

    def test_switching_strategy(self, sample_interaction_matrix, sample_item_features):
        """Test switching strategy for cold-start."""
        cf_model = PopularityRecommender()
        cf_model.fit(sample_interaction_matrix)

        content_model = ContentBasedRecommender(
            text_features=["description"],
            categorical_features=["category"],
        )
        content_model.fit(sample_interaction_matrix, item_features_df=sample_item_features)

        hybrid = HybridRecommender(
            cf_model=cf_model,
            content_model=content_model,
            strategy="switching",
            cold_start_threshold=3,
        )
        hybrid.fit(sample_interaction_matrix)

        # User 0 has 3 interactions, should use CF
        recs = hybrid.recommend(user_idx=0, n=5)
        assert len(recs) > 0


class TestFeatureAugmentedCF:
    """Tests for FeatureAugmentedCF."""

    def test_fit_and_recommend(self, sample_interaction_matrix):
        """Test feature-augmented CF training and recommendations."""
        # Create simple item features
        item_features = np.random.randn(10, 5).astype(np.float32)

        model = FeatureAugmentedCF(n_factors=8, feature_weight=0.3)
        model.fit(
            sample_interaction_matrix,
            item_features=item_features,
            n_iterations=5,
        )

        assert model.is_fitted
        assert model.user_factors is not None
        assert model.item_factors is not None

        recs = model.recommend(user_idx=0, n=5)
        assert len(recs) > 0


class TestBPRRecommender:
    """Tests for BPRRecommender."""

    def test_fit(self, sample_interaction_matrix):
        """Test BPR model fitting."""
        model = BPRRecommender(n_factors=8, n_epochs=5, n_samples=1000)
        model.fit(sample_interaction_matrix)

        assert model.is_fitted
        assert model.user_factors is not None
        assert model.item_factors is not None

    def test_recommend(self, sample_interaction_matrix):
        """Test BPR recommendations."""
        model = BPRRecommender(n_factors=8, n_epochs=5, n_samples=1000)
        model.fit(sample_interaction_matrix)

        recs = model.recommend(user_idx=0, n=5, exclude_seen=True)

        assert len(recs) > 0
        assert all(isinstance(r, tuple) for r in recs)

    def test_similar_items(self, sample_interaction_matrix):
        """Test BPR similar items."""
        model = BPRRecommender(n_factors=8, n_epochs=5, n_samples=1000)
        model.fit(sample_interaction_matrix)

        similar = model.get_similar_items(item_idx=0, n=3)
        assert len(similar) > 0


class TestEnsembleRecommender:
    """Tests for EnsembleRecommender."""

    def test_weighted_average_ensemble(self, sample_interaction_matrix):
        """Test weighted average ensemble."""
        # Create and fit models
        pop_model = PopularityRecommender()
        pop_model.fit(sample_interaction_matrix)

        knn_model = ItemKNNRecommender(k=3)
        knn_model.fit(sample_interaction_matrix)

        # Create ensemble
        ensemble = EnsembleRecommender(
            models=[pop_model, knn_model],
            weights=[0.4, 0.6],
            strategy="weighted_average",
        )
        ensemble.fit(sample_interaction_matrix)

        recs = ensemble.recommend(user_idx=0, n=5)

        assert len(recs) > 0
        assert ensemble.is_fitted

    def test_rank_average_ensemble(self, sample_interaction_matrix):
        """Test rank average ensemble."""
        pop_model = PopularityRecommender()
        pop_model.fit(sample_interaction_matrix)

        knn_model = ItemKNNRecommender(k=3)
        knn_model.fit(sample_interaction_matrix)

        ensemble = EnsembleRecommender(
            models=[pop_model, knn_model],
            strategy="rank_average",
        )
        ensemble.fit(sample_interaction_matrix)

        recs = ensemble.recommend(user_idx=0, n=5)
        assert len(recs) > 0

    def test_voting_ensemble(self, sample_interaction_matrix):
        """Test voting ensemble."""
        pop_model = PopularityRecommender()
        pop_model.fit(sample_interaction_matrix)

        knn_model = ItemKNNRecommender(k=3)
        knn_model.fit(sample_interaction_matrix)

        ensemble = EnsembleRecommender(
            models=[pop_model, knn_model],
            strategy="voting",
        )
        ensemble.fit(sample_interaction_matrix)

        recs = ensemble.recommend(user_idx=0, n=5)
        assert len(recs) > 0


class TestReRanker:
    """Tests for ReRanker."""

    def test_basic_rerank(self):
        """Test basic re-ranking."""
        recommendations = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]

        reranker = ReRanker()
        result = reranker.rerank(recommendations, n=3)

        assert len(result) == 3
        # Should maintain order without diversity/freshness
        assert result[0][0] == 0

    def test_mmr_diversity_rerank(self):
        """Test MMR diversity re-ranking."""
        recommendations = [(0, 0.9), (1, 0.85), (2, 0.8), (3, 0.75), (4, 0.7)]

        # Create similarity matrix where items 0,1 are similar
        similarity = np.eye(5) * 0
        similarity[0, 1] = similarity[1, 0] = 0.9  # Very similar
        similarity[0, 2] = similarity[2, 0] = 0.1  # Not similar

        reranker = ReRanker(diversity_weight=0.5)
        result = reranker.rerank(recommendations, item_similarity=similarity, n=3)

        assert len(result) == 3
        # Item 1 should be penalized due to similarity to item 0
        result_items = [idx for idx, _ in result]
        # Item 2 might be promoted over item 1 due to diversity
        assert 0 in result_items

    def test_category_diversity(self):
        """Test category-based diversity."""
        recommendations = [(0, 0.9), (1, 0.85), (2, 0.8), (3, 0.75), (4, 0.7)]

        item_features = {
            "categories": {0: "A", 1: "A", 2: "B", 3: "B", 4: "C"}
        }

        reranker = ReRanker(category_diversity=True)
        result = reranker.rerank(recommendations, item_features=item_features, n=3)

        assert len(result) == 3
        # Should have items from different categories
        result_items = [idx for idx, _ in result]
        categories = [item_features["categories"][idx] for idx in result_items]
        # With round-robin, should get diverse categories
        assert len(set(categories)) > 1


class TestBusinessRulesFilter:
    """Tests for BusinessRulesFilter."""

    def test_in_stock_filter(self):
        """Test inventory filtering."""
        recommendations = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
        inventory = {0: 5, 1: 0, 2: 10, 3: 0}  # Items 1, 3 out of stock

        from src.models.ensemble import in_stock_rule

        filter_obj = BusinessRulesFilter()
        filter_obj.add_rule(in_stock_rule(inventory))

        result = filter_obj.filter(recommendations)

        result_items = [idx for idx, _ in result]
        assert 0 in result_items
        assert 2 in result_items
        assert 1 not in result_items
        assert 3 not in result_items

    def test_price_range_filter(self):
        """Test price range filtering."""
        recommendations = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
        prices = {0: 10, 1: 50, 2: 100, 3: 200}

        from src.models.ensemble import price_range_rule

        filter_obj = BusinessRulesFilter()
        filter_obj.add_rule(price_range_rule(prices, min_price=20, max_price=150))

        result = filter_obj.filter(recommendations)

        result_items = [idx for idx, _ in result]
        assert 1 in result_items
        assert 2 in result_items
        assert 0 not in result_items  # Too cheap
        assert 3 not in result_items  # Too expensive

    def test_multiple_rules(self):
        """Test combining multiple business rules."""
        recommendations = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
        inventory = {0: 5, 1: 5, 2: 0, 3: 5}
        prices = {0: 10, 1: 50, 2: 100, 3: 200}

        from src.models.ensemble import in_stock_rule, price_range_rule

        filter_obj = BusinessRulesFilter()
        filter_obj.add_rule(in_stock_rule(inventory))
        filter_obj.add_rule(price_range_rule(prices, min_price=20, max_price=150))

        result = filter_obj.filter(recommendations)

        # Only item 1 passes both rules
        result_items = [idx for idx, _ in result]
        assert result_items == [1]
