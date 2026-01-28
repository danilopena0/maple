"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    average_precision,
    coverage,
)


class TestPrecisionAtK:
    """Tests for precision@k metric."""

    def test_perfect_precision(self):
        """All recommendations are relevant."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(recommended, relevant, k=5) == 1.0

    def test_zero_precision(self):
        """No recommendations are relevant."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8, 9, 10}
        assert precision_at_k(recommended, relevant, k=5) == 0.0

    def test_partial_precision(self):
        """Some recommendations are relevant."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        assert precision_at_k(recommended, relevant, k=5) == 0.6

    def test_k_smaller_than_recommendations(self):
        """K is smaller than number of recommendations."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        assert precision_at_k(recommended, relevant, k=2) == 1.0

    def test_empty_relevant(self):
        """No relevant items."""
        recommended = [1, 2, 3]
        relevant = set()
        assert precision_at_k(recommended, relevant, k=3) == 0.0


class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_perfect_recall(self):
        """All relevant items are recommended."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=5) == 1.0

    def test_zero_recall(self):
        """No relevant items are recommended."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8}
        assert recall_at_k(recommended, relevant, k=5) == 0.0

    def test_partial_recall(self):
        """Some relevant items are recommended."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 6, 7}
        assert recall_at_k(recommended, relevant, k=5) == 0.5

    def test_empty_relevant(self):
        """No relevant items in ground truth."""
        recommended = [1, 2, 3]
        relevant = set()
        assert recall_at_k(recommended, relevant, k=3) == 0.0


class TestNDCGAtK:
    """Tests for NDCG@k metric."""

    def test_perfect_ranking(self):
        """Perfect ranking of relevant items."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert ndcg_at_k(recommended, relevant, k=5) == 1.0

    def test_worst_ranking(self):
        """Relevant items at the end."""
        recommended = [4, 5, 1, 2, 3]
        relevant = {1, 2, 3}
        result = ndcg_at_k(recommended, relevant, k=5)
        assert 0 < result < 1  # Should be less than perfect

    def test_no_relevant_items(self):
        """No relevant items in recommendations."""
        recommended = [1, 2, 3]
        relevant = {4, 5, 6}
        assert ndcg_at_k(recommended, relevant, k=3) == 0.0

    def test_empty_relevant(self):
        """Empty relevant set."""
        recommended = [1, 2, 3]
        relevant = set()
        assert ndcg_at_k(recommended, relevant, k=3) == 0.0


class TestHitRateAtK:
    """Tests for hit rate@k metric."""

    def test_hit(self):
        """At least one relevant item in top-k."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {3}
        assert hit_rate_at_k(recommended, relevant, k=5) == 1.0

    def test_no_hit(self):
        """No relevant items in top-k."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8}
        assert hit_rate_at_k(recommended, relevant, k=5) == 0.0

    def test_hit_outside_k(self):
        """Relevant item outside top-k."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {5}
        assert hit_rate_at_k(recommended, relevant, k=3) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_first_position(self):
        """Relevant item at first position."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1}
        assert mean_reciprocal_rank(recommended, relevant) == 1.0

    def test_second_position(self):
        """Relevant item at second position."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {2}
        assert mean_reciprocal_rank(recommended, relevant) == 0.5

    def test_third_position(self):
        """Relevant item at third position."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {3}
        assert mean_reciprocal_rank(recommended, relevant) == pytest.approx(1/3)

    def test_no_relevant(self):
        """No relevant items."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8}
        assert mean_reciprocal_rank(recommended, relevant) == 0.0


class TestAveragePrecision:
    """Tests for Average Precision."""

    def test_perfect_ap(self):
        """All relevant items at top."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert average_precision(recommended, relevant) == 1.0

    def test_worst_ap(self):
        """Relevant items at end."""
        recommended = [4, 5, 1, 2, 3]
        relevant = {1, 2, 3}
        result = average_precision(recommended, relevant)
        assert 0 < result < 1

    def test_no_relevant(self):
        """No relevant items in recommendations."""
        recommended = [1, 2, 3]
        relevant = {4, 5, 6}
        assert average_precision(recommended, relevant) == 0.0


class TestCoverage:
    """Tests for catalog coverage."""

    def test_full_coverage(self):
        """All items recommended."""
        all_recs = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
        assert coverage(all_recs, n_items=10) == 1.0

    def test_partial_coverage(self):
        """Some items recommended."""
        all_recs = [[0, 1, 2], [0, 1, 3]]
        assert coverage(all_recs, n_items=10) == 0.4

    def test_empty_recommendations(self):
        """No recommendations."""
        all_recs = []
        assert coverage(all_recs, n_items=10) == 0.0
