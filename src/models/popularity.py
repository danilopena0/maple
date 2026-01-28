"""Popularity-based recommendation model."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    Popularity-based recommender.

    Recommends items based on global popularity (interaction count).
    Simple but effective baseline that handles cold-start well.
    """

    def __init__(self, name: str = "popularity") -> None:
        super().__init__(name=name)
        self.item_popularity: np.ndarray = np.array([])
        self.popularity_ranking: np.ndarray = np.array([])
        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "PopularityRecommender":
        """
        Compute item popularity from interaction matrix.

        Args:
            interaction_matrix: User-item interaction matrix

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        # Sum interactions per item (column-wise sum)
        self.item_popularity = np.asarray(interaction_matrix.sum(axis=0)).flatten()

        # Pre-compute popularity ranking (descending)
        self.popularity_ranking = np.argsort(-self.item_popularity)

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Recommend popular items to a user.

        Args:
            user_idx: User index (only used for excluding seen items)
            n: Number of recommendations
            exclude_seen: Whether to exclude items user has seen
            seen_items: Specific items to exclude

        Returns:
            List of (item_idx, popularity_score) tuples
        """
        self._check_fitted()

        # Determine items to exclude
        if seen_items is None and exclude_seen and self._interaction_matrix is not None:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        # Get recommendations from pre-computed ranking
        recommendations = []
        for item_idx in self.popularity_ranking:
            if item_idx not in seen_items:
                score = float(self.item_popularity[item_idx])
                recommendations.append((int(item_idx), score))
                if len(recommendations) >= n:
                    break

        return recommendations

    def get_popularity_scores(self) -> np.ndarray:
        """Return raw popularity scores for all items."""
        self._check_fitted()
        return self.item_popularity.copy()


class TimeDecayPopularityRecommender(BaseRecommender):
    """
    Time-weighted popularity recommender.

    Recent interactions are weighted more heavily than older ones.
    """

    def __init__(
        self,
        name: str = "time_decay_popularity",
        decay_factor: float = 0.1,
    ) -> None:
        super().__init__(name=name)
        self.decay_factor = decay_factor
        self.item_scores: np.ndarray = np.array([])
        self.score_ranking: np.ndarray = np.array([])
        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        timestamps: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "TimeDecayPopularityRecommender":
        """
        Compute time-weighted popularity.

        Args:
            interaction_matrix: User-item interaction matrix
            timestamps: Array of timestamps for each interaction

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        if timestamps is not None:
            # Apply exponential time decay
            max_time = timestamps.max()
            time_diffs = max_time - timestamps
            weights = np.exp(-self.decay_factor * time_diffs)

            # Weight the interactions
            weighted_matrix = interaction_matrix.multiply(weights.reshape(-1, 1))
            self.item_scores = np.asarray(weighted_matrix.sum(axis=0)).flatten()
        else:
            # Fall back to simple popularity
            self.item_scores = np.asarray(interaction_matrix.sum(axis=0)).flatten()

        self.score_ranking = np.argsort(-self.item_scores)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """Generate time-decay weighted recommendations."""
        self._check_fitted()

        if seen_items is None and exclude_seen and self._interaction_matrix is not None:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        recommendations = []
        for item_idx in self.score_ranking:
            if item_idx not in seen_items:
                score = float(self.item_scores[item_idx])
                recommendations.append((int(item_idx), score))
                if len(recommendations) >= n:
                    break

        return recommendations
