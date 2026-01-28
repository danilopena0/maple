"""Base class for recommendation models."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""

    def __init__(self, name: str = "base") -> None:
        self.name = name
        self.is_fitted = False
        self.n_users: int = 0
        self.n_items: int = 0

    @abstractmethod
    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "BaseRecommender":
        """
        Train the model on interaction data.

        Args:
            interaction_matrix: User-item interaction matrix (sparse)
            **kwargs: Additional training parameters

        Returns:
            self
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_seen: Whether to exclude items user has interacted with
            seen_items: Set of item indices to exclude (overrides exclude_seen)

        Returns:
            List of (item_idx, score) tuples, sorted by score descending
        """
        pass

    def recommend_batch(
        self,
        user_indices: list[int],
        n: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Generate recommendations for multiple users.

        Args:
            user_indices: List of user indices
            n: Number of recommendations per user
            exclude_seen: Whether to exclude seen items

        Returns:
            Dict mapping user_idx to list of (item_idx, score) tuples
        """
        return {
            user_idx: self.recommend(user_idx, n, exclude_seen)
            for user_idx in user_indices
        }

    def _check_fitted(self) -> None:
        """Raise error if model hasn't been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before making predictions"
            )

    def _get_seen_items(
        self,
        user_idx: int,
        interaction_matrix: csr_matrix,
    ) -> set[int]:
        """Get items a user has interacted with."""
        return set(interaction_matrix[user_idx].indices)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
