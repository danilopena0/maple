"""Collaborative filtering recommendation models."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """
    Item-based K-Nearest Neighbors collaborative filtering.

    Recommends items similar to those the user has interacted with.
    Uses cosine similarity between item vectors.
    """

    def __init__(
        self,
        name: str = "item_knn",
        k: int = 50,
        min_similarity: float = 0.0,
    ) -> None:
        """
        Initialize Item KNN recommender.

        Args:
            name: Model name
            k: Number of similar items to consider
            min_similarity: Minimum similarity threshold
        """
        super().__init__(name=name)
        self.k = k
        self.min_similarity = min_similarity
        self.item_similarity: Optional[np.ndarray] = None
        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "ItemKNNRecommender":
        """
        Compute item-item similarity matrix.

        Args:
            interaction_matrix: User-item interaction matrix

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        logger.info(f"Computing item similarities for {self.n_items} items...")

        # Compute item-item cosine similarity
        # Transpose so items are rows for pairwise computation
        item_matrix = interaction_matrix.T.tocsr()
        self.item_similarity = cosine_similarity(item_matrix, dense_output=True)

        # Zero out diagonal (item is not similar to itself for recommendations)
        np.fill_diagonal(self.item_similarity, 0)

        # Apply minimum similarity threshold
        if self.min_similarity > 0:
            self.item_similarity[self.item_similarity < self.min_similarity] = 0

        self.is_fitted = True
        logger.info("Item similarity matrix computed")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Generate recommendations based on similar items.

        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Specific items to exclude

        Returns:
            List of (item_idx, score) tuples
        """
        self._check_fitted()

        if self._interaction_matrix is None or self.item_similarity is None:
            raise RuntimeError("Model not properly fitted")

        # Get user's interaction vector
        user_vector = self._interaction_matrix[user_idx].toarray().flatten()

        # Determine items to exclude
        if seen_items is None and exclude_seen:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        # Get items the user has interacted with
        interacted_items = np.where(user_vector > 0)[0]

        if len(interacted_items) == 0:
            # Cold start: return empty or fall back to popularity
            return []

        # Compute scores: weighted sum of similarities
        # For each candidate item, sum similarity to items user has interacted with
        scores = np.zeros(self.n_items)

        for item_idx in interacted_items:
            # Get top-k similar items
            sim_scores = self.item_similarity[item_idx]
            top_k_indices = np.argsort(sim_scores)[-self.k:]

            # Add weighted similarity (weight by user's interaction strength)
            weight = user_vector[item_idx]
            scores[top_k_indices] += weight * sim_scores[top_k_indices]

        # Exclude seen items
        for item_idx in seen_items:
            scores[item_idx] = -np.inf

        # Get top-n items
        top_indices = np.argsort(scores)[-n:][::-1]
        recommendations = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

        return recommendations

    def get_similar_items(
        self,
        item_idx: int,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get items most similar to a given item.

        Args:
            item_idx: Item index
            n: Number of similar items

        Returns:
            List of (item_idx, similarity) tuples
        """
        self._check_fitted()

        if self.item_similarity is None:
            raise RuntimeError("Model not properly fitted")

        similarities = self.item_similarity[item_idx]
        top_indices = np.argsort(similarities)[-n:][::-1]

        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0
        ]


class ALSRecommender(BaseRecommender):
    """
    Alternating Least Squares matrix factorization.

    Uses the implicit library for efficient ALS on implicit feedback data.
    """

    def __init__(
        self,
        name: str = "als",
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        random_state: int = 42,
    ) -> None:
        """
        Initialize ALS recommender.

        Args:
            name: Model name
            factors: Number of latent factors
            regularization: L2 regularization weight
            iterations: Number of ALS iterations
            random_state: Random seed
        """
        super().__init__(name=name)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.model = None
        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "ALSRecommender":
        """
        Train ALS model on interaction data.

        Args:
            interaction_matrix: User-item interaction matrix

        Returns:
            self
        """
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError(
                "The 'implicit' library is required for ALS. "
                "Install with: pip install implicit"
            )

        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        logger.info(
            f"Training ALS model: {self.factors} factors, "
            f"{self.iterations} iterations"
        )

        # Initialize and train model
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        # implicit expects item-user matrix (transposed)
        self.model.fit(interaction_matrix.T.tocsr())

        self.is_fitted = True
        logger.info("ALS model trained")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Generate recommendations using learned embeddings.

        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Specific items to exclude

        Returns:
            List of (item_idx, score) tuples
        """
        self._check_fitted()

        if self.model is None or self._interaction_matrix is None:
            raise RuntimeError("Model not properly fitted")

        # Get recommendations from implicit
        # Filter items based on the interaction matrix
        filter_items = None
        if not exclude_seen:
            # If we don't want to exclude seen, pass empty filter
            filter_items = []

        item_indices, scores = self.model.recommend(
            user_idx,
            self._interaction_matrix[user_idx],
            N=n,
            filter_already_liked_items=exclude_seen,
        )

        # Apply additional filtering if seen_items provided
        if seen_items is not None:
            filtered = [
                (int(idx), float(score))
                for idx, score in zip(item_indices, scores)
                if idx not in seen_items
            ]
            return filtered[:n]

        return [(int(idx), float(score)) for idx, score in zip(item_indices, scores)]

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get latent embedding for a user."""
        self._check_fitted()
        if self.model is None:
            raise RuntimeError("Model not properly fitted")
        return self.model.user_factors[user_idx]

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get latent embedding for an item."""
        self._check_fitted()
        if self.model is None:
            raise RuntimeError("Model not properly fitted")
        return self.model.item_factors[item_idx]

    def get_similar_items(
        self,
        item_idx: int,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get items most similar in embedding space.

        Args:
            item_idx: Item index
            n: Number of similar items

        Returns:
            List of (item_idx, similarity) tuples
        """
        self._check_fitted()

        if self.model is None:
            raise RuntimeError("Model not properly fitted")

        item_indices, scores = self.model.similar_items(item_idx, N=n + 1)

        # Exclude the item itself (first result)
        return [
            (int(idx), float(score))
            for idx, score in zip(item_indices[1:], scores[1:])
        ]


class UserKNNRecommender(BaseRecommender):
    """
    User-based K-Nearest Neighbors collaborative filtering.

    Recommends items liked by similar users.
    """

    def __init__(
        self,
        name: str = "user_knn",
        k: int = 50,
        min_similarity: float = 0.0,
    ) -> None:
        """
        Initialize User KNN recommender.

        Args:
            name: Model name
            k: Number of similar users to consider
            min_similarity: Minimum similarity threshold
        """
        super().__init__(name=name)
        self.k = k
        self.min_similarity = min_similarity
        self.user_similarity: Optional[np.ndarray] = None
        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "UserKNNRecommender":
        """
        Compute user-user similarity matrix.

        Args:
            interaction_matrix: User-item interaction matrix

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        logger.info(f"Computing user similarities for {self.n_users} users...")

        # Compute user-user cosine similarity
        self.user_similarity = cosine_similarity(interaction_matrix, dense_output=True)

        # Zero out diagonal
        np.fill_diagonal(self.user_similarity, 0)

        # Apply minimum similarity threshold
        if self.min_similarity > 0:
            self.user_similarity[self.user_similarity < self.min_similarity] = 0

        self.is_fitted = True
        logger.info("User similarity matrix computed")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Generate recommendations based on similar users.

        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Specific items to exclude

        Returns:
            List of (item_idx, score) tuples
        """
        self._check_fitted()

        if self._interaction_matrix is None or self.user_similarity is None:
            raise RuntimeError("Model not properly fitted")

        # Get top-k similar users
        user_sims = self.user_similarity[user_idx]
        top_k_users = np.argsort(user_sims)[-self.k:]
        top_k_sims = user_sims[top_k_users]

        # Determine items to exclude
        if seen_items is None and exclude_seen:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        # Compute weighted scores from similar users
        scores = np.zeros(self.n_items)

        for sim_user_idx, similarity in zip(top_k_users, top_k_sims):
            if similarity > 0:
                user_items = self._interaction_matrix[sim_user_idx].toarray().flatten()
                scores += similarity * user_items

        # Exclude seen items
        for item_idx in seen_items:
            scores[item_idx] = -np.inf

        # Get top-n items
        top_indices = np.argsort(scores)[-n:][::-1]
        recommendations = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

        return recommendations
