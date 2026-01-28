"""Content-based recommendation models."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, normalize

from src.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommender using product features.

    Uses TF-IDF for text features and one-hot encoding for categorical features
    to compute item-item similarity based on content.
    """

    def __init__(
        self,
        name: str = "content_based",
        text_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None,
        use_tfidf: bool = True,
    ) -> None:
        """
        Initialize Content-Based recommender.

        Args:
            name: Model name
            text_features: List of text feature column names (e.g., ['description', 'name'])
            categorical_features: List of categorical feature column names (e.g., ['category', 'brand'])
            use_tfidf: Whether to use TF-IDF for text features
        """
        super().__init__(name=name)
        self.text_features = text_features or []
        self.categorical_features = categorical_features or []
        self.use_tfidf = use_tfidf

        self.item_features: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None
        self._interaction_matrix: Optional[csr_matrix] = None

        # Feature transformers
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.onehot_encoder: Optional[OneHotEncoder] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        item_features_df=None,
        **kwargs,
    ) -> "ContentBasedRecommender":
        """
        Build item feature matrix and compute similarities.

        Args:
            interaction_matrix: User-item interaction matrix
            item_features_df: DataFrame with item features (indexed by item_idx)

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        if item_features_df is None:
            raise ValueError("item_features_df is required for content-based filtering")

        logger.info(f"Building content features for {self.n_items} items...")

        feature_matrices = []

        # Process text features with TF-IDF
        if self.text_features:
            text_data = item_features_df[self.text_features].fillna("").agg(" ".join, axis=1)

            if self.use_tfidf:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words="english",
                    ngram_range=(1, 2),
                )
                text_matrix = self.tfidf_vectorizer.fit_transform(text_data)
            else:
                # Simple count vectorizer fallback
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer(max_features=1000, stop_words="english")
                text_matrix = vectorizer.fit_transform(text_data)

            feature_matrices.append(text_matrix)
            logger.info(f"Text features shape: {text_matrix.shape}")

        # Process categorical features with one-hot encoding
        if self.categorical_features:
            cat_data = item_features_df[self.categorical_features].fillna("unknown")

            self.onehot_encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
            cat_matrix = self.onehot_encoder.fit_transform(cat_data)

            feature_matrices.append(cat_matrix)
            logger.info(f"Categorical features shape: {cat_matrix.shape}")

        # Combine all features
        if len(feature_matrices) > 1:
            combined_features = hstack(feature_matrices)
        elif len(feature_matrices) == 1:
            combined_features = feature_matrices[0]
        else:
            raise ValueError("No features specified for content-based filtering")

        # Normalize features
        self.item_features = normalize(combined_features, norm="l2")

        # Compute item-item similarity
        logger.info("Computing item similarities...")
        self.item_similarity = cosine_similarity(self.item_features)

        # Zero diagonal
        np.fill_diagonal(self.item_similarity, 0)

        self.is_fitted = True
        logger.info(f"Content-based model fitted with {self.item_features.shape[1]} features")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Generate recommendations based on content similarity to user's liked items.

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

        # Get items user has interacted with
        interacted_items = np.where(user_vector > 0)[0]

        if len(interacted_items) == 0:
            # Cold start: return empty
            return []

        # Compute weighted average similarity to user's items
        # Weight by interaction strength
        weights = user_vector[interacted_items]
        weights = weights / weights.sum()  # Normalize

        # Score for each item = weighted avg similarity to user's items
        scores = np.zeros(self.n_items)
        for item_idx, weight in zip(interacted_items, weights):
            scores += weight * self.item_similarity[item_idx]

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
        Get items most similar in content/features.

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

    def get_item_features(self, item_idx: int) -> np.ndarray:
        """Get feature vector for an item."""
        self._check_fitted()
        if self.item_features is None:
            raise RuntimeError("Model not properly fitted")
        return np.asarray(self.item_features[item_idx].todense()).flatten()


class TFIDFRecommender(BaseRecommender):
    """
    Simple TF-IDF based recommender using only text descriptions.

    Lightweight alternative when only text features are available.
    """

    def __init__(
        self,
        name: str = "tfidf",
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        super().__init__(name=name)
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_vectors: Optional[csr_matrix] = None
        self.item_similarity: Optional[np.ndarray] = None
        self._interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        item_texts: Optional[list[str]] = None,
        **kwargs,
    ) -> "TFIDFRecommender":
        """
        Build TF-IDF vectors and compute similarities.

        Args:
            interaction_matrix: User-item interaction matrix
            item_texts: List of text descriptions for each item

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        if item_texts is None:
            raise ValueError("item_texts is required for TF-IDF recommender")

        if len(item_texts) != self.n_items:
            raise ValueError(f"Expected {self.n_items} texts, got {len(item_texts)}")

        logger.info(f"Building TF-IDF vectors for {self.n_items} items...")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=self.ngram_range,
        )

        self.item_vectors = self.vectorizer.fit_transform(item_texts)
        self.item_similarity = cosine_similarity(self.item_vectors)
        np.fill_diagonal(self.item_similarity, 0)

        self.is_fitted = True
        logger.info(f"TF-IDF model fitted with {self.item_vectors.shape[1]} features")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """Generate recommendations based on TF-IDF similarity."""
        self._check_fitted()

        if self._interaction_matrix is None or self.item_similarity is None:
            raise RuntimeError("Model not properly fitted")

        user_vector = self._interaction_matrix[user_idx].toarray().flatten()

        if seen_items is None and exclude_seen:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        interacted_items = np.where(user_vector > 0)[0]

        if len(interacted_items) == 0:
            return []

        # Average similarity to user's items
        scores = self.item_similarity[interacted_items].mean(axis=0)

        for item_idx in seen_items:
            scores[item_idx] = -np.inf

        top_indices = np.argsort(scores)[-n:][::-1]
        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

    def get_similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Get similar items by TF-IDF similarity."""
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
