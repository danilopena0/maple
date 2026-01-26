"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix

from src.data.schemas import INTERACTION_WEIGHTS


class DataLoader:
    """Load and preprocess interaction data for recommendation models."""

    def __init__(self) -> None:
        self.interactions_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None

        # Mappings between IDs and indices
        self.user_id_to_idx: dict[str, int] = {}
        self.idx_to_user_id: dict[int, str] = {}
        self.product_id_to_idx: dict[str, int] = {}
        self.idx_to_product_id: dict[int, str] = {}

        self.n_users: int = 0
        self.n_products: int = 0

    def load_interactions(
        self,
        filepath: Optional[str | Path] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Load interaction data from file or DataFrame.

        Expected columns: user_id, product_id, interaction_type, timestamp
        Optional columns: rating, quantity, session_id
        """
        if df is not None:
            self.interactions_df = df.copy()
        elif filepath is not None:
            filepath = Path(filepath)
            if filepath.suffix == ".csv":
                self.interactions_df = pd.read_csv(filepath)
            elif filepath.suffix == ".parquet":
                self.interactions_df = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        else:
            raise ValueError("Either filepath or df must be provided")

        # Validate required columns
        required_cols = ["user_id", "product_id", "interaction_type", "timestamp"]
        missing = set(required_cols) - set(self.interactions_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert timestamp to datetime
        self.interactions_df["timestamp"] = pd.to_datetime(
            self.interactions_df["timestamp"]
        )

        # Sort by timestamp
        self.interactions_df = self.interactions_df.sort_values("timestamp")

        # Build ID mappings
        self._build_id_mappings()

        logger.info(
            f"Loaded {len(self.interactions_df)} interactions, "
            f"{self.n_users} users, {self.n_products} products"
        )

        return self.interactions_df

    def load_products(
        self,
        filepath: Optional[str | Path] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load product catalog data."""
        if df is not None:
            self.products_df = df.copy()
        elif filepath is not None:
            filepath = Path(filepath)
            if filepath.suffix == ".csv":
                self.products_df = pd.read_csv(filepath)
            elif filepath.suffix == ".parquet":
                self.products_df = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        else:
            raise ValueError("Either filepath or df must be provided")

        logger.info(f"Loaded {len(self.products_df)} products")
        return self.products_df

    def load_users(
        self,
        filepath: Optional[str | Path] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load user data."""
        if df is not None:
            self.users_df = df.copy()
        elif filepath is not None:
            filepath = Path(filepath)
            if filepath.suffix == ".csv":
                self.users_df = pd.read_csv(filepath)
            elif filepath.suffix == ".parquet":
                self.users_df = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        else:
            raise ValueError("Either filepath or df must be provided")

        logger.info(f"Loaded {len(self.users_df)} users")
        return self.users_df

    def _build_id_mappings(self) -> None:
        """Build mappings between string IDs and integer indices."""
        if self.interactions_df is None:
            raise ValueError("Interactions must be loaded first")

        # Get unique users and products
        unique_users = self.interactions_df["user_id"].unique()
        unique_products = self.interactions_df["product_id"].unique()

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}

        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(unique_products)}
        self.idx_to_product_id = {idx: pid for pid, idx in self.product_id_to_idx.items()}

        self.n_users = len(unique_users)
        self.n_products = len(unique_products)

    def get_interaction_matrix(
        self,
        weighted: bool = True,
        binary: bool = False,
    ) -> csr_matrix:
        """
        Build user-item interaction matrix.

        Args:
            weighted: Apply interaction type weights
            binary: Convert to binary matrix (0/1)

        Returns:
            Sparse CSR matrix of shape (n_users, n_products)
        """
        if self.interactions_df is None:
            raise ValueError("Interactions must be loaded first")

        df = self.interactions_df.copy()

        # Map IDs to indices
        df["user_idx"] = df["user_id"].map(self.user_id_to_idx)
        df["product_idx"] = df["product_id"].map(self.product_id_to_idx)

        # Calculate interaction values
        if weighted:
            df["value"] = df["interaction_type"].map(INTERACTION_WEIGHTS)
            # For ratings, multiply by rating value
            mask = (df["interaction_type"] == "rating") & df["rating"].notna()
            df.loc[mask, "value"] = df.loc[mask, "rating"]
        else:
            df["value"] = 1.0

        # Aggregate multiple interactions (sum)
        aggregated = (
            df.groupby(["user_idx", "product_idx"])["value"]
            .sum()
            .reset_index()
        )

        # Build sparse matrix
        matrix = csr_matrix(
            (aggregated["value"], (aggregated["user_idx"], aggregated["product_idx"])),
            shape=(self.n_users, self.n_products),
        )

        if binary:
            matrix.data = np.ones_like(matrix.data)

        logger.info(
            f"Built interaction matrix: {matrix.shape}, "
            f"density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}"
        )

        return matrix

    def train_test_split(
        self,
        test_ratio: float = 0.2,
        by_time: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into train and test sets.

        Args:
            test_ratio: Fraction of data for testing
            by_time: If True, split by timestamp (recommended for recsys)

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.interactions_df is None:
            raise ValueError("Interactions must be loaded first")

        df = self.interactions_df.copy()

        if by_time:
            # Time-based split
            split_idx = int(len(df) * (1 - test_ratio))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
        else:
            # Random split (not recommended for production)
            mask = np.random.random(len(df)) > test_ratio
            train_df = df[mask].copy()
            test_df = df[~mask].copy()

        logger.info(
            f"Split data: {len(train_df)} train, {len(test_df)} test "
            f"({len(test_df) / len(df):.1%} test ratio)"
        )

        return train_df, test_df

    def get_user_history(self, user_id: str) -> list[str]:
        """Get list of products a user has interacted with."""
        if self.interactions_df is None:
            return []

        user_interactions = self.interactions_df[
            self.interactions_df["user_id"] == user_id
        ]
        return user_interactions["product_id"].unique().tolist()

    def get_popular_products(self, n: int = 100) -> list[tuple[str, int]]:
        """Get most popular products by interaction count."""
        if self.interactions_df is None:
            return []

        popularity = (
            self.interactions_df.groupby("product_id")
            .size()
            .sort_values(ascending=False)
            .head(n)
        )

        return list(popularity.items())
