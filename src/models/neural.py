"""Neural network-based recommendation models."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None


class BPRRecommender(BaseRecommender):
    """
    Bayesian Personalized Ranking recommender.

    Optimizes for pairwise ranking: positive items should score higher
    than negative items for each user. Does not require PyTorch.
    """

    def __init__(
        self,
        name: str = "bpr",
        n_factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        n_epochs: int = 20,
        n_samples: int = 100000,
    ) -> None:
        """
        Initialize BPR recommender.

        Args:
            name: Model name
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            regularization: L2 regularization strength
            n_epochs: Number of training epochs
            n_samples: Number of samples per epoch
        """
        super().__init__(name=name)
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.n_samples = n_samples

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self._interaction_matrix: Optional[csr_matrix] = None
        self._user_items: Optional[dict[int, set[int]]] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs,
    ) -> "BPRRecommender":
        """
        Train BPR model using SGD.

        Args:
            interaction_matrix: User-item interaction matrix

        Returns:
            self
        """
        self.n_users, self.n_items = interaction_matrix.shape
        self._interaction_matrix = interaction_matrix

        logger.info(
            f"Training BPR: {self.n_factors} factors, {self.n_epochs} epochs"
        )

        # Build user->items mapping
        self._user_items = {}
        for user in range(self.n_users):
            self._user_items[user] = set(interaction_matrix[user].indices)

        # Initialize factors
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.item_bias = np.zeros(self.n_items)

        # Get users with at least one interaction
        active_users = [u for u in range(self.n_users) if len(self._user_items[u]) > 0]
        all_items = set(range(self.n_items))

        # Training
        for epoch in range(self.n_epochs):
            total_loss = 0.0

            for _ in range(self.n_samples):
                # Sample user
                user = np.random.choice(active_users)
                user_items = self._user_items[user]

                # Sample positive item
                pos_item = np.random.choice(list(user_items))

                # Sample negative item
                neg_pool = list(all_items - user_items)
                if len(neg_pool) == 0:
                    continue
                neg_item = np.random.choice(neg_pool)

                # Compute scores
                pos_score = (
                    np.dot(self.user_factors[user], self.item_factors[pos_item])
                    + self.item_bias[pos_item]
                )
                neg_score = (
                    np.dot(self.user_factors[user], self.item_factors[neg_item])
                    + self.item_bias[neg_item]
                )

                # BPR loss gradient
                diff = pos_score - neg_score
                sigmoid = 1.0 / (1.0 + np.exp(min(diff, 500)))  # Clip for stability
                total_loss += -np.log(max(1 - sigmoid, 1e-10))

                # Update factors
                self.user_factors[user] += self.learning_rate * (
                    sigmoid * (self.item_factors[pos_item] - self.item_factors[neg_item])
                    - self.regularization * self.user_factors[user]
                )

                self.item_factors[pos_item] += self.learning_rate * (
                    sigmoid * self.user_factors[user]
                    - self.regularization * self.item_factors[pos_item]
                )

                self.item_factors[neg_item] += self.learning_rate * (
                    -sigmoid * self.user_factors[user]
                    - self.regularization * self.item_factors[neg_item]
                )

                self.item_bias[pos_item] += self.learning_rate * (
                    sigmoid - self.regularization * self.item_bias[pos_item]
                )
                self.item_bias[neg_item] += self.learning_rate * (
                    -sigmoid - self.regularization * self.item_bias[neg_item]
                )

            avg_loss = total_loss / self.n_samples
            if (epoch + 1) % 5 == 0:
                logger.debug(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        logger.info("BPR model trained")
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """Generate recommendations using learned factors."""
        self._check_fitted()

        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model not properly fitted")

        if self._interaction_matrix is None:
            raise RuntimeError("Interaction matrix not available")

        # Compute scores
        scores = (
            self.item_factors @ self.user_factors[user_idx]
            + self.item_bias
        )

        # Determine items to exclude
        if seen_items is None and exclude_seen:
            seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
        elif seen_items is None:
            seen_items = set()

        for item_idx in seen_items:
            scores[item_idx] = -np.inf

        # Get top-n
        top_indices = np.argsort(scores)[-n:][::-1]
        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

    def get_similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Get similar items by factor similarity."""
        self._check_fitted()
        if self.item_factors is None:
            raise RuntimeError("Model not properly fitted")

        item_embedding = self.item_factors[item_idx]
        similarities = self.item_factors @ item_embedding
        similarities[item_idx] = -np.inf

        top_indices = np.argsort(similarities)[-n:][::-1]
        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > -np.inf
        ]


# PyTorch-dependent classes - only defined if torch is available
if TORCH_AVAILABLE:

    class InteractionDataset:
        """PyTorch Dataset for user-item interactions."""

        def __init__(
            self,
            users: np.ndarray,
            items: np.ndarray,
            labels: np.ndarray,
        ):
            self.users = torch.LongTensor(users)
            self.items = torch.LongTensor(items)
            self.labels = torch.FloatTensor(labels)

        def __len__(self) -> int:
            return len(self.users)

        def __getitem__(self, idx: int) -> tuple:
            return self.users[idx], self.items[idx], self.labels[idx]

    class NCFNetwork(nn.Module):
        """
        Neural Collaborative Filtering network.

        Combines:
        - GMF (Generalized Matrix Factorization): Element-wise product of embeddings
        - MLP (Multi-Layer Perceptron): Concatenated embeddings through neural network
        """

        def __init__(
            self,
            n_users: int,
            n_items: int,
            embedding_dim: int = 32,
            mlp_layers: list[int] = None,
            dropout: float = 0.2,
        ):
            super().__init__()

            if mlp_layers is None:
                mlp_layers = [64, 32, 16]

            self.n_users = n_users
            self.n_items = n_items
            self.embedding_dim = embedding_dim

            # GMF embeddings
            self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
            self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

            # MLP embeddings
            self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
            self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

            # MLP layers
            mlp_input_dim = embedding_dim * 2
            layers = []
            for hidden_dim in mlp_layers:
                layers.append(nn.Linear(mlp_input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                mlp_input_dim = hidden_dim

            self.mlp = nn.Sequential(*layers)

            # Final prediction layer
            final_input_dim = embedding_dim + mlp_layers[-1]
            self.final = nn.Linear(final_input_dim, 1)

            self._init_weights()

        def _init_weights(self):
            """Initialize embeddings with normal distribution."""
            nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
            nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
            nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
            nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        def forward(self, user_ids, item_ids):
            """Forward pass."""
            # GMF path
            user_gmf = self.user_embedding_gmf(user_ids)
            item_gmf = self.item_embedding_gmf(item_ids)
            gmf_output = user_gmf * item_gmf

            # MLP path
            user_mlp = self.user_embedding_mlp(user_ids)
            item_mlp = self.item_embedding_mlp(item_ids)
            mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
            mlp_output = self.mlp(mlp_input)

            # Combine
            concat = torch.cat([gmf_output, mlp_output], dim=-1)
            output = self.final(concat)

            return output.squeeze(-1)

    class NeuralCFRecommender(BaseRecommender):
        """
        Neural Collaborative Filtering recommender.

        Uses deep learning to model user-item interactions with both
        linear (GMF) and non-linear (MLP) components.
        """

        def __init__(
            self,
            name: str = "ncf",
            embedding_dim: int = 32,
            mlp_layers: Optional[list[int]] = None,
            dropout: float = 0.2,
            learning_rate: float = 0.001,
            batch_size: int = 256,
            n_epochs: int = 10,
            n_negatives: int = 4,
            device: Optional[str] = None,
        ) -> None:
            super().__init__(name=name)

            self.embedding_dim = embedding_dim
            self.mlp_layers = mlp_layers or [64, 32, 16]
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.n_negatives = n_negatives

            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model: Optional[NCFNetwork] = None
            self._interaction_matrix: Optional[csr_matrix] = None

        def _create_training_data(
            self,
            interaction_matrix: csr_matrix,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Create training data with negative sampling."""
            users, items = interaction_matrix.nonzero()
            n_interactions = len(users)

            positive_users = users.astype(np.int64)
            positive_items = items.astype(np.int64)
            positive_labels = np.ones(n_interactions, dtype=np.float32)

            all_items = set(range(self.n_items))
            negative_users = []
            negative_items = []

            for user in users:
                user_items = set(interaction_matrix[user].indices)
                negative_pool = list(all_items - user_items)

                if len(negative_pool) > 0:
                    n_neg = min(self.n_negatives, len(negative_pool))
                    neg_items = np.random.choice(negative_pool, n_neg, replace=False)
                    negative_users.extend([user] * n_neg)
                    negative_items.extend(neg_items)

            negative_users = np.array(negative_users, dtype=np.int64)
            negative_items = np.array(negative_items, dtype=np.int64)
            negative_labels = np.zeros(len(negative_users), dtype=np.float32)

            all_users = np.concatenate([positive_users, negative_users])
            all_items = np.concatenate([positive_items, negative_items])
            all_labels = np.concatenate([positive_labels, negative_labels])

            return all_users, all_items, all_labels

        def fit(
            self,
            interaction_matrix: csr_matrix,
            **kwargs,
        ) -> "NeuralCFRecommender":
            """Train NCF model."""
            self.n_users, self.n_items = interaction_matrix.shape
            self._interaction_matrix = interaction_matrix

            logger.info(
                f"Training NCF: embedding_dim={self.embedding_dim}, "
                f"mlp_layers={self.mlp_layers}, epochs={self.n_epochs}"
            )

            self.model = NCFNetwork(
                n_users=self.n_users,
                n_items=self.n_items,
                embedding_dim=self.embedding_dim,
                mlp_layers=self.mlp_layers,
                dropout=self.dropout,
            ).to(self.device)

            users, items, labels = self._create_training_data(interaction_matrix)
            dataset = InteractionDataset(users, items, labels)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            self.model.train()
            for epoch in range(self.n_epochs):
                total_loss = 0.0
                n_batches = 0

                for user_batch, item_batch, label_batch in dataloader:
                    user_batch = user_batch.to(self.device)
                    item_batch = item_batch.to(self.device)
                    label_batch = label_batch.to(self.device)

                    optimizer.zero_grad()
                    predictions = self.model(user_batch, item_batch)
                    loss = criterion(predictions, label_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                avg_loss = total_loss / n_batches
                logger.debug(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

            self.is_fitted = True
            logger.info("NCF model trained")
            return self

        def recommend(
            self,
            user_idx: int,
            n: int = 10,
            exclude_seen: bool = True,
            seen_items: Optional[set[int]] = None,
        ) -> list[tuple[int, float]]:
            """Generate recommendations using trained NCF model."""
            self._check_fitted()

            if self.model is None or self._interaction_matrix is None:
                raise RuntimeError("Model not properly fitted")

            if seen_items is None and exclude_seen:
                seen_items = self._get_seen_items(user_idx, self._interaction_matrix)
            elif seen_items is None:
                seen_items = set()

            self.model.eval()
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx] * self.n_items).to(self.device)
                item_tensor = torch.LongTensor(list(range(self.n_items))).to(self.device)
                scores = self.model(user_tensor, item_tensor).cpu().numpy()

            for item_idx in seen_items:
                scores[item_idx] = -np.inf

            top_indices = np.argsort(scores)[-n:][::-1]
            return [
                (int(idx), float(scores[idx]))
                for idx in top_indices
                if scores[idx] > -np.inf
            ]

        def get_user_embedding(self, user_idx: int) -> np.ndarray:
            """Get combined user embedding (GMF + MLP)."""
            self._check_fitted()
            if self.model is None:
                raise RuntimeError("Model not properly fitted")

            self.model.eval()
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                gmf_emb = self.model.user_embedding_gmf(user_tensor).cpu().numpy()
                mlp_emb = self.model.user_embedding_mlp(user_tensor).cpu().numpy()
            return np.concatenate([gmf_emb, mlp_emb], axis=-1).flatten()

        def get_item_embedding(self, item_idx: int) -> np.ndarray:
            """Get combined item embedding (GMF + MLP)."""
            self._check_fitted()
            if self.model is None:
                raise RuntimeError("Model not properly fitted")

            self.model.eval()
            with torch.no_grad():
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                gmf_emb = self.model.item_embedding_gmf(item_tensor).cpu().numpy()
                mlp_emb = self.model.item_embedding_mlp(item_tensor).cpu().numpy()
            return np.concatenate([gmf_emb, mlp_emb], axis=-1).flatten()

else:
    # Placeholder when PyTorch is not available
    NeuralCFRecommender = None
    NCFNetwork = None
    InteractionDataset = None
