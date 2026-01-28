"""Recommendation models for Maple."""

from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender, TimeDecayPopularityRecommender
from src.models.collaborative import (
    ItemKNNRecommender,
    UserKNNRecommender,
    ALSRecommender,
)
from src.models.content_based import ContentBasedRecommender, TFIDFRecommender
from src.models.hybrid import HybridRecommender, FeatureAugmentedCF
from src.models.neural import BPRRecommender
from src.models.ensemble import (
    EnsembleRecommender,
    ReRanker,
    BusinessRulesFilter,
)

# NCF requires PyTorch
try:
    from src.models.neural import NeuralCFRecommender
    NEURAL_AVAILABLE = True
except ImportError:
    NeuralCFRecommender = None
    NEURAL_AVAILABLE = False

__all__ = [
    # Base
    "BaseRecommender",
    # Popularity
    "PopularityRecommender",
    "TimeDecayPopularityRecommender",
    # Collaborative Filtering
    "ItemKNNRecommender",
    "UserKNNRecommender",
    "ALSRecommender",
    # Content-Based
    "ContentBasedRecommender",
    "TFIDFRecommender",
    # Hybrid
    "HybridRecommender",
    "FeatureAugmentedCF",
    # Neural
    "BPRRecommender",
    "NeuralCFRecommender",
    "NEURAL_AVAILABLE",
    # Ensemble
    "EnsembleRecommender",
    "ReRanker",
    "BusinessRulesFilter",
]
