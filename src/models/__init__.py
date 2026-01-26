"""Recommendation models for Maple."""

from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.collaborative import (
    ItemKNNRecommender,
    ALSRecommender,
)

__all__ = [
    "BaseRecommender",
    "PopularityRecommender",
    "ItemKNNRecommender",
    "ALSRecommender",
]
