"""FastAPI REST API for Maple recommendations."""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.data.loader import DataLoader
from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemKNNRecommender, ALSRecommender
from src.models.base import BaseRecommender


# Global state for loaded models and data
class AppState:
    """Application state container."""

    def __init__(self):
        self.data_loader: Optional[DataLoader] = None
        self.models: dict[str, BaseRecommender] = {}
        self.default_model: str = "popularity"
        self.is_ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("Starting Maple Recommendation API...")

    # Initialize with sample data if available
    sample_data_path = os.getenv("MAPLE_DATA_PATH")
    if sample_data_path and os.path.exists(sample_data_path):
        try:
            await load_data_and_train(sample_data_path)
        except Exception as e:
            logger.warning(f"Failed to load initial data: {e}")

    yield

    logger.info("Shutting down Maple Recommendation API...")


app = FastAPI(
    title="Maple Recommendation API",
    description="Product recommendation system API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class RecommendationItem(BaseModel):
    """Single recommendation item."""

    product_id: str
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    """Response containing recommendations."""

    user_id: str
    recommendations: list[RecommendationItem]
    model_used: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SimilarItemsResponse(BaseModel):
    """Response for similar items query."""

    product_id: str
    similar_items: list[RecommendationItem]
    model_used: str


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str
    type: str
    is_fitted: bool
    n_users: int
    n_items: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    is_ready: bool
    models_loaded: list[str]
    default_model: str


class TrainRequest(BaseModel):
    """Request to train models."""

    data_path: str = Field(..., description="Path to interaction data file")
    models_to_train: list[str] = Field(
        default=["popularity", "item_knn", "als"],
        description="Models to train",
    )


class TrainResponse(BaseModel):
    """Response after training."""

    status: str
    models_trained: list[str]
    n_users: int
    n_items: int
    n_interactions: int


# Helper functions
async def load_data_and_train(data_path: str, models_to_train: list[str] = None):
    """Load data and train specified models."""
    if models_to_train is None:
        models_to_train = ["popularity", "item_knn", "als"]

    state.data_loader = DataLoader()
    state.data_loader.load_interactions(data_path)

    interaction_matrix = state.data_loader.get_interaction_matrix(weighted=True)

    # Train requested models
    if "popularity" in models_to_train:
        pop_model = PopularityRecommender()
        pop_model.fit(interaction_matrix)
        state.models["popularity"] = pop_model
        logger.info("Popularity model trained")

    if "item_knn" in models_to_train:
        knn_model = ItemKNNRecommender(k=50)
        knn_model.fit(interaction_matrix)
        state.models["item_knn"] = knn_model
        logger.info("Item KNN model trained")

    if "als" in models_to_train:
        try:
            als_model = ALSRecommender(factors=64, iterations=15)
            als_model.fit(interaction_matrix)
            state.models["als"] = als_model
            logger.info("ALS model trained")
        except ImportError:
            logger.warning("ALS model skipped - implicit library not installed")

    state.is_ready = len(state.models) > 0
    if state.models:
        state.default_model = list(state.models.keys())[0]


def get_model(model_name: Optional[str] = None) -> BaseRecommender:
    """Get a model by name or return default."""
    if not state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Train models first.",
        )

    name = model_name or state.default_model
    if name not in state.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available: {list(state.models.keys())}",
        )

    return state.models[name]


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        is_ready=state.is_ready,
        models_loaded=list(state.models.keys()),
        default_model=state.default_model,
    )


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List all loaded models."""
    return [
        ModelInfo(
            name=name,
            type=model.__class__.__name__,
            is_fitted=model.is_fitted,
            n_users=model.n_users,
            n_items=model.n_items,
        )
        for name, model in state.models.items()
    ]


@app.post("/train", response_model=TrainResponse)
async def train_models(request: TrainRequest):
    """Train models on provided data."""
    try:
        await load_data_and_train(request.data_path, request.models_to_train)

        return TrainResponse(
            status="success",
            models_trained=list(state.models.keys()),
            n_users=state.data_loader.n_users,
            n_items=state.data_loader.n_products,
            n_interactions=len(state.data_loader.interactions_df),
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    n: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
    model: Optional[str] = Query(default=None, description="Model to use"),
    exclude_seen: bool = Query(default=True, description="Exclude items user has seen"),
):
    """
    Get personalized recommendations for a user.

    Args:
        user_id: User identifier
        n: Number of recommendations (1-100)
        model: Model name (default: popularity)
        exclude_seen: Whether to exclude previously seen items
    """
    rec_model = get_model(model)

    # Map user_id to index
    if state.data_loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    if user_id not in state.data_loader.user_id_to_idx:
        # Cold start: return popular items
        logger.info(f"Cold start for user {user_id}, falling back to popularity")
        pop_model = state.models.get("popularity")
        if pop_model:
            recs = pop_model.recommend(0, n=n, exclude_seen=False)
        else:
            raise HTTPException(
                status_code=404,
                detail=f"User '{user_id}' not found and no fallback available",
            )
    else:
        user_idx = state.data_loader.user_id_to_idx[user_id]
        recs = rec_model.recommend(user_idx, n=n, exclude_seen=exclude_seen)

    # Map indices back to IDs
    recommendations = [
        RecommendationItem(
            product_id=state.data_loader.idx_to_product_id.get(idx, str(idx)),
            score=score,
            rank=rank + 1,
        )
        for rank, (idx, score) in enumerate(recs)
    ]

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        model_used=rec_model.name,
    )


@app.get("/similar/{product_id}", response_model=SimilarItemsResponse)
async def get_similar_items(
    product_id: str,
    n: int = Query(default=10, ge=1, le=100, description="Number of similar items"),
    model: Optional[str] = Query(default="item_knn", description="Model to use"),
):
    """
    Get items similar to a given product.

    Args:
        product_id: Product identifier
        n: Number of similar items (1-100)
        model: Model name (default: item_knn)
    """
    rec_model = get_model(model)

    if not hasattr(rec_model, "get_similar_items"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' does not support similar items",
        )

    if state.data_loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    if product_id not in state.data_loader.product_id_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found",
        )

    item_idx = state.data_loader.product_id_to_idx[product_id]
    similar = rec_model.get_similar_items(item_idx, n=n)

    similar_items = [
        RecommendationItem(
            product_id=state.data_loader.idx_to_product_id.get(idx, str(idx)),
            score=score,
            rank=rank + 1,
        )
        for rank, (idx, score) in enumerate(similar)
    ]

    return SimilarItemsResponse(
        product_id=product_id,
        similar_items=similar_items,
        model_used=rec_model.name,
    )


@app.get("/popular", response_model=RecommendationResponse)
async def get_popular_items(
    n: int = Query(default=10, ge=1, le=100, description="Number of items"),
):
    """Get globally popular items."""
    if state.data_loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    popular = state.data_loader.get_popular_products(n=n)

    recommendations = [
        RecommendationItem(
            product_id=product_id,
            score=float(count),
            rank=rank + 1,
        )
        for rank, (product_id, count) in enumerate(popular)
    ]

    return RecommendationResponse(
        user_id="__global__",
        recommendations=recommendations,
        model_used="popularity_direct",
    )


def run():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
