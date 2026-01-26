"""Data schemas and models for Maple."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class User(BaseModel):
    """User entity."""

    user_id: str = Field(..., description="Unique user identifier")
    created_at: Optional[datetime] = Field(default=None, description="Account creation time")

    # Optional demographic features (for content-based filtering)
    age_group: Optional[str] = Field(default=None, description="Age group bucket")
    gender: Optional[str] = Field(default=None, description="Gender")
    location: Optional[str] = Field(default=None, description="Geographic location")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "created_at": "2024-01-15T10:30:00Z",
                "age_group": "25-34",
                "gender": "F",
                "location": "US-CA",
            }
        }


class Product(BaseModel):
    """Product entity."""

    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    category: Optional[str] = Field(default=None, description="Product category")
    subcategory: Optional[str] = Field(default=None, description="Product subcategory")
    price: Optional[float] = Field(default=None, ge=0, description="Product price")
    brand: Optional[str] = Field(default=None, description="Brand name")
    description: Optional[str] = Field(default=None, description="Product description")
    tags: list[str] = Field(default_factory=list, description="Product tags")
    created_at: Optional[datetime] = Field(default=None, description="Product creation time")
    is_active: bool = Field(default=True, description="Whether product is available")

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "prod_456",
                "name": "Wireless Bluetooth Headphones",
                "category": "Electronics",
                "subcategory": "Audio",
                "price": 79.99,
                "brand": "AudioTech",
                "description": "High-quality wireless headphones with noise cancellation",
                "tags": ["wireless", "bluetooth", "noise-cancelling"],
                "created_at": "2024-01-10T08:00:00Z",
                "is_active": True,
            }
        }


class Interaction(BaseModel):
    """User-product interaction event."""

    user_id: str = Field(..., description="User identifier")
    product_id: str = Field(..., description="Product identifier")
    interaction_type: str = Field(
        ...,
        description="Type of interaction: view, click, add_to_cart, purchase, rating"
    )
    timestamp: datetime = Field(..., description="When the interaction occurred")

    # Optional fields depending on interaction type
    rating: Optional[float] = Field(
        default=None,
        ge=1,
        le=5,
        description="Rating value (1-5) for rating interactions"
    )
    quantity: Optional[int] = Field(
        default=None,
        ge=1,
        description="Quantity for purchase interactions"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for session-based recommendations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "product_id": "prod_456",
                "interaction_type": "purchase",
                "timestamp": "2024-01-20T14:25:00Z",
                "quantity": 1,
                "session_id": "sess_789",
            }
        }


# Interaction weights for implicit feedback
INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
    "rating": 4.0,  # Explicit feedback, weighted by rating value
}


class RecommendationRequest(BaseModel):
    """Request for recommendations."""

    user_id: str = Field(..., description="User to get recommendations for")
    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations"
    )
    exclude_purchased: bool = Field(
        default=True,
        description="Exclude previously purchased items"
    )
    category_filter: Optional[str] = Field(
        default=None,
        description="Filter by category"
    )


class RecommendationResponse(BaseModel):
    """Response containing recommendations."""

    user_id: str
    recommendations: list[dict] = Field(
        ...,
        description="List of recommended products with scores"
    )
    model_used: str = Field(..., description="Model that generated recommendations")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "recommendations": [
                    {"product_id": "prod_001", "score": 0.95, "rank": 1},
                    {"product_id": "prod_002", "score": 0.87, "rank": 2},
                ],
                "model_used": "als_collaborative_filter",
                "generated_at": "2024-01-20T15:00:00Z",
            }
        }
