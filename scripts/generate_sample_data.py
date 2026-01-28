#!/usr/bin/env python3
"""Generate sample interaction data for testing and development."""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np


def generate_sample_data(
    n_users: int = 1000,
    n_products: int = 500,
    n_interactions: int = 50000,
    output_dir: str = "data/sample",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic e-commerce interaction data.

    Creates realistic patterns:
    - Power-law distribution for product popularity
    - User activity varies (some users more active)
    - Temporal patterns (more recent = more interactions)

    Args:
        n_users: Number of users
        n_products: Number of products
        n_interactions: Total number of interactions
        output_dir: Directory to save CSV files
        seed: Random seed for reproducibility

    Returns:
        Tuple of (interactions_df, products_df, users_df)
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate products
    categories = [
        "Electronics", "Clothing", "Home & Kitchen", "Sports", "Books",
        "Beauty", "Toys", "Food & Grocery", "Automotive", "Health"
    ]
    subcategories = {
        "Electronics": ["Phones", "Laptops", "Audio", "Cameras", "Accessories"],
        "Clothing": ["Men's", "Women's", "Kids", "Shoes", "Accessories"],
        "Home & Kitchen": ["Furniture", "Appliances", "Decor", "Cookware", "Storage"],
        "Sports": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Cycling"],
        "Books": ["Fiction", "Non-Fiction", "Technical", "Children's", "Comics"],
        "Beauty": ["Skincare", "Makeup", "Hair Care", "Fragrance", "Tools"],
        "Toys": ["Action Figures", "Board Games", "Educational", "Dolls", "Outdoor"],
        "Food & Grocery": ["Snacks", "Beverages", "Organic", "International", "Pantry"],
        "Automotive": ["Parts", "Accessories", "Tools", "Care", "Electronics"],
        "Health": ["Vitamins", "Personal Care", "Medical", "Fitness", "Wellness"],
    }
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "BrandF", "Generic"]

    products = []
    for i in range(n_products):
        category = random.choice(categories)
        products.append({
            "product_id": f"prod_{i:05d}",
            "name": f"Product {i} - {category}",
            "category": category,
            "subcategory": random.choice(subcategories[category]),
            "price": round(random.uniform(5, 500), 2),
            "brand": random.choice(brands),
            "created_at": (
                datetime.now() - timedelta(days=random.randint(0, 365))
            ).isoformat(),
            "is_active": random.random() > 0.05,  # 95% active
        })

    products_df = pd.DataFrame(products)

    # Generate users
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    genders = ["M", "F", "Other", None]
    locations = ["US-CA", "US-NY", "US-TX", "US-FL", "UK", "DE", "FR", "JP", "AU", None]

    users = []
    for i in range(n_users):
        users.append({
            "user_id": f"user_{i:05d}",
            "created_at": (
                datetime.now() - timedelta(days=random.randint(0, 730))
            ).isoformat(),
            "age_group": random.choice(age_groups) if random.random() > 0.1 else None,
            "gender": random.choice(genders),
            "location": random.choice(locations),
        })

    users_df = pd.DataFrame(users)

    # Generate interactions with realistic patterns
    interaction_types = ["view", "click", "add_to_cart", "purchase", "rating"]
    interaction_weights = [0.50, 0.25, 0.12, 0.08, 0.05]  # Views most common

    # Product popularity follows power law
    product_popularity = np.random.pareto(1.5, n_products) + 1
    product_popularity = product_popularity / product_popularity.sum()

    # User activity follows power law
    user_activity = np.random.pareto(1.2, n_users) + 1
    user_activity = user_activity / user_activity.sum()

    interactions = []
    base_time = datetime.now() - timedelta(days=180)

    for _ in range(n_interactions):
        user_idx = np.random.choice(n_users, p=user_activity)
        product_idx = np.random.choice(n_products, p=product_popularity)

        interaction_type = random.choices(
            interaction_types, weights=interaction_weights
        )[0]

        # Timestamp: more recent interactions more likely
        days_ago = int(np.random.exponential(30))  # Exponential decay
        days_ago = min(days_ago, 180)
        timestamp = base_time + timedelta(
            days=180 - days_ago,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        interaction = {
            "user_id": f"user_{user_idx:05d}",
            "product_id": f"prod_{product_idx:05d}",
            "interaction_type": interaction_type,
            "timestamp": timestamp.isoformat(),
            "session_id": f"sess_{random.randint(0, n_interactions // 10):06d}",
        }

        # Add rating for rating interactions
        if interaction_type == "rating":
            interaction["rating"] = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.05, 0.10, 0.15, 0.35, 0.35]  # Positive skew
            )[0]

        # Add quantity for purchases
        if interaction_type == "purchase":
            interaction["quantity"] = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.70, 0.15, 0.08, 0.04, 0.03]
            )[0]

        interactions.append(interaction)

    interactions_df = pd.DataFrame(interactions)
    interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)

    # Save to CSV
    interactions_df.to_csv(output_path / "interactions.csv", index=False)
    products_df.to_csv(output_path / "products.csv", index=False)
    users_df.to_csv(output_path / "users.csv", index=False)

    print(f"Generated data saved to {output_path}/")
    print(f"  - {len(interactions_df)} interactions")
    print(f"  - {len(products_df)} products")
    print(f"  - {len(users_df)} users")
    print(f"\nInteraction type distribution:")
    print(interactions_df["interaction_type"].value_counts())

    return interactions_df, products_df, users_df


if __name__ == "__main__":
    generate_sample_data()
