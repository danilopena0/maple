#!/usr/bin/env python3
"""
Quickstart example for Maple Recommendation System.

This script demonstrates:
1. Loading interaction data
2. Training multiple recommendation models
3. Generating recommendations
4. Evaluating model performance
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.loader import DataLoader
from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemKNNRecommender, ALSRecommender
from src.evaluation.metrics import evaluate_model, print_evaluation_results, compare_models


def generate_demo_data() -> pd.DataFrame:
    """Generate small demo dataset for quick testing."""
    np.random.seed(42)

    n_users = 100
    n_products = 50
    n_interactions = 2000

    # Generate interactions
    interactions = []
    base_time = datetime.now() - timedelta(days=30)

    for _ in range(n_interactions):
        user_id = f"user_{np.random.randint(0, n_users):03d}"
        product_id = f"prod_{np.random.choice(n_products, p=np.random.dirichlet(np.ones(n_products) * 0.5)):03d}"
        interaction_type = np.random.choice(
            ["view", "click", "add_to_cart", "purchase"],
            p=[0.5, 0.25, 0.15, 0.1]
        )
        timestamp = base_time + timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24)
        )

        interactions.append({
            "user_id": user_id,
            "product_id": product_id,
            "interaction_type": interaction_type,
            "timestamp": timestamp.isoformat(),
        })

    return pd.DataFrame(interactions)


def main():
    print("=" * 60)
    print(" Maple Recommendation System - Quickstart Demo")
    print("=" * 60)

    # Step 1: Load or generate data
    print("\n[1/5] Loading data...")

    data_path = Path("data/sample/interactions.csv")
    if data_path.exists():
        print(f"  Loading from {data_path}")
        loader = DataLoader()
        loader.load_interactions(str(data_path))
    else:
        print("  Generating demo data...")
        demo_df = generate_demo_data()
        loader = DataLoader()
        loader.load_interactions(df=demo_df)

    print(f"  Loaded {len(loader.interactions_df)} interactions")
    print(f"  {loader.n_users} users, {loader.n_products} products")

    # Step 2: Create train/test split
    print("\n[2/5] Creating train/test split...")
    train_df, test_df = loader.train_test_split(test_ratio=0.2, by_time=True)
    print(f"  Train: {len(train_df)} interactions")
    print(f"  Test: {len(test_df)} interactions")

    # Create a loader for training data only
    train_loader = DataLoader()
    train_loader.load_interactions(df=train_df)
    train_matrix = train_loader.get_interaction_matrix(weighted=True)

    print(f"  Interaction matrix shape: {train_matrix.shape}")
    print(f"  Sparsity: {1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.2%}")

    # Step 3: Train models
    print("\n[3/5] Training models...")

    models = []

    # Popularity model
    print("  Training Popularity model...")
    pop_model = PopularityRecommender()
    pop_model.fit(train_matrix)
    models.append(pop_model)
    print("    Done!")

    # Item KNN model
    print("  Training Item-KNN model...")
    knn_model = ItemKNNRecommender(k=20)
    knn_model.fit(train_matrix)
    models.append(knn_model)
    print("    Done!")

    # ALS model (if implicit library is available)
    try:
        print("  Training ALS model...")
        als_model = ALSRecommender(factors=32, iterations=10)
        als_model.fit(train_matrix)
        models.append(als_model)
        print("    Done!")
    except ImportError:
        print("    Skipped (implicit library not installed)")

    # Step 4: Generate example recommendations
    print("\n[4/5] Generating example recommendations...")

    # Pick a random user
    sample_user_idx = 0
    sample_user_id = train_loader.idx_to_user_id[sample_user_idx]

    print(f"\n  Recommendations for user: {sample_user_id}")
    print("-" * 50)

    for model in models:
        recs = model.recommend(sample_user_idx, n=5, exclude_seen=True)
        print(f"\n  {model.name}:")
        for rank, (item_idx, score) in enumerate(recs, 1):
            product_id = train_loader.idx_to_product_id[item_idx]
            print(f"    {rank}. {product_id} (score: {score:.4f})")

    # Step 5: Evaluate models
    print("\n[5/5] Evaluating models...")

    # Build test set: user -> set of items they interacted with in test period
    test_interactions = {}
    for _, row in test_df.iterrows():
        user_id = row["user_id"]
        product_id = row["product_id"]

        # Only include users/products in training set
        if user_id in train_loader.user_id_to_idx and product_id in train_loader.product_id_to_idx:
            user_idx = train_loader.user_id_to_idx[user_id]
            item_idx = train_loader.product_id_to_idx[product_id]

            if user_idx not in test_interactions:
                test_interactions[user_idx] = set()
            test_interactions[user_idx].add(item_idx)

    print(f"  Test users with interactions: {len(test_interactions)}")

    # Evaluate each model
    for model in models:
        results = evaluate_model(
            model,
            test_interactions,
            k_values=[5, 10, 20],
            n_items=train_loader.n_products,
        )
        print_evaluation_results(results, model.name)

    # Step 6: Similar items example
    print("\n[Bonus] Similar items example:")
    print("-" * 50)

    if hasattr(knn_model, "get_similar_items"):
        sample_item_idx = 0
        sample_product_id = train_loader.idx_to_product_id[sample_item_idx]
        print(f"\n  Items similar to: {sample_product_id}")

        similar = knn_model.get_similar_items(sample_item_idx, n=5)
        for rank, (item_idx, score) in enumerate(similar, 1):
            product_id = train_loader.idx_to_product_id[item_idx]
            print(f"    {rank}. {product_id} (similarity: {score:.4f})")

    print("\n" + "=" * 60)
    print(" Demo complete! ")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Generate full sample data: python scripts/generate_sample_data.py")
    print("  2. Start the API: python -m src.api.main")
    print("  3. Visit: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
