#!/usr/bin/env python
"""
MLflow Experiment Tracking Example

This example demonstrates how to use the MLflow tracking integration
to track experiments, compare models, and manage model versions.

Usage:
    python examples/mlflow_tracking_example.py

After running, view results with:
    mlflow ui --backend-store-uri ./mlruns

Then open http://localhost:5000 in your browser.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.loader import DataLoader
from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemKNNRecommender, ALSRecommender
from src.evaluation.metrics import evaluate_model
from src.tracking import ExperimentTracker


def generate_sample_data(n_users: int = 100, n_items: int = 50, n_interactions: int = 2000):
    """Generate sample interaction data."""
    np.random.seed(42)

    interactions = []
    for _ in range(n_interactions):
        user_id = f"user_{np.random.randint(0, n_users):03d}"
        product_id = f"prod_{np.random.randint(0, n_items):03d}"
        interaction_type = np.random.choice(
            ["view", "click", "add_to_cart", "purchase"],
            p=[0.5, 0.3, 0.15, 0.05]
        )
        interactions.append({
            "user_id": user_id,
            "product_id": product_id,
            "interaction_type": interaction_type,
        })

    return pd.DataFrame(interactions)


def main():
    print("=" * 60)
    print(" MLflow Experiment Tracking Demo")
    print("=" * 60)

    # 1. Generate and load data
    print("\n[1/5] Preparing data...")
    df = generate_sample_data()
    loader = DataLoader()
    loader.load_interactions(df)

    train_df, test_df = loader.train_test_split(test_ratio=0.2)
    train_loader = DataLoader()
    train_loader.load_interactions(train_df)
    interaction_matrix = train_loader.get_interaction_matrix()

    print(f"  Train: {len(train_df)} interactions")
    print(f"  Test: {len(test_df)} interactions")

    # 2. Initialize tracker
    print("\n[2/5] Initializing MLflow tracker...")
    tracker = ExperimentTracker(
        experiment_name="model_comparison",
        tracking_uri="./mlruns",
    )
    print(f"  Experiment: {tracker.experiment_name}")
    print(f"  Tracking URI: {tracker.tracking_uri}")

    # 3. Train and track multiple models
    print("\n[3/5] Training and tracking models...")

    models_to_train = [
        ("Popularity", PopularityRecommender()),
        ("ItemKNN_k20", ItemKNNRecommender(k=20)),
        ("ItemKNN_k50", ItemKNNRecommender(k=50)),
        ("ItemKNN_k100", ItemKNNRecommender(k=100)),
        ("ALS_f32", ALSRecommender(factors=32, iterations=10)),
        ("ALS_f64", ALSRecommender(factors=64, iterations=10)),
        ("ALS_f128", ALSRecommender(factors=128, iterations=10)),
    ]

    for run_name, model in models_to_train:
        print(f"\n  Training {run_name}...")

        with tracker.start_run(run_name=run_name, model_name=model.name) as run:
            # Train
            model.fit(interaction_matrix)

            # Log model parameters
            tracker.log_model_params(model)

            # Evaluate
            results = evaluate_model(
                model=model,
                test_interactions=test_df,
                data_loader=loader,
                k_values=[5, 10, 20],
            )

            # Log metrics
            tracker.log_evaluation_results(results)

            # Log model artifact
            tracker.log_model(model, artifact_name="model")

            # Add custom tags
            tracker.set_tags({
                "stage": "experiment",
                "dataset": "sample",
            })

            # Print summary
            print(f"    NDCG@10: {results.get('ndcg', {}).get('ndcg@10', 0):.4f}")
            print(f"    Precision@10: {results.get('precision', {}).get('precision@10', 0):.4f}")
            print(f"    Coverage: {results.get('coverage', 0):.4f}")

    # 4. Compare runs
    print("\n[4/5] Comparing runs...")
    comparison = tracker.compare_runs(metric="ndcg_10", top_n=5)

    print("\n  Top 5 models by NDCG@10:")
    print("  " + "-" * 50)
    for i, run in enumerate(comparison, 1):
        ndcg = run["metrics"].get("ndcg_10", 0)
        precision = run["metrics"].get("precision_10", 0)
        print(f"  {i}. {run['run_name']:20s} NDCG@10={ndcg:.4f}  P@10={precision:.4f}")

    # 5. Get best model
    print("\n[5/5] Best model selection...")
    best = tracker.get_best_run(metric="ndcg_10")

    if best:
        print(f"\n  Best model: {best['run_name']}")
        print(f"  Run ID: {best['run_id']}")
        print(f"  Metrics:")
        for key, value in sorted(best["metrics"].items()):
            print(f"    {key}: {value:.4f}")

        # Load best model
        print("\n  Loading best model from MLflow...")
        loaded_model = tracker.load_model(best["run_id"], "model")
        print(f"  Loaded: {loaded_model.__class__.__name__}")

        # Generate sample recommendation
        recs = loaded_model.recommend(user_idx=0, n=5)
        print(f"  Sample recommendations for user 0:")
        for item_idx, score in recs:
            print(f"    Item {item_idx}: {score:.4f}")

    print("\n" + "=" * 60)
    print(" Demo complete!")
    print("=" * 60)
    print("\nTo view experiment results in MLflow UI:")
    print("  1. Run: mlflow ui --backend-store-uri ./mlruns")
    print("  2. Open: http://localhost:5000")
    print("\nYou can:")
    print("  - Compare runs side-by-side")
    print("  - View parameter and metric histories")
    print("  - Download model artifacts")
    print("  - Register models to the Model Registry")


if __name__ == "__main__":
    main()
