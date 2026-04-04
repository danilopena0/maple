"""
Autoresearch training script — rewritten by the agent each iteration.

Contract
--------
• Read MAPLE_VAL_PATH from the environment (path to the pickled val split).
• Fit a recommendation model on the training portion of the split.
• Evaluate on the validation split.
• Print the following lines to stdout (the harness parses them):

    notes: <your hypothesis / what you changed and why>
    model: <ModelClassName>
    val_ndcg10: <float>

Available model classes (import from src.models.*):
    PopularityRecommender       – no hyperparams
    ItemKNNRecommender(k, min_similarity)
    UserKNNRecommender(k, min_similarity)
    ALSRecommender(factors, regularization, iterations)
    ContentBasedRecommender(use_tfidf)
    HybridRecommender(cf_model, content_model, cf_weight, strategy)
    BPRRecommender(n_factors, learning_rate, regularization, n_epochs)
    NeuralCFRecommender(embedding_dim, dropout, learning_rate, batch_size, n_epochs, n_negatives)
    EnsembleRecommender(models, weights, strategy)

Rules
-----
• Do NOT import: subprocess, socket, requests, urllib, http, ctypes, or any networking library.
• Do NOT download external data.
• Do NOT modify the harness, experiment store, or val split.
• Stay within the 5-minute wall-clock budget.
• The last printed line must be:  val_ndcg10: <float>
"""

import os
import pickle
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.sparse import csr_matrix

from src.models.popularity import PopularityRecommender
from src.evaluation.metrics import evaluate_model


def build_matrix(train_df, n_users: int, n_items: int) -> csr_matrix:
    return csr_matrix(
        (train_df["value"].values, (train_df["user_idx"].values, train_df["item_idx"].values)),
        shape=(n_users, n_items),
    )


def main() -> None:
    val_path = os.environ["MAPLE_VAL_PATH"]

    with open(val_path, "rb") as f:
        val_data = pickle.load(f)

    train_df = val_data["train_df"]
    val_interactions = val_data["val_interactions"]
    n_users = val_data["n_users"]
    n_items = val_data["n_items"]

    interaction_matrix = build_matrix(train_df, n_users, n_items)

    # ── Mutable zone ── the agent rewrites from here ─────────────────────────
    model = PopularityRecommender()
    model.fit(interaction_matrix)
    notes = "Baseline: global popularity model — no personalisation"
    # ── End mutable zone ─────────────────────────────────────────────────────

    results = evaluate_model(model, val_interactions, k_values=[10], n_items=n_items)
    val_ndcg10 = results.get("ndcg@10", 0.0)

    print(f"notes: {notes}")
    print(f"model: {model.__class__.__name__}")
    print(f"val_ndcg10: {val_ndcg10:.6f}")


if __name__ == "__main__":
    main()
