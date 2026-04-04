"""Create and manage the fixed validation split used across all iterations."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix

from src.evaluation.metrics import evaluate_model


def create_val_split(
    data_path: str,
    val_path: str,
    val_ratio: float = 0.2,
) -> None:
    """
    Create a time-based validation split from the interactions CSV and save it.

    The split is deterministic (no random seed needed — purely time-based).
    Once created it is never regenerated so all iterations are comparable.

    The pickle contains:
        train_df          – DataFrame with user_idx, item_idx, value columns
        val_interactions  – dict[user_idx -> set[item_idx]]  (ground truth)
        n_users           – int
        n_items           – int
        user_id_to_idx    – dict[str -> int]
        item_id_to_idx    – dict[str -> int]
    """
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Build ID mappings from the full dataset so indices are stable
    unique_users = df["user_id"].unique()
    unique_items = df["product_id"].unique()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_to_idx = {pid: idx for idx, pid in enumerate(unique_items)}

    df["user_idx"] = df["user_id"].map(user_id_to_idx)
    df["item_idx"] = df["product_id"].map(item_id_to_idx)

    # Time-based split: last val_ratio of rows go to validation
    split_idx = int(len(df) * (1 - val_ratio))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    # Compute interaction weights (same logic as DataLoader)
    interaction_weights = {
        "view": 1.0,
        "click": 2.0,
        "add_to_cart": 3.0,
        "purchase": 5.0,
        "rating": 4.0,
        "review": 3.5,
        "wishlist": 2.5,
        "share": 2.0,
        "search_click": 1.5,
    }

    train_df["value"] = train_df["interaction_type"].map(interaction_weights).fillna(1.0)

    # Aggregate duplicate (user, item) pairs in train
    train_agg = (
        train_df.groupby(["user_idx", "item_idx"])["value"]
        .sum()
        .reset_index()
    )

    # Build val_interactions: only include users present in training
    train_users = set(train_agg["user_idx"].unique())
    val_interactions: dict[int, set[int]] = {}
    for user_idx, group in val_df.groupby("user_idx"):
        if user_idx in train_users:
            val_interactions[int(user_idx)] = set(group["item_idx"].astype(int).tolist())

    n_users = len(unique_users)
    n_items = len(unique_items)

    payload = {
        "train_df": train_agg,
        "val_interactions": val_interactions,
        "n_users": n_users,
        "n_items": n_items,
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
    }

    Path(val_path).parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(
        f"Val split created: {len(train_agg)} train interactions, "
        f"{len(val_interactions)} val users → {val_path}"
    )


def load_val_split(val_path: str) -> dict:
    with open(val_path, "rb") as f:
        return pickle.load(f)


def build_interaction_matrix(train_df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    return csr_matrix(
        (train_df["value"].values, (train_df["user_idx"].values, train_df["item_idx"].values)),
        shape=(n_users, n_items),
    )


def compute_val_ndcg10(
    model,
    val_interactions: dict[int, set[int]],
    n_items: int,
) -> tuple[float, dict[str, float]]:
    """
    Evaluate a fitted model on the val split.

    Returns:
        (val_ndcg10, full_metrics_dict)
    """
    metrics = evaluate_model(model, val_interactions, k_values=[10], n_items=n_items)
    return metrics.get("ndcg@10", 0.0), metrics
