# Maple Autoresearch Program

## Task
Improve the NDCG@10 of a product recommendation system on an e-commerce dataset.
The dataset contains user-product interactions (views, clicks, add_to_cart, purchases, ratings).

## Metric
`val_ndcg10` — Normalised Discounted Cumulative Gain at K=10.
Higher is better. Range: [0, 1].

## Time budget
Each experiment has a **5-minute wall-clock limit**.
Keep model complexity proportional to this budget.

## Interface contract
You must produce a valid `autoresearch/train.py` that:
1. Loads the val split from `os.environ["MAPLE_VAL_PATH"]`
2. Builds a recommendation model and fits it on the training data
3. Evaluates on the validation interactions
4. Prints these three lines to stdout (in any order, but all three must be present):
   ```
   notes: <your hypothesis and what you changed>
   model: <ModelClassName>
   val_ndcg10: <float>
   ```

## Available model classes

```python
# Collaborative filtering
from src.models.collaborative import (
    ItemKNNRecommender,    # k: int, min_similarity: float
    UserKNNRecommender,    # k: int, min_similarity: float
    ALSRecommender,        # factors: int, regularization: float, iterations: int
    BPRRecommender,        # n_factors: int, learning_rate: float, regularization: float, n_epochs: int
)

# Content-based
from src.models.content_based import (
    ContentBasedRecommender,   # use_tfidf: bool
    TFIDFRecommender,          # max_features: int
)

# Neural
from src.models.neural import NeuralCFRecommender  # embedding_dim, dropout, learning_rate, batch_size, n_epochs, n_negatives

# Hybrid / ensemble
from src.models.hybrid import HybridRecommender    # cf_model, content_model, cf_weight, strategy
from src.models.ensemble import EnsembleRecommender # models: list, weights: list, strategy: str

# Baseline
from src.models.popularity import PopularityRecommender
```

## Hyperparameter search spaces (from SEARCH_SPACES in src/tuning/optuna_tuner.py)

| Model | Parameter | Type | Range |
|---|---|---|---|
| ItemKNNRecommender | k | int | 10–200 (step 10) |
| ItemKNNRecommender | min_similarity | float | 0.0–0.3 (step 0.05) |
| UserKNNRecommender | k | int | 10–200 (step 10) |
| ALSRecommender | factors | int | 16–256 (log) |
| ALSRecommender | regularization | float | 0.001–0.1 (log) |
| ALSRecommender | iterations | int | 5–50 (step 5) |
| BPRRecommender | n_factors | int | 16–128 (log) |
| BPRRecommender | learning_rate | float | 0.001–0.1 (log) |
| BPRRecommender | regularization | float | 0.0001–0.1 (log) |
| BPRRecommender | n_epochs | int | 5–50 (step 5) |
| NeuralCFRecommender | embedding_dim | int | 8–64 (log) |
| NeuralCFRecommender | dropout | float | 0.1–0.5 |
| NeuralCFRecommender | learning_rate | float | 0.0001–0.01 (log) |
| NeuralCFRecommender | batch_size | categorical | 64, 128, 256, 512 |
| NeuralCFRecommender | n_epochs | int | 5–30 |
| HybridRecommender | cf_weight | float | 0.1–0.9 |
| HybridRecommender | strategy | categorical | weighted, switching, cascade |
| EnsembleRecommender | strategy | categorical | weighted_average, rank_average, voting |

## Constraints
- **No external data**: only use data from `MAPLE_VAL_PATH`.
- **No networking**: do not import subprocess, socket, requests, urllib, http, ctypes.
- **No disk writes** (except printing to stdout).
- Use the `build_matrix()` helper already in `train.py` to construct the sparse matrix.
- The printed `val_ndcg10` line **must be the last output line**.

## Strategy hints
- Start simple: ItemKNN or ALS tends to beat popularity by a large margin.
- Tune a single hyperparameter at a time to understand its effect.
- If you try an ensemble, combine a collaborative and content-based model.
- Note what worked and what didn't in the `notes` field — this history is shown to you.
- Coverage constraint: do not sacrifice more than 20% of catalog coverage for NDCG gains.
