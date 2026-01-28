# Maple Model Usage Guide

This guide provides detailed documentation on how to use each recommendation model, their design parameters, and the architectural rationale behind them.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Phase 1: Baseline Models](#phase-1-baseline-models)
   - [PopularityRecommender](#popularityrecommender)
   - [ItemKNNRecommender](#itemknnrecommender)
   - [UserKNNRecommender](#userknnrecommender)
   - [ALSRecommender](#alsrecommender)
3. [Phase 2: Advanced Models](#phase-2-advanced-models)
   - [ContentBasedRecommender](#contentbasedrecommender)
   - [TFIDFRecommender](#tfidfRecommender)
   - [HybridRecommender](#hybridrecommender)
   - [FeatureAugmentedCF](#featureaugmentedcf)
   - [BPRRecommender](#bprrecommender)
   - [NeuralCFRecommender](#neuralcfrecommender)
4. [Ensemble & Re-ranking](#ensemble--re-ranking)
   - [EnsembleRecommender](#ensemblerecommender)
   - [ReRanker](#reranker)
   - [BusinessRulesFilter](#businessrulesfilter)
5. [Architecture Design Rationale](#architecture-design-rationale)
6. [Model Selection Guide](#model-selection-guide)
7. [Common Patterns](#common-patterns)

---

## Quick Start

```python
import pandas as pd
from src.data.loader import DataLoader
from src.models.collaborative import ItemKNNRecommender, ALSRecommender
from src.models.popularity import PopularityRecommender
from src.evaluation.metrics import evaluate_model

# 1. Load data
loader = DataLoader()
loader.load_interactions(interactions_df)  # DataFrame with user_id, product_id, interaction_type

# 2. Create train/test split
train_df, test_df = loader.train_test_split(test_ratio=0.2)
train_loader = DataLoader()
train_loader.load_interactions(train_df)

# 3. Get interaction matrix
interaction_matrix = train_loader.get_interaction_matrix()

# 4. Train model
model = ItemKNNRecommender(k=50)
model.fit(interaction_matrix)

# 5. Generate recommendations
user_idx = loader.user_to_idx["user_001"]
recommendations = model.recommend(user_idx, n=10)

# 6. Evaluate
results = evaluate_model(model, test_df, loader, k_values=[5, 10, 20])
```

---

## Phase 1: Baseline Models

### PopularityRecommender

**What it does**: Recommends the most popular items based on interaction counts. All users receive the same recommendations (non-personalized).

**When to use**:
- As a baseline to compare against
- For cold-start users with no history
- When you need fast, simple recommendations
- As a fallback when other models fail

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "popularity" | Model identifier |

#### Usage

```python
from src.models.popularity import PopularityRecommender

# Initialize
model = PopularityRecommender()

# Fit
model.fit(interaction_matrix)

# Get recommendations for any user
recommendations = model.recommend(user_idx=0, n=10)
# Returns: [(item_idx, count), ...]

# Get globally popular items
popular = model.get_popular_items(n=20)
```

#### Design Rationale

Popularity-based recommendation is the simplest possible baseline. Despite its simplicity, it often performs surprisingly well because:

1. **Exploits power law**: In most domains, a small number of items receive the majority of interactions
2. **No cold-start**: Works for new users immediately
3. **Computationally trivial**: O(n log n) for sorting items by count

**Limitations**: No personalization, creates filter bubbles, ignores user preferences entirely.

---

### ItemKNNRecommender

**What it does**: Finds items similar to what the user has interacted with using cosine similarity between item vectors (columns of the interaction matrix).

**When to use**:
- When you have dense interaction data
- For "similar items" features
- When interpretability matters ("because you bought X")
- Small to medium catalogs (< 100K items)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "item_knn" | Model identifier |
| `k` | int | 50 | Number of similar items to consider when scoring |
| `min_similarity` | float | 0.0 | Minimum similarity threshold (filters weak similarities) |

#### Usage

```python
from src.models.collaborative import ItemKNNRecommender

# Initialize with parameters
model = ItemKNNRecommender(
    k=50,              # Consider top 50 similar items
    min_similarity=0.1  # Ignore similarities below 0.1
)

# Fit computes item-item similarity matrix
model.fit(interaction_matrix)

# Recommend for a user
recommendations = model.recommend(
    user_idx=5,
    n=10,
    exclude_seen=True  # Don't recommend items user already interacted with
)

# Get similar items (for "customers also bought" feature)
similar = model.get_similar_items(item_idx=42, n=5)
# Returns: [(similar_item_idx, similarity_score), ...]
```

#### Parameter Tuning Guide

| Parameter | Low Value Effect | High Value Effect | Recommended Range |
|-----------|-----------------|-------------------|-------------------|
| `k` | More focused, less diverse | More diverse, potentially noisy | 20-100 |
| `min_similarity` | Include weak signals | Stricter, may miss items | 0.0-0.2 |

#### Design Rationale

Item-KNN uses the principle that **similar items are liked by similar users**. The similarity is computed from co-interaction patterns:

```
similarity(item_a, item_b) = cosine(users_who_liked_a, users_who_liked_b)
```

**Architecture decisions**:

1. **Cosine similarity over Pearson**: Cosine handles implicit feedback (binary/counts) better than Pearson which assumes ratings
2. **k parameter**: Limits computation and acts as regularization; without it, one interaction could influence all items
3. **Pre-computed similarity matrix**: O(items²) space but enables O(1) similarity lookups

**Trade-offs**:
- **Pro**: Stable similarities (items don't change as fast as users)
- **Pro**: Natural "similar items" feature
- **Con**: O(items²) memory for similarity matrix
- **Con**: Cold-start for new items (no interactions = no similarity)

---

### UserKNNRecommender

**What it does**: Finds users similar to the target user and recommends items those similar users liked.

**When to use**:
- When user behavior patterns are informative
- Small to medium user bases (< 100K users)
- When "users like you" narrative is valuable

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "user_knn" | Model identifier |
| `k` | int | 50 | Number of similar users to consider |
| `min_similarity` | float | 0.0 | Minimum similarity threshold |

#### Usage

```python
from src.models.collaborative import UserKNNRecommender

model = UserKNNRecommender(k=30, min_similarity=0.05)
model.fit(interaction_matrix)

# Recommend based on similar users' preferences
recommendations = model.recommend(user_idx=10, n=10)
```

#### Design Rationale

User-KNN operates on the principle that **similar users like similar items**:

```
score(user_a, item) = Σ similarity(user_a, user_b) × interaction(user_b, item)
                      for user_b in top-k similar users
```

**Why User-KNN is less common than Item-KNN**:

1. **User drift**: User preferences change over time; item characteristics are more stable
2. **Scalability**: Typically more users than items, making O(users²) more expensive
3. **Sparsity**: User vectors are often sparser than item vectors

**When User-KNN excels**:
- Domains with stable user preferences (e.g., music taste)
- Small, tight-knit communities
- When user similarity itself is valuable (social features)

---

### ALSRecommender

**What it does**: Matrix factorization using Alternating Least Squares. Learns latent embeddings for users and items that explain the interaction patterns.

**When to use**:
- Implicit feedback data (clicks, views, purchases)
- Medium to large datasets
- When you need user/item embeddings for downstream tasks
- Better quality than KNN with sufficient data

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "als" | Model identifier |
| `factors` | int | 64 | Dimension of latent embeddings |
| `regularization` | float | 0.01 | L2 regularization to prevent overfitting |
| `iterations` | int | 15 | Number of ALS optimization iterations |
| `random_state` | int | 42 | Random seed for reproducibility |

#### Usage

```python
from src.models.collaborative import ALSRecommender

model = ALSRecommender(
    factors=64,          # 64-dimensional embeddings
    regularization=0.01, # L2 regularization
    iterations=15        # Training iterations
)
model.fit(interaction_matrix)

# Recommendations
recommendations = model.recommend(user_idx=5, n=10)

# Access learned embeddings
user_embedding = model.get_user_embedding(user_idx=5)  # Shape: (64,)
item_embedding = model.get_item_embedding(item_idx=10) # Shape: (64,)

# Similar items in embedding space
similar = model.get_similar_items(item_idx=42, n=5)
```

#### Parameter Tuning Guide

| Parameter | Low Value | High Value | Typical Range |
|-----------|-----------|------------|---------------|
| `factors` | Underfitting, simple patterns | Overfitting, captures noise | 32-128 |
| `regularization` | May overfit | Underfitting | 0.001-0.1 |
| `iterations` | Underfitting | Diminishing returns | 10-30 |

**Tuning strategy**:
1. Start with `factors=64, regularization=0.01, iterations=15`
2. Increase `factors` if underfitting (low training accuracy)
3. Increase `regularization` if overfitting (train >> test performance)
4. Monitor convergence; stop when loss plateaus

#### Design Rationale

ALS decomposes the interaction matrix R into user factors U and item factors V:

```
R ≈ U × V^T

where:
  R[u,i] = predicted score for user u, item i
  U[u] = user u's latent factors (embedding)
  V[i] = item i's latent factors (embedding)
```

**Why ALS for implicit feedback**:

1. **Handles missing data**: Unlike SVD, ALS treats missing entries as zeros with confidence weighting, not as missing values
2. **Parallelizable**: Alternating optimization allows embarrassingly parallel updates
3. **Scalable**: The `implicit` library uses efficient sparse matrix operations

**The alternating optimization**:
```
While not converged:
    Fix V, solve for optimal U  (closed-form solution)
    Fix U, solve for optimal V  (closed-form solution)
```

**Implicit feedback weighting**:
```
confidence[u,i] = 1 + α × interaction_count[u,i]
```
More interactions = higher confidence that the user prefers the item.

---

## Phase 2: Advanced Models

### ContentBasedRecommender

**What it does**: Recommends items similar in content/features to what the user has liked, regardless of what other users think.

**When to use**:
- Cold-start items (new products with no interactions)
- When rich item metadata is available
- Domains where content similarity implies preference similarity
- As a component in hybrid systems

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "content_based" | Model identifier |
| `text_features` | list[str] | None | Column names for text features |
| `categorical_features` | list[str] | None | Column names for categorical features |
| `use_tfidf` | bool | True | Use TF-IDF (True) or raw counts (False) |

#### Usage

```python
from src.models.content_based import ContentBasedRecommender
import pandas as pd

# Prepare item features DataFrame
item_features_df = pd.DataFrame({
    'description': ['Blue cotton t-shirt', 'Red wool sweater', ...],
    'category': ['clothing', 'clothing', ...],
    'brand': ['Nike', 'Adidas', ...]
})

# Initialize
model = ContentBasedRecommender(
    text_features=['description'],
    categorical_features=['category', 'brand']
)

# Fit requires both interaction matrix and item features
model.fit(
    interaction_matrix,
    item_features_df=item_features_df
)

# Recommend items similar to user's past preferences
recommendations = model.recommend(user_idx=5, n=10)

# Find similar items by content
similar = model.get_similar_items(item_idx=42, n=5)

# Get item's feature vector
features = model.get_item_features(item_idx=42)
```

#### Design Rationale

Content-based filtering solves the cold-start problem by using item attributes instead of interaction patterns:

```
similarity(item_a, item_b) = cosine(features_a, features_b)
```

**Feature engineering pipeline**:

1. **Text features → TF-IDF**: Converts text to numerical vectors where:
   - TF (Term Frequency): How often a word appears in this item
   - IDF (Inverse Document Frequency): Downweights common words across all items

2. **Categorical features → One-hot encoding**: Converts categories to binary vectors
   - `category='shoes'` → `[0, 0, 1, 0, 0]`

3. **Feature combination**: Horizontally stacks all feature vectors

4. **L2 normalization**: Ensures all items have unit-length feature vectors for fair comparison

**Why this architecture**:

- **TF-IDF over raw counts**: Prevents common words ("the", "product") from dominating similarity
- **One-hot over embeddings**: Simpler, more interpretable, works with small category sets
- **Cosine similarity**: Scale-invariant, works well with sparse high-dimensional vectors

---

### TFIDFRecommender

**What it does**: Lightweight content-based recommender using only text descriptions.

**When to use**:
- When only text descriptions are available
- Quick prototyping
- When text is the primary differentiator between items

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "tfidf" | Model identifier |
| `max_features` | int | 5000 | Maximum vocabulary size |
| `ngram_range` | tuple | (1, 2) | N-gram range for tokenization |

#### Usage

```python
from src.models.content_based import TFIDFRecommender

# Prepare text descriptions (one per item)
item_texts = [
    "Blue cotton t-shirt comfortable casual wear",
    "Red wool sweater warm winter clothing",
    # ... one text per item
]

model = TFIDFRecommender(
    max_features=5000,      # Vocabulary size
    ngram_range=(1, 2)      # Include bigrams
)

model.fit(interaction_matrix, item_texts=item_texts)
recommendations = model.recommend(user_idx=5, n=10)
```

#### Parameter Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `max_features` | Higher = more vocabulary, slower | 1000-10000 |
| `ngram_range` | (1,1) = unigrams only, (1,2) = + bigrams | (1,2) for most cases |

---

### HybridRecommender

**What it does**: Combines collaborative filtering with content-based recommendations using configurable strategies.

**When to use**:
- To get the best of both CF and content-based
- Handling cold-start while maintaining personalization
- When you have both interaction data and item features

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cf_model` | BaseRecommender | required | Fitted collaborative filtering model |
| `content_model` | BaseRecommender | required | Fitted content-based model |
| `name` | str | "hybrid" | Model identifier |
| `cf_weight` | float | 0.7 | Weight for CF scores (normalized) |
| `content_weight` | float | 0.3 | Weight for content scores (normalized) |
| `strategy` | str | "weighted" | Combination strategy |
| `cold_start_threshold` | int | 5 | Min interactions for CF (switching strategy) |

#### Strategies Explained

**1. Weighted** (`strategy="weighted"`):
```
final_score = cf_weight × normalize(cf_score) + content_weight × normalize(content_score)
```
Best for: General use, when both signals are valuable

**2. Switching** (`strategy="switching"`):
```
if user_interactions >= cold_start_threshold:
    use cf_model
else:
    use content_model
```
Best for: Explicit cold-start handling

**3. Cascade** (`strategy="cascade"`):
```
candidates = content_model.recommend(n=n*5)  # Broad candidates
final = cf_model.score(candidates)            # Precise ranking
```
Best for: Large catalogs where CF is expensive

#### Usage

```python
from src.models.collaborative import ItemKNNRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender

# Train component models first
cf_model = ItemKNNRecommender(k=50)
cf_model.fit(interaction_matrix)

content_model = ContentBasedRecommender(
    text_features=['description'],
    categorical_features=['category']
)
content_model.fit(interaction_matrix, item_features_df=item_df)

# Create hybrid
hybrid = HybridRecommender(
    cf_model=cf_model,
    content_model=content_model,
    cf_weight=0.7,
    content_weight=0.3,
    strategy="weighted"
)
hybrid.fit(interaction_matrix)

# Recommendations blend both signals
recommendations = hybrid.recommend(user_idx=5, n=10)
```

#### Design Rationale

Hybrid models address the fundamental trade-off:
- **CF**: Great personalization, but cold-start problem
- **Content**: No cold-start, but less personalized

**Score normalization**: Before combining, scores are normalized to [0, 1]:
```python
normalized = (score - min_score) / (max_score - min_score)
```
This ensures fair weighting regardless of score scales.

**Strategy selection guide**:
| Scenario | Recommended Strategy |
|----------|---------------------|
| General purpose | `weighted` with cf_weight=0.7 |
| Many new users | `switching` with threshold=5-10 |
| Large catalog | `cascade` for efficiency |

---

### FeatureAugmentedCF

**What it does**: Matrix factorization that incorporates item features directly into the embeddings, similar to LightFM.

**When to use**:
- When item features can enhance CF
- Better cold-start handling than pure CF
- When you want end-to-end learned feature weights

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "feature_augmented_cf" | Model identifier |
| `n_factors` | int | 64 | Latent factor dimension |
| `feature_weight` | float | 0.3 | Weight for feature-based component |

#### Training Parameters (passed to fit)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `item_features` | np.ndarray | None | Item feature matrix (n_items × n_features) |
| `n_iterations` | int | 10 | SGD iterations |
| `learning_rate` | float | 0.01 | SGD learning rate |
| `regularization` | float | 0.01 | L2 regularization |

#### Usage

```python
from src.models.hybrid import FeatureAugmentedCF
import numpy as np

# Prepare item features as numpy array
item_features = np.array([
    [1, 0, 0, 0.5],  # Item 0: one-hot category + price
    [0, 1, 0, 0.3],  # Item 1
    # ...
])

model = FeatureAugmentedCF(
    n_factors=64,
    feature_weight=0.3
)

model.fit(
    interaction_matrix,
    item_features=item_features,
    n_iterations=20,
    learning_rate=0.01
)

recommendations = model.recommend(user_idx=5, n=10)
```

#### Design Rationale

The model computes:
```
item_embedding = (1 - feature_weight) × learned_factors + feature_weight × (features @ feature_factors)
score(user, item) = user_embedding · item_embedding
```

This allows:
1. **New items**: Even without interactions, they have embeddings from features
2. **Feature learning**: The model learns which features matter for recommendations
3. **Graceful degradation**: As interactions accumulate, learned factors dominate

---

### BPRRecommender

**What it does**: Bayesian Personalized Ranking optimizes for correct pairwise ordering: positive items should score higher than negative items.

**When to use**:
- Implicit feedback (clicks, purchases, views)
- When ranking order matters more than absolute scores
- As an alternative to ALS with different optimization objective

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "bpr" | Model identifier |
| `n_factors` | int | 64 | Latent factor dimension |
| `learning_rate` | float | 0.01 | SGD learning rate |
| `regularization` | float | 0.01 | L2 regularization |
| `n_epochs` | int | 20 | Training epochs |
| `n_samples` | int | 100000 | Samples per epoch |

#### Usage

```python
from src.models.neural import BPRRecommender

model = BPRRecommender(
    n_factors=64,
    learning_rate=0.01,
    regularization=0.01,
    n_epochs=20,
    n_samples=100000
)

model.fit(interaction_matrix)
recommendations = model.recommend(user_idx=5, n=10)

# Similar items
similar = model.get_similar_items(item_idx=42, n=5)
```

#### Design Rationale

BPR optimizes the following objective:

```
For each (user, positive_item, negative_item) triple:
    maximize: sigmoid(score(user, positive) - score(user, negative))
```

**Why BPR over ALS**:

| Aspect | ALS | BPR |
|--------|-----|-----|
| Objective | Minimize squared error | Maximize correct ranking |
| Training | Closed-form alternating | Stochastic gradient descent |
| Implicit feedback | Confidence weighting | Pairwise comparison |
| Computation | Faster per iteration | More iterations, but flexible |

**Negative sampling**: BPR samples negative items (items user hasn't interacted with) to create training triples. This is why `n_samples` matters - more samples = better gradient estimates.

---

### NeuralCFRecommender

**What it does**: Deep learning model combining linear (GMF) and non-linear (MLP) interaction modeling.

**When to use**:
- Large datasets (>1M interactions)
- When you have GPU resources
- When non-linear interaction patterns exist
- Research/experimentation

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "ncf" | Model identifier |
| `embedding_dim` | int | 32 | Embedding dimension |
| `mlp_layers` | list[int] | [64, 32, 16] | MLP hidden layer sizes |
| `dropout` | float | 0.2 | Dropout rate |
| `learning_rate` | float | 0.001 | Adam learning rate |
| `batch_size` | int | 256 | Training batch size |
| `n_epochs` | int | 10 | Training epochs |
| `n_negatives` | int | 4 | Negative samples per positive |
| `device` | str | None | "cuda" or "cpu" (auto-detected) |

#### Usage

```python
from src.models.neural import NeuralCFRecommender

# Requires PyTorch
model = NeuralCFRecommender(
    embedding_dim=32,
    mlp_layers=[64, 32, 16],
    dropout=0.2,
    learning_rate=0.001,
    batch_size=256,
    n_epochs=10,
    n_negatives=4
)

model.fit(interaction_matrix)
recommendations = model.recommend(user_idx=5, n=10)

# Access embeddings
user_emb = model.get_user_embedding(user_idx=5)
item_emb = model.get_item_embedding(item_idx=42)
```

#### Architecture

```
                    ┌─────────────┐
                    │   Sigmoid   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   Linear    │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │        Concatenate      │
              └────────────┬────────────┘
                    ┌──────┴──────┐
         ┌──────────┤             ├──────────┐
         │          │             │          │
    ┌────┴────┐     │         ┌───┴───┐     │
    │   GMF   │     │         │  MLP  │     │
    │ (u * v) │     │         │Layers │     │
    └────┬────┘     │         └───┬───┘     │
         │          │             │          │
    ┌────┴────┐     │         ┌───┴───┐     │
    │User Emb │     │         │ Concat│     │
    │  (GMF)  │     │         └───┬───┘     │
    └────┬────┘     │       ┌─────┴─────┐   │
         │          │       │           │   │
    ┌────┴────┐     │  ┌────┴────┐ ┌────┴────┐
    │Item Emb │     │  │User Emb │ │Item Emb │
    │  (GMF)  │     │  │  (MLP)  │ │  (MLP)  │
    └─────────┘     │  └─────────┘ └─────────┘
                    │
               [User ID]                [Item ID]
```

#### Design Rationale

NCF combines two complementary approaches:

**1. GMF (Generalized Matrix Factorization)**:
```python
gmf_output = user_embedding * item_embedding  # Element-wise
```
- Captures linear interactions (like matrix factorization)
- Fast, interpretable

**2. MLP (Multi-Layer Perceptron)**:
```python
mlp_input = concat(user_embedding, item_embedding)
mlp_output = mlp_layers(mlp_input)
```
- Captures non-linear interactions
- Can learn complex patterns

**Final prediction**:
```python
final = sigmoid(linear(concat(gmf_output, mlp_output)))
```

**Separate embeddings**: GMF and MLP have different embedding spaces because:
- GMF needs embeddings aligned for element-wise product
- MLP can learn arbitrary transformations
- Separate embeddings give more flexibility

**Negative sampling**: Training uses 4 negative samples per positive to create balanced binary classification (interacted vs. not interacted).

---

## Ensemble & Re-ranking

### EnsembleRecommender

**What it does**: Combines predictions from multiple models to get more robust recommendations.

**When to use**:
- Combining models with different strengths
- Reducing variance in recommendations
- Competition scenarios (often wins competitions)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | list[BaseRecommender] | required | List of fitted models |
| `weights` | list[float] | None | Model weights (default: equal) |
| `name` | str | "ensemble" | Model identifier |
| `strategy` | str | "weighted_average" | Combination strategy |

#### Strategies

**1. weighted_average**: Weighted average of normalized scores
```python
ensemble = EnsembleRecommender(
    models=[model1, model2, model3],
    weights=[0.5, 0.3, 0.2],
    strategy="weighted_average"
)
```

**2. rank_average**: Average positions across models
```python
ensemble = EnsembleRecommender(
    models=[model1, model2],
    strategy="rank_average"
)
```

**3. voting**: Count how many models recommend each item
```python
ensemble = EnsembleRecommender(
    models=[model1, model2, model3],
    strategy="voting"
)
```

#### Usage

```python
from src.models.ensemble import EnsembleRecommender

# Train individual models
popularity = PopularityRecommender()
popularity.fit(interaction_matrix)

item_knn = ItemKNNRecommender(k=50)
item_knn.fit(interaction_matrix)

als = ALSRecommender(factors=64)
als.fit(interaction_matrix)

# Create ensemble
ensemble = EnsembleRecommender(
    models=[popularity, item_knn, als],
    weights=[0.2, 0.4, 0.4],  # ALS and KNN get more weight
    strategy="weighted_average"
)
ensemble.fit(interaction_matrix)

recommendations = ensemble.recommend(user_idx=5, n=10)
```

#### Strategy Selection

| Strategy | Best For | Note |
|----------|----------|------|
| `weighted_average` | Different score scales | Normalizes before combining |
| `rank_average` | Position matters | Ignores score magnitudes |
| `voting` | Agreement matters | Good for diverse model types |

---

### ReRanker

**What it does**: Post-processes recommendation lists to inject diversity, freshness, or other business objectives.

**When to use**:
- Avoiding filter bubbles (diversity)
- Promoting new items (freshness)
- Balancing relevance with exploration

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diversity_weight` | float | 0.0 | MMR diversity injection (0-1) |
| `freshness_weight` | float | 0.0 | Boost for newer items (0-1) |
| `category_diversity` | bool | False | Enforce category variety |

#### Usage

```python
from src.models.ensemble import ReRanker
import numpy as np

# Basic re-ranker
reranker = ReRanker(
    diversity_weight=0.3,      # 30% diversity vs 70% relevance
    freshness_weight=0.1,      # Slight boost for new items
    category_diversity=True    # Mix categories
)

# Get initial recommendations from any model
initial_recs = model.recommend(user_idx=5, n=50)

# Prepare metadata
item_similarity = model.item_similarity  # From ItemKNN or ContentBased
item_timestamps = np.array([...])        # Creation timestamps
item_features = {
    'categories': {0: 'electronics', 1: 'clothing', ...}
}

# Re-rank
final_recs = reranker.rerank(
    recommendations=initial_recs,
    item_similarity=item_similarity,
    item_timestamps=item_timestamps,
    item_features=item_features,
    n=10
)
```

#### Diversity Algorithms

**MMR (Maximal Marginal Relevance)**:
```
MMR_score = λ × relevance - (1-λ) × max_similarity_to_selected

where λ = 1 - diversity_weight
```

Iteratively selects items that are both relevant AND different from already selected items.

**Category diversity**: Round-robin selection across categories ensures variety:
```
Selected: [electronics, clothing, electronics, clothing, books, ...]
```

---

### BusinessRulesFilter

**What it does**: Filters recommendations based on business constraints.

**When to use**:
- Excluding out-of-stock items
- Price range filtering
- Category restrictions
- Excluding specific items

#### Usage

```python
from src.models.ensemble import (
    BusinessRulesFilter,
    in_stock_rule,
    price_range_rule,
    category_filter_rule,
    exclude_items_rule
)

# Create filter with rules
filter = BusinessRulesFilter()

# Add pre-built rules
inventory = {0: 10, 1: 0, 2: 5, ...}  # item_idx -> stock count
filter.add_rule(in_stock_rule(inventory))

prices = {0: 29.99, 1: 149.99, ...}
filter.add_rule(price_range_rule(prices, min_price=10, max_price=100))

# Add custom rule
def no_adult_content(item_idx: int, context: dict) -> bool:
    return item_idx not in context.get('adult_items', set())

filter.add_rule(no_adult_content)

# Apply filter
initial_recs = model.recommend(user_idx=5, n=20)
filtered_recs = filter.filter(
    initial_recs,
    context={'adult_items': {42, 43, 44}}
)
```

#### Pre-built Rules

| Rule | Parameters | Description |
|------|------------|-------------|
| `in_stock_rule` | inventory dict | Item must have stock > 0 |
| `price_range_rule` | prices dict, min, max | Price in range |
| `category_filter_rule` | categories dict, allowed set | Category must be allowed |
| `exclude_items_rule` | excluded set | Exclude specific items |

---

## Architecture Design Rationale

### Why This Model Hierarchy?

```
BaseRecommender (abstract)
├── PopularityRecommender    # Baseline
├── ItemKNNRecommender       # Memory-based CF
├── UserKNNRecommender       # Memory-based CF
├── ALSRecommender           # Model-based CF
├── ContentBasedRecommender  # Content-based
├── TFIDFRecommender         # Content-based
├── HybridRecommender        # Hybrid
├── FeatureAugmentedCF       # Hybrid
├── BPRRecommender           # Neural (no PyTorch)
├── NeuralCFRecommender      # Neural (PyTorch)
└── EnsembleRecommender      # Meta-model
```

**Design principles**:

1. **Common interface**: All models implement `fit()`, `recommend()`, enabling easy swapping
2. **Gradual complexity**: Start simple (popularity), add complexity as needed
3. **Optional dependencies**: Neural models gracefully degrade without PyTorch
4. **Composability**: Models can be combined (Hybrid, Ensemble)

### Scoring Conventions

All models return `list[tuple[int, float]]` where:
- `int`: Item index (not original ID)
- `float`: Score (higher = better)

Use `DataLoader` to convert between IDs and indices:
```python
item_id = loader.idx_to_product[item_idx]
user_idx = loader.user_to_idx[user_id]
```

### Memory vs. Speed Trade-offs

| Model | Training Memory | Inference Speed | When to Use |
|-------|-----------------|-----------------|-------------|
| Popularity | O(items) | O(1) | Always as baseline |
| Item-KNN | O(items²) | O(user_items × k) | < 100K items |
| User-KNN | O(users²) | O(k × items) | < 100K users |
| ALS | O(users + items) × factors | O(items × factors) | General purpose |
| Content-Based | O(items × features) | O(user_items × items) | Feature-rich items |
| BPR | O(users + items) × factors | O(items × factors) | Implicit feedback |
| NCF | O(embeddings + MLP params) | O(items × forward_pass) | Large data, GPU |

---

## Model Selection Guide

```
START
  │
  ├─▶ Do you have interaction data?
  │     │
  │     NO ──▶ Content-Based only
  │     │
  │     YES
  │       │
  │       ├─▶ Is it your first model?
  │       │     │
  │       │     YES ──▶ Start with Popularity + ItemKNN
  │       │     │
  │       │     NO
  │       │       │
  │       │       ├─▶ Do you have item features?
  │       │       │     │
  │       │       │     YES ──▶ Try HybridRecommender
  │       │       │     │
  │       │       │     NO
  │       │       │       │
  │       │       │       ├─▶ Is data implicit (clicks/views)?
  │       │       │       │     │
  │       │       │       │     YES ──▶ ALS or BPR
  │       │       │       │     │
  │       │       │       │     NO ──▶ SVD (via Surprise library)
  │       │       │       │
  │       │       ├─▶ Want better accuracy?
  │       │       │     │
  │       │       │     YES ──▶ Try EnsembleRecommender
  │       │       │     │
  │       │       │     ├─▶ Have GPU + large data?
  │       │       │           │
  │       │       │           YES ──▶ NeuralCFRecommender
  │       │       │
  │       │       ├─▶ Need diversity?
  │       │             │
  │       │             YES ──▶ Add ReRanker
  │
  END
```

---

## Common Patterns

### Pattern 1: Full Pipeline

```python
# 1. Data preparation
loader = DataLoader()
loader.load_interactions(df)
train_df, test_df = loader.train_test_split(test_ratio=0.2)

train_loader = DataLoader()
train_loader.load_interactions(train_df)
matrix = train_loader.get_interaction_matrix()

# 2. Train multiple models
popularity = PopularityRecommender()
popularity.fit(matrix)

item_knn = ItemKNNRecommender(k=50)
item_knn.fit(matrix)

als = ALSRecommender(factors=64)
als.fit(matrix)

# 3. Ensemble
ensemble = EnsembleRecommender(
    models=[popularity, item_knn, als],
    weights=[0.1, 0.4, 0.5]
)
ensemble.fit(matrix)

# 4. Re-rank for diversity
reranker = ReRanker(diversity_weight=0.2)
recs = ensemble.recommend(user_idx, n=50)
final_recs = reranker.rerank(recs, item_similarity=item_knn.item_similarity, n=10)

# 5. Business rules
filter = BusinessRulesFilter()
filter.add_rule(in_stock_rule(inventory))
final_recs = filter.filter(final_recs)
```

### Pattern 2: A/B Testing Different Models

```python
import random

def get_recommendations(user_idx: int, variant: str) -> list:
    if variant == "control":
        return popularity_model.recommend(user_idx, n=10)
    elif variant == "treatment_a":
        return als_model.recommend(user_idx, n=10)
    elif variant == "treatment_b":
        return hybrid_model.recommend(user_idx, n=10)

# Assign user to variant
variant = random.choice(["control", "treatment_a", "treatment_b"])
recs = get_recommendations(user_idx, variant)
# Log variant for analysis
```

### Pattern 3: Cold-Start Handling

```python
def recommend_with_fallback(user_idx: int, n: int = 10) -> list:
    user_interactions = interaction_matrix[user_idx].nnz

    if user_interactions == 0:
        # Brand new user: popularity
        return popularity_model.recommend(user_idx, n=n)
    elif user_interactions < 5:
        # Few interactions: content-based or hybrid switching
        return hybrid_model.recommend(user_idx, n=n)  # switching strategy
    else:
        # Active user: full CF
        return als_model.recommend(user_idx, n=n)
```

### Pattern 4: Real-time Updates

```python
# For real-time, re-compute recommendations periodically
# but cache results

import redis
import json

cache = redis.Redis()

def get_cached_recommendations(user_id: str, n: int = 10) -> list:
    cache_key = f"recs:{user_id}"
    cached = cache.get(cache_key)

    if cached:
        return json.loads(cached)[:n]

    # Compute fresh
    user_idx = loader.user_to_idx.get(user_id)
    if user_idx is None:
        recs = popularity_model.get_popular_items(n=n)
    else:
        recs = als_model.recommend(user_idx, n=n)

    # Convert to IDs and cache
    rec_ids = [(loader.idx_to_product[idx], score) for idx, score in recs]
    cache.setex(cache_key, 3600, json.dumps(rec_ids))  # 1 hour TTL

    return rec_ids
```

---

## Summary

| Model | Complexity | Data Needs | Cold-Start | Best For |
|-------|------------|------------|------------|----------|
| Popularity | Low | Any | Yes | Baseline, fallback |
| Item-KNN | Low | Sparse OK | Items: No | Similar items |
| User-KNN | Low | Sparse OK | Users: No | Small user base |
| ALS | Medium | Implicit | No | General purpose |
| Content-Based | Medium | Features | Items: Yes | Feature-rich catalogs |
| Hybrid | Medium | Both | Partial | Best of both worlds |
| BPR | Medium | Implicit | No | Ranking optimization |
| NCF | High | Large | No | Deep patterns |
| Ensemble | Varies | Multiple models | Varies | Production systems |

Start simple, measure, iterate. The best model depends on your specific data and business needs.
