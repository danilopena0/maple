# Product Recommendation ML Model Plan

## Table of Contents
1. [Fine-Tunable Models Overview](#1-fine-tunable-models-overview)
2. [Phased Development Plan](#2-phased-development-plan)
3. [Training, Testing & Validation Strategy](#3-training-testing--validation-strategy)
4. [Model Structures: Pros & Cons](#4-model-structures-pros--cons)

---

## 1. Fine-Tunable Models Overview

### 1.1 Pre-trained Models Available for Fine-Tuning

#### **Collaborative Filtering Based**

| Model | Source | License | Best For |
|-------|--------|---------|----------|
| **LightFM** | [GitHub](https://github.com/lyst/lightfm) | Apache 2.0 | Hybrid recommendations (collaborative + content) |
| **Implicit** | [GitHub](https://github.com/benfred/implicit) | MIT | Implicit feedback (clicks, views, purchases) |
| **Surprise** | [GitHub](https://github.com/NicolasHug/Surprise) | BSD | Traditional CF algorithms (SVD, KNN, NMF) |

#### **Deep Learning Based**

| Model | Source | License | Best For |
|-------|--------|---------|----------|
| **BERT4Rec** | HuggingFace/GitHub | Apache 2.0 | Sequential recommendations using transformers |
| **SASRec** | [GitHub](https://github.com/kang205/SASRec) | MIT | Self-attention sequential recommendations |
| **RecBole** | [GitHub](https://github.com/RUCAIBox/RecBole) | MIT | 90+ recommendation models unified framework |
| **Microsoft Recommenders** | [GitHub](https://github.com/microsoft/recommenders) | MIT | Production-ready implementations |
| **NVIDIA Merlin** | [GitHub](https://github.com/NVIDIA-Merlin) | Apache 2.0 | GPU-accelerated recommendations |

#### **Embedding/Representation Learning**

| Model | Source | License | Best For |
|-------|--------|---------|----------|
| **Sentence-BERT** | HuggingFace | Apache 2.0 | Product description embeddings |
| **CLIP** | OpenAI/HuggingFace | MIT | Multi-modal (text + image) product matching |
| **Word2Vec/FastText** | Gensim | LGPL | Simple product embeddings |
| **Item2Vec** | Custom implementation | - | Product co-occurrence embeddings |

#### **Graph Neural Networks**

| Model | Source | License | Best For |
|-------|--------|---------|----------|
| **PinSage** | PyTorch Geometric | MIT | Large-scale graph recommendations |
| **LightGCN** | [GitHub](https://github.com/gusye1234/LightGCN-PyTorch) | MIT | Simplified graph convolution for CF |
| **GraphSAGE** | PyTorch Geometric | MIT | Inductive node embeddings |

### 1.2 Recommended Starting Points

For a **new project starting small**, I recommend:

1. **Phase 1**: Start with **LightFM** or **Implicit** (easiest to implement, good baselines)
2. **Phase 2**: Move to **RecBole** (unified framework for experimenting)
3. **Phase 3**: Custom deep learning with **PyTorch** using learnings from Phase 2

---

## 2. Phased Development Plan

### Phase 1: Foundation (MVP)
**Goal**: Get a working recommendation system with basic functionality

```
Timeline Milestones:
├── Data Pipeline Setup
│   ├── Data collection & storage schema
│   ├── ETL pipelines for user interactions
│   └── Basic data quality checks
│
├── Baseline Models
│   ├── Popularity-based recommendations
│   ├── Simple collaborative filtering (User-User or Item-Item KNN)
│   └── Basic matrix factorization (SVD)
│
├── Evaluation Framework
│   ├── Offline metrics (Precision@K, Recall@K, NDCG)
│   ├── A/B testing infrastructure
│   └── Logging for online metrics
│
└── Basic API
    ├── REST endpoint for recommendations
    ├── Basic caching layer
    └── Response time < 200ms target
```

**Deliverables**:
- Working API returning top-N recommendations
- Baseline metrics established
- Data pipeline operational

### Phase 2: Enhancement
**Goal**: Improve recommendation quality with more sophisticated models

```
Enhancements:
├── Advanced Models
│   ├── Hybrid approach (LightFM with content features)
│   ├── Sequential models (session-based)
│   └── Implicit feedback optimization (BPR, WARP)
│
├── Feature Engineering
│   ├── User features (demographics, behavior patterns)
│   ├── Product features (categories, attributes, embeddings)
│   ├── Contextual features (time, device, location)
│   └── Interaction features (recency, frequency)
│
├── Personalization Layers
│   ├── Cold-start handling (new users/products)
│   ├── Diversity injection
│   └── Business rules integration
│
└── Infrastructure Improvements
    ├── Model versioning (MLflow/DVC)
    ├── Feature store implementation
    └── Batch + real-time hybrid serving
```

**Deliverables**:
- 20-30% improvement over baseline metrics
- Cold-start solution implemented
- Feature store operational

### Phase 3: Scale & Sophisttic Models
**Goal**: Production-grade system with advanced capabilities

```
Advanced Capabilities:
├── Deep Learning Models
│   ├── Two-tower architecture (retrieval)
│   ├── Transformer-based ranking
│   ├── Multi-task learning
│   └── Graph neural networks
│
├── Multi-Stage Pipeline
│   ├── Candidate generation (fast, broad)
│   ├── Ranking (accurate, slower)
│   ├── Re-ranking (business logic, diversity)
│   └── Post-filtering (availability, eligibility)
│
├── Advanced Features
│   ├── Multi-modal embeddings (text + images)
│   ├── Real-time personalization
│   ├── Reinforcement learning for exploration
│   └── Causal inference for debiasing
│
└── Production Excellence
    ├── Auto-scaling infrastructure
    ├── Model monitoring & drift detection
    ├── Automated retraining pipelines
    └── Shadow mode testing
```

**Deliverables**:
- Sub-100ms latency at scale
- Automated ML pipeline
- Real-time personalization

### Phase 4: Optimization & Innovation
**Goal**: Cutting-edge capabilities and continuous improvement

```
Innovation Areas:
├── Advanced Personalization
│   ├── Contextual bandits
│   ├── Slate optimization
│   └── Long-term user satisfaction modeling
│
├── Explainability
│   ├── "Why recommended" features
│   ├── User control over recommendations
│   └── Transparency dashboards
│
├── Cross-Domain
│   ├── Transfer learning across categories
│   ├── Multi-marketplace recommendations
│   └── Cross-device user stitching
│
└── Emerging Techniques
    ├── LLM-powered recommendations
    ├── Generative recommendations
    └── Conversational recommendations
```

---

## 3. Training, Testing & Validation Strategy

### 3.1 Data Splitting Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     TIME-BASED SPLITTING                        │
│  (Recommended for recommendation systems)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   TRAINING   │ │  VALIDATION  │ │    TEST      │            │
│  │   (70-80%)   │ │   (10-15%)   │ │   (10-15%)   │            │
│  │              │ │              │ │              │            │
│  │  Historical  │ │   Recent     │ │   Most       │            │
│  │  Data        │ │   Data       │ │   Recent     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│        ↑                ↑                ↑                      │
│     t_start          t_val            t_test           t_now    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Split Strategies Comparison

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Time-based** | Production simulation | Realistic, prevents leakage | May miss seasonal patterns |
| **User-based** | Per-user evaluation | Tests generalization | Users in train/test may interact with same items |
| **Leave-one-out** | Sequential models | Simple, common benchmark | Only tests last interaction |
| **K-fold CV** | Small datasets | Robust estimates | Computationally expensive, temporal leakage risk |

### 3.2 Training Strategy

#### Batch Training Pipeline

```python
# Pseudocode for training pipeline
class TrainingPipeline:
    def __init__(self):
        self.stages = [
            DataValidation(),      # Check data quality
            FeatureEngineering(),  # Create features
            ModelTraining(),       # Train model
            ModelValidation(),     # Validate performance
            ModelRegistry(),       # Version and store
        ]

    def run(self, config):
        for stage in self.stages:
            result = stage.execute(config)
            if not result.success:
                self.alert_and_rollback()
```

#### Training Configurations by Phase

| Phase | Batch Size | Learning Rate | Epochs | Early Stopping |
|-------|------------|---------------|--------|----------------|
| Phase 1 | N/A (traditional ML) | N/A | N/A | N/A |
| Phase 2 | 256-1024 | 1e-3 to 1e-4 | 10-50 | patience=5 |
| Phase 3 | 1024-4096 | 1e-4 to 1e-5 | 20-100 | patience=10 |
| Phase 4 | 4096+ | Scheduled | 50-200 | patience=15 |

#### Hyperparameter Tuning Strategy

```
Recommended Approach: Bayesian Optimization

Tools:
├── Optuna (recommended - easy to use, efficient)
├── Ray Tune (for distributed tuning)
├── Weights & Biases Sweeps
└── SigOpt (enterprise)

Search Strategy:
1. Start with random search (20-50 trials)
2. Switch to Bayesian optimization
3. Use early stopping (pruning) for efficiency
4. Focus on high-impact parameters first:
   - Embedding dimension
   - Learning rate
   - Number of layers
   - Regularization strength
```

### 3.3 Evaluation Metrics

#### Offline Metrics

| Metric | Formula/Description | When to Use |
|--------|---------------------|-------------|
| **Precision@K** | Relevant items in top K / K | When false positives are costly |
| **Recall@K** | Relevant items in top K / Total relevant | When coverage matters |
| **NDCG@K** | Normalized Discounted Cumulative Gain | When ranking order matters |
| **MAP** | Mean Average Precision | Overall ranking quality |
| **MRR** | Mean Reciprocal Rank | When first relevant item matters |
| **Hit Rate@K** | Users with ≥1 hit in top K / Total users | Simple engagement proxy |
| **Coverage** | Unique items recommended / Total items | Catalog exploration |
| **Diversity** | Avg. pairwise distance in recommendations | Avoiding filter bubbles |
| **Novelty** | Avg. popularity rank of recommendations | Discovering long-tail |

#### Online Metrics (A/B Testing)

| Metric | Description | Target |
|--------|-------------|--------|
| **CTR** | Click-through rate | Primary engagement |
| **Conversion Rate** | Purchases / Impressions | Primary business |
| **Revenue per Session** | Total revenue / Sessions | Business impact |
| **Add-to-Cart Rate** | Add to cart / Impressions | Intent signal |
| **Time to First Click** | Engagement speed | UX quality |
| **Return Rate** | Products returned | Quality check |
| **Long-term Retention** | 30/60/90 day retention | Sustainable value |

### 3.4 Validation Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. OFFLINE VALIDATION                                          │
│     ├── Hold-out test set evaluation                            │
│     ├── Cross-validation (if appropriate)                       │
│     ├── Stratified evaluation (by user activity level)         │
│     └── Counterfactual evaluation (IPS weighting)              │
│                                                                  │
│  2. SHADOW MODE                                                  │
│     ├── Run new model alongside production                      │
│     ├── Log predictions without serving                         │
│     ├── Compare against production model                        │
│     └── Monitor latency and resource usage                      │
│                                                                  │
│  3. A/B TESTING                                                  │
│     ├── Statistical significance (p < 0.05)                     │
│     ├── Minimum detectable effect calculation                   │
│     ├── Sample size determination                               │
│     ├── Guardrail metrics monitoring                            │
│     └── Segment analysis (new vs returning users)              │
│                                                                  │
│  4. CONTINUOUS MONITORING                                        │
│     ├── Model drift detection                                   │
│     ├── Data drift detection                                    │
│     ├── Performance degradation alerts                          │
│     └── Automated rollback triggers                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Testing Checklist

#### Pre-Training Tests
- [ ] Data schema validation
- [ ] Missing value handling
- [ ] Duplicate detection
- [ ] Feature distribution checks
- [ ] Label leakage detection

#### During Training Tests
- [ ] Loss convergence monitoring
- [ ] Gradient health checks (no NaN/Inf)
- [ ] Learning rate schedule verification
- [ ] Memory usage monitoring
- [ ] Checkpoint saving verification

#### Post-Training Tests
- [ ] Metric threshold checks
- [ ] Prediction distribution analysis
- [ ] Latency benchmarks
- [ ] Edge case testing (cold users, new items)
- [ ] Fairness audits (if applicable)

#### Pre-Deployment Tests
- [ ] Integration tests with API
- [ ] Load testing
- [ ] Fallback mechanism verification
- [ ] Rollback procedure test
- [ ] Monitoring dashboard verification

---

## 4. Model Structures: Pros & Cons

### 4.1 Traditional Collaborative Filtering

#### User-User KNN
```
Structure: Find similar users, recommend their items

User A ──similarity──► User B ──liked──► Item X
                                            │
                              Recommend to User A
```

| Pros | Cons |
|------|------|
| Simple to understand and implement | Doesn't scale (O(n²) users) |
| No training required | Sparse user-item matrix problems |
| Good for small datasets | Cold-start for new users |
| Interpretable ("users like you bought...") | Can't incorporate item features |

#### Item-Item KNN
```
Structure: Find similar items based on co-interactions

Item A ──co-purchased with──► Item B
   │
User liked A ──► Recommend B
```

| Pros | Cons |
|------|------|
| More scalable than user-user | Still O(n²) items |
| Item similarities stable over time | Cold-start for new items |
| Good for "similar items" use case | Popularity bias |
| Pre-computable | Limited personalization |

### 4.2 Matrix Factorization

#### SVD / SVD++
```
Structure: Decompose user-item matrix into latent factors

R ≈ U × V^T

User Matrix (U)     Item Matrix (V)
[u1_f1 u1_f2 ...]   [i1_f1 i1_f2 ...]
[u2_f1 u2_f2 ...]   [i2_f1 i2_f2 ...]
[...            ]   [...            ]
```

| Pros | Cons |
|------|------|
| Captures latent patterns | Requires explicit ratings |
| Handles sparsity well | Cold-start problem |
| Computationally efficient | Linear interactions only |
| Well-understood theory | Hyperparameter sensitive |

#### ALS (Alternating Least Squares)
```
Structure: Iteratively optimize user/item factors

While not converged:
    Fix V, optimize U
    Fix U, optimize V
```

| Pros | Cons |
|------|------|
| Works with implicit feedback | Requires careful regularization |
| Parallelizable | Convergence can be slow |
| Handles large scale | Memory intensive |
| No negative sampling needed | Limited expressiveness |

### 4.3 Deep Learning Models

#### Neural Collaborative Filtering (NCF)
```
Structure:
                    ┌─────────────┐
                    │   Output    │
                    │   Layer     │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │    MLP      │
                    │   Layers    │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────┴─────┐             ┌─────┴─────┐
        │   User    │             │   Item    │
        │ Embedding │             │ Embedding │
        └─────┬─────┘             └─────┬─────┘
              │                         │
         [User ID]                 [Item ID]
```

| Pros | Cons |
|------|------|
| Learns non-linear interactions | Requires more data |
| End-to-end trainable | Longer training time |
| Can incorporate features | Hyperparameter tuning needed |
| Flexible architecture | Prone to overfitting |

#### Two-Tower Architecture (Retrieval)
```
Structure:
        ┌─────────────┐
        │  Similarity │
        │   (dot/cos) │
        └──────┬──────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───┴───┐            ┌────┴────┐
│ User  │            │  Item   │
│ Tower │            │  Tower  │
│ (DNN) │            │  (DNN)  │
└───┬───┘            └────┬────┘
    │                     │
[User Features]     [Item Features]
```

| Pros | Cons |
|------|------|
| Efficient retrieval (ANN) | Limited interaction modeling |
| Scales to millions of items | Requires negative sampling |
| Separate training possible | Embedding space alignment |
| Industry standard | May miss fine-grained patterns |

#### Sequential Models (SASRec/BERT4Rec)
```
Structure (BERT4Rec):
        ┌─────────────────────────────────────┐
        │        Transformer Layers           │
        │  (Self-Attention + Feed-Forward)    │
        └─────────────────┬───────────────────┘
                          │
        ┌─────────────────┴───────────────────┐
        │          Position Encoding          │
        └─────────────────┬───────────────────┘
                          │
    ┌─────────┬───────────┼───────────┬─────────┐
    │         │           │           │         │
  [Item1]  [Item2]     [MASK]     [Item4]    [Item5]
    ↓         ↓           ↓           ↓         ↓
  Past behavior sequence    Predict next item
```

| Pros | Cons |
|------|------|
| Captures sequential patterns | Computationally expensive |
| Handles variable-length sequences | Requires sequence data |
| Self-attention is powerful | May overfit on short sequences |
| Pre-training possible | Position encoding choices matter |

### 4.4 Graph Neural Networks

#### LightGCN
```
Structure:
Layer 0:    User/Item Embeddings
               │
Layer 1:    ──►│◄── Aggregate neighbor embeddings
               │
Layer 2:    ──►│◄── Aggregate 2-hop neighbors
               │
Layer K:    ──►│◄── Aggregate K-hop neighbors
               │
Final:      Layer combination (mean/concat)
```

| Pros | Cons |
|------|------|
| Captures graph structure | Requires graph construction |
| Multi-hop information | Scalability challenges |
| State-of-the-art performance | Over-smoothing in deep networks |
| No feature transformation needed | Cold-start still problematic |

#### PinSage (Pinterest's Approach)
```
Structure:
- Random walk sampling for neighbors
- Importance pooling aggregation
- Curriculum learning (hard negatives)
- MapReduce for billion-scale
```

| Pros | Cons |
|------|------|
| Billion-scale capable | Complex infrastructure |
| Sampling reduces computation | Sampling introduces variance |
| Industrial proven | Requires significant engineering |
| Combines content + structure | Long training times |

### 4.5 Hybrid Approaches

#### LightFM
```
Structure:
Score(u,i) = <user_embedding, item_embedding>

Where:
user_embedding = user_bias + Σ(user_feature_embeddings)
item_embedding = item_bias + Σ(item_feature_embeddings)
```

| Pros | Cons |
|------|------|
| Handles cold-start well | Linear interactions only |
| Fast training | Limited expressiveness |
| Works with sparse features | Requires feature engineering |
| Good baseline performance | Not state-of-the-art |

#### Wide & Deep
```
Structure:
                ┌─────────────┐
                │   Output    │
                └──────┬──────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────┴─────┐             ┌─────┴─────┐
    │   Wide    │             │   Deep    │
    │ (Linear)  │             │  (DNN)    │
    └─────┬─────┘             └─────┬─────┘
          │                         │
    [Cross Features]         [Dense Features]
    (Memorization)           (Generalization)
```

| Pros | Cons |
|------|------|
| Memorization + Generalization | Feature engineering for wide |
| Handles sparse + dense | Two components to tune |
| Google-proven at scale | May be overkill for simple cases |
| Good for CTR prediction | Wide part can overfit |

### 4.6 Model Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SELECTION FLOWCHART                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Size?                                                      │
│  ├── Small (<100K interactions)                                  │
│  │   └── Start with: Item-KNN, SVD, LightFM                     │
│  │                                                               │
│  ├── Medium (100K - 10M interactions)                           │
│  │   └── Consider: ALS, NCF, LightGCN                           │
│  │                                                               │
│  └── Large (>10M interactions)                                  │
│      └── Go with: Two-Tower, GNN, Transformer                   │
│                                                                  │
│  Have Sequential Data?                                          │
│  ├── Yes → SASRec, BERT4Rec, GRU4Rec                           │
│  └── No  → Matrix Factorization, GNN                           │
│                                                                  │
│  Cold-Start Important?                                          │
│  ├── Yes → LightFM, Two-Tower with features, Content-based     │
│  └── No  → Pure collaborative filtering OK                      │
│                                                                  │
│  Have Rich Features?                                            │
│  ├── Yes → Wide&Deep, DeepFM, LightFM                          │
│  └── No  → NCF, LightGCN, ALS                                  │
│                                                                  │
│  Latency Requirements?                                          │
│  ├── <10ms  → Pre-computed, ANN with Two-Tower                 │
│  ├── <100ms → Light models, caching                            │
│  └── >100ms → Can use heavier ranking models                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 Recommended Architecture for Maple

Based on a typical e-commerce recommendation system, here's a suggested architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAPLE RECOMMENDATION SYSTEM                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CANDIDATE GENERATION (Fast, Broad)                             │
│  ├── Two-Tower Model (user/item embeddings)                     │
│  ├── ANN Index (FAISS/ScaNN) for fast retrieval                │
│  └── Returns: ~1000 candidates in <10ms                         │
│                                                                  │
│  RANKING (Accurate, Slower)                                     │
│  ├── Deep ranking model (Wide&Deep or DCN)                      │
│  ├── Rich features: user context, item attributes, history     │
│  └── Returns: Scored and sorted candidates                      │
│                                                                  │
│  RE-RANKING (Business Logic)                                    │
│  ├── Diversity injection                                        │
│  ├── Business rules (inventory, margins, promotions)           │
│  ├── Freshness boost                                            │
│  └── Returns: Final recommendation list                         │
│                                                                  │
│  SERVING                                                         │
│  ├── Feature store (Feast/Tecton)                              │
│  ├── Model serving (TensorFlow Serving/Triton)                 │
│  ├── Caching layer (Redis)                                      │
│  └── Fallback: Popularity-based recommendations                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Aspect | Phase 1 Recommendation | Long-term Goal |
|--------|------------------------|----------------|
| **Model** | LightFM or Implicit ALS | Two-Tower + Deep Ranker |
| **Framework** | Surprise/LightFM | RecBole → Custom PyTorch |
| **Validation** | Time-based split, NDCG@10 | Full A/B testing framework |
| **Training** | Weekly batch | Daily batch + real-time updates |
| **Infrastructure** | Single server | Distributed, auto-scaling |

### Next Steps

1. **Data Audit**: Assess available data (interactions, features, volume)
2. **Baseline Implementation**: Start with LightFM for quick wins
3. **Evaluation Setup**: Implement offline metrics pipeline
4. **Iterate**: Use learnings to guide Phase 2 model selection
