# Product Recommendation Neural Network Plan

> A comprehensive guide to building a product recommendation/ranking system using free resources and PyTorch.

## Table of Contents
- [Dataset Recommendations](#dataset-recommendations)
- [Architecture Options](#architecture-options)
- [Training Strategy](#training-strategy)
- [Free Compute Resources](#free-compute-resources)
- [Evaluation Metrics](#evaluation-metrics)
- [Implementation TODOs](#implementation-todos)
- [Project Structure](#project-structure)
- [References](#references)

---

## Dataset Recommendations

### Tier 1: Best for Learning (Recommended)

#### 1. OTTO Recommender System Dataset ⭐ (Top Pick)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/otto/recsys-dataset) | [GitHub](https://github.com/otto-de/recsys-dataset)
- **Size**: 12M sessions, 220M events, 1.8M unique items
- **Data Type**: Session-based implicit feedback (clicks, carts, orders)
- **Why it's great**:
  - Real-world e-commerce data from OTTO (German retailer)
  - Multi-objective: predict clicks, cart additions, AND orders
  - Large Kaggle competition community with open solutions
  - Perfect for learning session-based and sequential recommendations
- **Enables**: Session-based models, Two-tower architectures, Sequence models (Transformers/RNNs)
- **Metrics**: Recall@20 (weighted by event type)

#### 2. Amazon Reviews 2023
- **Source**: [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | [Website](https://amazon-reviews-2023.github.io/)
- **Size**: 571M reviews, 48M items across 33 categories
- **Data Type**: Explicit ratings + text reviews + item metadata + images
- **Why it's great**:
  - Most comprehensive and recent Amazon dataset
  - Rich metadata (descriptions, prices, images, categories)
  - Text reviews enable content-based and hybrid approaches
  - Can filter by category to reduce size for learning
- **Enables**: Collaborative filtering, Content-based, Hybrid models, Multi-modal
- **Metrics**: RMSE, MAE (for ratings), NDCG, Hit Rate (for ranking)

### Tier 2: Good Alternatives

#### 3. Retailrocket E-commerce Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- **Size**: 2.7M events, 1.4M visitors, 235K items
- **Data Type**: Implicit feedback (view, addtocart, transaction) + item properties
- **Why it's great**:
  - Smaller, easier to work with while learning
  - Includes item properties and category hierarchy
  - Good for understanding implicit feedback systems
- **Enables**: Implicit collaborative filtering, Session-based models
- **Metrics**: Precision@K, Recall@K, MRR

#### 4. H&M Personalized Fashion Recommendations
- **Source**: [Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- **Size**: 31M transactions, 1.3M customers, 105K articles
- **Data Type**: Transaction history + customer metadata + article metadata + images
- **Why it's great**:
  - Fashion-specific with rich metadata
  - Images available for multi-modal approaches
  - Recent Kaggle competition with solutions
- **Enables**: Hybrid models, Image-based recommendations
- **Metrics**: MAP@12

### Dataset Selection Matrix

| Dataset | Size | Difficulty | Best For | Multi-objective |
|---------|------|------------|----------|-----------------|
| OTTO | Large | Medium | Session-based, Production-scale | ✅ Yes |
| Amazon 2023 | Huge | Medium-High | Hybrid, Content-based | ❌ No |
| Retailrocket | Medium | Easy | Beginners, Implicit feedback | ❌ No |
| H&M | Large | Medium | Fashion, Multi-modal | ❌ No |

**Recommendation**: Start with **OTTO** for the best learning experience - it has clear objectives, a supportive community, and represents real production challenges.

---

## Architecture Options

### 1. Matrix Factorization (Baseline - Train from Scratch)
```
User → Embedding(dim=64) ─┐
                          ├→ Dot Product → Prediction
Item → Embedding(dim=64) ─┘
```
- **Complexity**: Low
- **Training Time**: Minutes to hours
- **Best for**: Learning fundamentals, establishing baselines
- **PyTorch Implementation**: ~50 lines of code

### 2. Neural Collaborative Filtering (NCF)
```
User → Embedding ─┐                    ┌→ GMF (element-wise product)
                  ├→ Concatenate → MLP ┤
Item → Embedding ─┘                    └→ Output Layer → Prediction
```
- **Complexity**: Medium
- **Training Time**: Hours
- **Best for**: Implicit feedback, learning deep learning for RecSys
- **Paper**: [Neural Collaborative Filtering (2017)](https://arxiv.org/abs/1708.05031)

### 3. Two-Tower Model (Industry Standard) ⭐
```
┌─────────────────┐     ┌─────────────────┐
│   Query Tower   │     │ Candidate Tower │
│   (User/Context)│     │     (Items)     │
│                 │     │                 │
│ Features → MLP  │     │ Features → MLP  │
│      ↓          │     │      ↓          │
│  User Embedding │     │  Item Embedding │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────→ Similarity ←──┘
                      ↓
                  Ranking Score
```
- **Complexity**: Medium-High
- **Training Time**: Hours to days
- **Best for**: Large-scale retrieval, production systems
- **Library**: [TorchRec](https://pytorch.org/torchrec/)

### 4. Session-Based Models (For OTTO Dataset) ⭐
```
Session Events: [click₁, click₂, cart₁, click₃, ...]
                           ↓
              ┌────────────────────────┐
              │  GRU / Transformer     │
              │  Sequence Encoder      │
              └───────────┬────────────┘
                          ↓
              Session Representation
                          ↓
              ┌────────────────────────┐
              │   Item Embeddings      │
              │   (Candidate Pool)     │
              └───────────┬────────────┘
                          ↓
              Similarity Scores → Top-K Predictions
```
- **Complexity**: High
- **Training Time**: Days
- **Best for**: OTTO dataset, sequential patterns
- **Architectures**:
  - GRU4Rec (RNN-based)
  - SASRec (Self-Attention)
  - BERT4Rec (Bidirectional)

### 5. Hybrid Content + Collaborative
```
                    User History
                         ↓
              Collaborative Embedding
                         ↓
┌──────────────────────────────────────┐
│            Fusion Layer              │
└──────────────────────────────────────┘
                         ↑
              Content Embedding
                         ↑
            Item Text/Image Features
            (Optional: Pre-trained BERT/CLIP)
```
- **Complexity**: High
- **Training Time**: Days
- **Best for**: Amazon dataset with rich metadata
- **Fine-tuning opportunity**: Use pre-trained text/image encoders

### Architecture Recommendation by Dataset

| Dataset | Recommended Architecture | Alternative |
|---------|-------------------------|-------------|
| OTTO | Session-based (GRU4Rec/SASRec) | Two-Tower with co-visitation |
| Amazon | Two-Tower + Content | NCF + Text embeddings |
| Retailrocket | NCF | Matrix Factorization |
| H&M | Two-Tower + Image | Hybrid with CLIP |

---

## Training Strategy

### From Scratch vs Fine-Tuning Decision Tree

```
                    ┌─────────────────────────┐
                    │ Do you have text/image  │
                    │ features in your data?  │
                    └───────────┬─────────────┘
                               │
              ┌────────────────┴────────────────┐
              │ NO                              │ YES
              ↓                                 ↓
    ┌─────────────────┐              ┌─────────────────────┐
    │ Train from      │              │ Use pre-trained     │
    │ Scratch         │              │ encoders (BERT/CLIP)│
    │                 │              │ + Train rec head    │
    │ • Matrix Factor │              │                     │
    │ • NCF           │              │ Fine-tune or freeze │
    │ • GRU4Rec       │              │ based on data size  │
    └─────────────────┘              └─────────────────────┘
```

### Recommended Learning Path

#### Phase 1: Foundations (Week 1-2)
1. **Matrix Factorization from Scratch**
   - Understand embeddings and dot-product similarity
   - Implement BPR (Bayesian Personalized Ranking) loss
   - Dataset: Retailrocket (smaller, easier)

#### Phase 2: Deep Learning (Week 3-4)
2. **Neural Collaborative Filtering**
   - Add MLP layers on top of embeddings
   - Understand implicit vs explicit feedback
   - Dataset: Retailrocket or OTTO subset

#### Phase 3: Production Architecture (Week 5-6)
3. **Two-Tower or Session-Based Model**
   - Full OTTO dataset
   - Learn candidate retrieval + ranking pipeline
   - Implement co-visitation matrices

#### Phase 4: Advanced (Week 7+)
4. **Hybrid/Multi-Modal** (Optional)
   - Add text features with sentence-transformers
   - Fine-tune on Amazon dataset
   - Experiment with cross-attention fusion

### Loss Functions by Task

| Task | Loss Function | Use Case |
|------|---------------|----------|
| Rating Prediction | MSE, MAE | Explicit feedback (Amazon ratings) |
| Implicit Feedback | BPR, BCE | Clicks, views, purchases |
| Multi-Objective | Weighted BCE | OTTO (clicks + carts + orders) |
| Ranking | Softmax Cross-Entropy, ListNet | Top-K recommendations |

---

## Free Compute Resources

### Comparison Table

| Platform | Free GPU | Time Limit | Weekly Quota | Best For |
|----------|----------|------------|--------------|----------|
| **Kaggle** | P100/T4 (16GB) | 9 hrs/session | 30 hrs/week | Training, Competitions |
| **Google Colab** | T4 (16GB) | 12 hrs/session | ~30 hrs/week* | Prototyping, Experimentation |
| **Hugging Face Spaces** | T4 (limited) | Varies | Limited | Deployment, Demos |
| **Lightning AI** | T4 | 22 hrs/month | 22 hrs/month | PyTorch Lightning projects |

*Colab limits are dynamic based on demand

### Recommended Strategy

```
Development Workflow:
┌─────────────────────────────────────────────────────────┐
│ 1. Google Colab (Prototyping)                          │
│    • Quick experiments                                  │
│    • Debug code                                         │
│    • Small-scale tests                                  │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Kaggle Notebooks (Training)                         │
│    • Full training runs                                 │
│    • Background execution (close tab, keeps running)   │
│    • Access to competition datasets                    │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Hugging Face Spaces (Optional - Deployment)         │
│    • Host demo/inference                                │
│    • Share with others                                  │
└─────────────────────────────────────────────────────────┘
```

### Tips for Maximizing Free Resources

1. **Use Kaggle's background execution** - training continues when you close the tab
2. **Save checkpoints frequently** - sessions can disconnect
3. **Use mixed precision (FP16)** - faster training, less memory
4. **Start with smaller data samples** - validate approach before full training
5. **Combine quotas** - 30 hrs Kaggle + 30 hrs Colab = 60 hrs/week free

---

## Evaluation Metrics

### Metrics by Dataset Type

#### OTTO Dataset (Multi-Objective Ranking)
```python
# Primary metric: Weighted Recall@20
score = (recall_clicks * 0.10) + (recall_carts * 0.30) + (recall_orders * 0.60)
```
- **Recall@K**: Of all relevant items, how many did we retrieve in top K?
- Weights prioritize orders > carts > clicks (business value)

#### Amazon Dataset (Rating Prediction + Ranking)
| Metric | Formula | Use |
|--------|---------|-----|
| RMSE | √(Σ(y - ŷ)²/n) | Rating prediction accuracy |
| NDCG@K | Normalized ranking quality | Ranking quality |
| Hit Rate@K | % users with ≥1 hit in top K | Retrieval success |

#### Retailrocket (Implicit Feedback)
| Metric | Description |
|--------|-------------|
| Precision@K | Relevant items / K |
| Recall@K | Relevant items retrieved / Total relevant |
| MRR | Mean Reciprocal Rank of first relevant item |
| MAP | Mean Average Precision |

### Implementation Note
```python
# Use torchmetrics for easy metric computation
from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG

recall = RetrievalRecall(top_k=20)
ndcg = RetrievalNormalizedDCG(top_k=20)
```

---

## Implementation TODOs

### Phase 0: Setup
- [ ] Set up development environment (Kaggle account, Colab notebook)
- [ ] Create project structure (see below)
- [ ] Install dependencies: `torch`, `pandas`, `polars`, `torchmetrics`

### Phase 1: Data Exploration & Baseline
- [ ] Download OTTO dataset from Kaggle
- [ ] Exploratory Data Analysis (EDA)
  - [ ] Session length distribution
  - [ ] Event type distribution (clicks/carts/orders)
  - [ ] Item popularity analysis
  - [ ] Temporal patterns
- [ ] Implement popularity baseline (recommend most popular items)
- [ ] Implement co-visitation matrix baseline
- [ ] Establish baseline metrics (Recall@20)

### Phase 2: Matrix Factorization
- [ ] Preprocess data for matrix factorization
  - [ ] Create user-item interaction matrix
  - [ ] Handle implicit feedback (view=1, cart=2, order=3 weighting)
- [ ] Implement Matrix Factorization in PyTorch
  - [ ] User/Item embedding layers
  - [ ] BPR loss function
  - [ ] Negative sampling
- [ ] Train and evaluate
- [ ] Compare against baseline

### Phase 3: Neural Collaborative Filtering
- [ ] Implement NCF architecture
  - [ ] GMF (Generalized Matrix Factorization) component
  - [ ] MLP component
  - [ ] Fusion layer
- [ ] Add regularization (dropout, weight decay)
- [ ] Hyperparameter tuning (embedding dim, hidden layers)
- [ ] Train and evaluate
- [ ] Compare against Matrix Factorization

### Phase 4: Session-Based Model
- [ ] Preprocess data for sequential modeling
  - [ ] Create session sequences
  - [ ] Add event type encoding
  - [ ] Implement data loader with padding/truncation
- [ ] Implement GRU4Rec or SASRec
  - [ ] Sequence encoder (GRU/Transformer)
  - [ ] Item embedding layer
  - [ ] Prediction head
- [ ] Multi-objective training (predict clicks, carts, orders)
- [ ] Train on full OTTO dataset
- [ ] Evaluate on held-out test set

### Phase 5: Optimization & Experimentation
- [ ] Implement candidate retrieval + re-ranking pipeline
- [ ] Add co-visitation features as additional signal
- [ ] Experiment with ensemble methods
- [ ] Mixed precision training (AMP)
- [ ] Learning rate scheduling

### Phase 6: Documentation & Sharing (Optional)
- [ ] Write training notebook with explanations
- [ ] Create inference demo
- [ ] Deploy to Hugging Face Spaces
- [ ] Write learnings blog post

---

## Project Structure

```
maple/
├── RECOMMENDATION_SYSTEM_PLAN.md    # This file
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_baseline.ipynb            # Popularity & Co-visitation baselines
│   ├── 03_matrix_factorization.ipynb
│   ├── 04_ncf.ipynb                 # Neural Collaborative Filtering
│   ├── 05_session_based.ipynb       # GRU4Rec / SASRec
│   └── 06_evaluation.ipynb          # Final evaluation & comparison
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # PyTorch Dataset classes
│   │   ├── preprocessing.py         # Data cleaning & feature engineering
│   │   └── utils.py                 # Data utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── matrix_factorization.py
│   │   ├── ncf.py
│   │   ├── gru4rec.py
│   │   └── sasrec.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   ├── losses.py                # BPR, BCE, etc.
│   │   └── metrics.py               # Recall@K, NDCG, etc.
│   └── utils/
│       ├── __init__.py
│       └── config.py                # Hyperparameters
├── configs/
│   └── default.yaml                 # Training configurations
├── tests/
│   └── test_models.py
└── requirements.txt
```

---

## References

### Datasets
- [OTTO Recommender System Dataset](https://www.kaggle.com/datasets/otto/recsys-dataset)
- [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [Retailrocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

### Papers
- [Neural Collaborative Filtering (2017)](https://arxiv.org/abs/1708.05031)
- [Session-based Recommendations with RNNs - GRU4Rec (2016)](https://arxiv.org/abs/1511.06939)
- [Self-Attentive Sequential Recommendation - SASRec (2018)](https://arxiv.org/abs/1808.09781)
- [BERT4Rec (2019)](https://arxiv.org/abs/1904.06690)
- [Two-Tower Models for Recommendations](https://blog.reachsumit.com/posts/2023/03/two-tower-model/)

### Libraries & Tools
- [TorchRec - PyTorch Recommendation Library](https://pytorch.org/torchrec/)
- [Spotlight - Deep Recommender Models](https://github.com/maciejkula/spotlight)

### Tutorials & Solutions
- [OTTO Competition Solutions](https://www.kaggle.com/competitions/otto-recommender-system/discussion)
- [PyTorch Lightning RecSys Tutorial](https://lightning.ai/lightning-ai/studios/recommendation-system-with-pytorch-lightning)
- [NVIDIA RecSys Blog](https://developer.nvidia.com/blog/using-neural-networks-for-your-recommender-system/)

### Free Compute
- [Google Colab](https://colab.research.google.com/)
- [Kaggle Notebooks](https://www.kaggle.com/code)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

## Quick Start

```bash
# 1. Clone this repo and navigate to it
cd maple

# 2. Open Kaggle or Colab and run:
!pip install torch pandas polars torchmetrics matplotlib seaborn

# 3. Download OTTO dataset (in Kaggle notebook):
# - Go to kaggle.com/datasets/otto/recsys-dataset
# - Click "Add to notebook" or download

# 4. Start with notebooks/01_eda.ipynb
```

---

*Last updated: January 2025*
