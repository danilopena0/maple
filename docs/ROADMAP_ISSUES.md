# Maple Roadmap - GitHub Issues

Copy each issue below to create them in GitHub.

---

## High Priority

### Issue 1: ML Experiment Tracking with MLflow

**Title:** `[Feature] ML Experiment Tracking with MLflow`

**Labels:** `enhancement`, `high-priority`

**Body:**
```
## Summary
Integrate MLflow for comprehensive experiment tracking and model management.

## Requirements
- [ ] Track hyperparameters, metrics, and artifacts for every experiment
- [ ] Compare model runs side-by-side in MLflow UI
- [ ] Model registry for versioning and deployment staging
- [ ] Ensure full reproducibility across training runs

## Implementation Plan
1. Add `mlflow` to dependencies
2. Create `src/tracking/mlflow_tracker.py` with:
   - Auto-logging wrapper for all models
   - Metric logging (Precision@K, NDCG, etc.)
   - Artifact storage (model files, plots)
3. Add MLflow tracking server configuration
4. Integrate model registry with staging/production stages
5. Add experiment comparison utilities

## Acceptance Criteria
- All model training automatically logged to MLflow
- Can compare any two experiments in UI
- Models can be promoted through registry stages

## Priority
High - Foundation for reproducible ML
```

---

### Issue 2: Hyperparameter Tuning with Optuna

**Title:** `[Feature] Hyperparameter Tuning with Optuna`

**Labels:** `enhancement`, `high-priority`

**Body:**
```
## Summary
Integrate Optuna for automated hyperparameter optimization.

## Requirements
- [ ] Automated search for optimal parameters
- [ ] Bayesian optimization instead of grid search
- [ ] Pruning of bad trials early (MedianPruner)
- [ ] Integration with MLflow for tracking trials

## Implementation Plan
1. Add `optuna` to dependencies
2. Create `src/tuning/optuna_tuner.py` with:
   - Objective functions for each model type
   - Search space definitions
   - Pruning callbacks
3. Add tuning CLI command
4. Store best parameters in MLflow

## Example Search Spaces
- ALS: factors [16-256], regularization [0.001-0.1], iterations [5-50]
- ItemKNN: k [10-200], min_similarity [0-0.3]
- NCF: embedding_dim [16-128], mlp_layers, dropout [0.1-0.5]

## Acceptance Criteria
- Can tune any model with single command
- Trials auto-logged to MLflow
- Best params saved and reproducible

## Priority
High - Enables systematic model improvement
```

---

### Issue 3: Data Validation with Pandera

**Title:** `[Feature] Data Validation with Pandera`

**Labels:** `enhancement`, `high-priority`

**Body:**
```
## Summary
Add data validation to catch data quality issues before they affect models.

## Requirements
- [ ] Schema validation for incoming interaction data
- [ ] Distribution drift detection between training batches
- [ ] Data quality alerts and reports
- [ ] Integration with data loader

## Implementation Plan
1. Add `pandera` to dependencies
2. Create `src/data/validation.py` with:
   - InteractionSchema (user_id, product_id, interaction_type, timestamp)
   - ProductSchema (product_id, name, category, price)
   - UserSchema (user_id, created_at)
3. Add drift detection:
   - Compare feature distributions between batches
   - Alert on significant shifts
4. Add validation to DataLoader.load_interactions()

## Validation Rules
- user_id: non-null, string
- product_id: non-null, string
- interaction_type: in ['view', 'click', 'add_to_cart', 'purchase', 'rating']
- timestamp: valid datetime, not in future
- rating: 1-5 when present

## Acceptance Criteria
- Invalid data raises clear errors
- Drift reports generated on each training run
- Can run validation independently

## Priority
High - Prevents garbage-in-garbage-out
```

---

### Issue 4: Model Monitoring

**Title:** `[Feature] Model Monitoring and Drift Detection`

**Labels:** `enhancement`, `high-priority`

**Body:**
```
## Summary
Monitor model performance and detect drift in production.

## Requirements
- [ ] Prediction drift detection (recommendation distribution changes)
- [ ] Feature drift monitoring (input data changes)
- [ ] Performance degradation alerts
- [ ] Dashboard for monitoring metrics

## Implementation Plan
1. Create `src/monitoring/` module:
   - `drift_detector.py`: Statistical tests for drift
   - `performance_tracker.py`: Track online metrics
   - `alerts.py`: Alerting logic
2. Implement drift detection:
   - PSI (Population Stability Index) for features
   - KS test for prediction distributions
3. Add performance tracking:
   - Log predictions with timestamps
   - Compare against baseline metrics
4. Create monitoring dashboard (Grafana/Streamlit)

## Metrics to Monitor
- Recommendation coverage over time
- Average prediction confidence
- Feature value distributions
- Response latency percentiles

## Acceptance Criteria
- Alerts fire when drift exceeds threshold
- Can view 7-day performance trends
- Automatic baseline comparison

## Priority
High - Critical for production reliability
```

---

## Medium Priority

### Issue 5: Feature Store

**Title:** `[Feature] Feature Store Implementation`

**Labels:** `enhancement`, `medium-priority`

**Body:**
```
## Summary
Centralized feature management for training and serving consistency.

## Requirements
- [ ] Centralized feature definitions
- [ ] Online/offline feature consistency
- [ ] Feature versioning
- [ ] Point-in-time correct feature retrieval

## Implementation Plan
1. Evaluate options: Feast, Tecton, or custom
2. Create `src/features/` module:
   - Feature definitions
   - Online store (Redis)
   - Offline store (Parquet/Delta)
3. Implement feature pipelines:
   - User features (activity level, preferences)
   - Item features (popularity, categories)
   - Interaction features (recency, frequency)

## Features to Implement
- user_interaction_count_7d
- user_avg_price_purchased
- item_popularity_score
- item_category_embedding
- user_item_affinity_score

## Acceptance Criteria
- Same feature values in training and serving
- Can retrieve historical features for any timestamp
- Feature lineage tracked

## Priority
Medium - Important for feature consistency
```

---

### Issue 6: Sequential Models (GRU4Rec, SASRec)

**Title:** `[Feature] Sequential Recommendation Models`

**Labels:** `enhancement`, `medium-priority`

**Body:**
```
## Summary
Add sequential models that capture temporal patterns in user behavior.

## Requirements
- [ ] GRU4Rec implementation
- [ ] SASRec (Self-Attention) implementation
- [ ] Session-based recommendation support
- [ ] Sequence data preprocessing

## Implementation Plan
1. Create `src/models/sequential.py`:
   - GRU4Rec: GRU-based session recommender
   - SASRec: Transformer-based sequential model
2. Add sequence data handling:
   - Session extraction from interactions
   - Sequence padding/truncation
   - Negative sampling for sequences
3. Integrate with existing evaluation framework

## Architecture
GRU4Rec:
- Embedding layer → GRU layers → Output layer
- Session-parallel mini-batches

SASRec:
- Embedding + Positional → Self-Attention blocks → Prediction

## Acceptance Criteria
- Both models trainable on interaction sequences
- Evaluation with sequential metrics
- Can recommend "next item" given session

## Priority
Medium - Captures important temporal signals
```

---

### Issue 7: Caching Layer

**Title:** `[Feature] Redis Caching Layer`

**Labels:** `enhancement`, `medium-priority`

**Body:**
```
## Summary
Add caching for pre-computed recommendations to reduce latency.

## Requirements
- [ ] Redis integration for recommendation caching
- [ ] Cache invalidation strategy
- [ ] Fallback to real-time computation
- [ ] Cache hit/miss metrics

## Implementation Plan
1. Add `redis` to dependencies
2. Create `src/cache/` module:
   - `redis_cache.py`: Cache client wrapper
   - `cache_manager.py`: Invalidation logic
3. Implement caching strategies:
   - User recommendations (TTL: 1 hour)
   - Similar items (TTL: 24 hours)
   - Popular items (TTL: 15 minutes)
4. Add cache-aside pattern to API

## Cache Keys
- `recs:user:{user_id}` → Top-N recommendations
- `similar:{item_id}` → Similar items
- `popular` → Global popular items

## Acceptance Criteria
- p99 latency < 50ms for cached users
- Cache hit rate > 80% for active users
- Graceful degradation on Redis failure

## Priority
Medium - Important for production latency
```

---

### Issue 8: API Improvements

**Title:** `[Feature] API Improvements - Batch, Async, Rate Limiting`

**Labels:** `enhancement`, `medium-priority`

**Body:**
```
## Summary
Enhance the REST API for production readiness.

## Requirements
- [ ] Batch recommendation endpoint
- [ ] Async processing for heavy operations
- [ ] Rate limiting per client
- [ ] Request validation improvements

## Implementation Plan
1. Add batch endpoint:
   - POST /recommendations/batch
   - Accept list of user_ids
   - Return recommendations for all users
2. Add async processing:
   - Background model training
   - Async batch recommendations
3. Add rate limiting:
   - Use slowapi or custom middleware
   - Per-IP and per-API-key limits
4. Improve validation:
   - Request size limits
   - Input sanitization

## New Endpoints
- POST /recommendations/batch - Batch recommendations
- POST /models/train/async - Async training job
- GET /models/train/status/{job_id} - Training status

## Rate Limits
- Anonymous: 100 req/min
- Authenticated: 1000 req/min
- Batch: 10 req/min, max 100 users/request

## Acceptance Criteria
- Batch endpoint handles 100 users in < 1s
- Training doesn't block API
- Rate limits enforced with clear error messages

## Priority
Medium - Required for production deployment
```

---

## Quick Create Script

Save this as `create_issues.sh` and run with a GitHub token:

```bash
#!/bin/bash
# Usage: GITHUB_TOKEN=your_token ./create_issues.sh

REPO="danilopena0/maple"
API="https://api.github.com/repos/$REPO/issues"

create_issue() {
  curl -s -X POST "$API" \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -d "$1"
}

# Issue 1: MLflow
create_issue '{
  "title": "[Feature] ML Experiment Tracking with MLflow",
  "labels": ["enhancement", "high-priority"],
  "body": "## Summary\nIntegrate MLflow for comprehensive experiment tracking.\n\n## Requirements\n- Track hyperparameters, metrics, and artifacts\n- Compare model runs side-by-side\n- Model registry for versioning\n- Reproducibility\n\n## Priority\nHigh"
}'

# Add more issues...
echo "Issues created!"
```
