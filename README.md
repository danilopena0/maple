# Maple

A modular product recommendation system built with Python.

## Features

- **Multiple Recommendation Models**
  - Popularity-based recommendations
  - Item-based Collaborative Filtering (KNN)
  - User-based Collaborative Filtering (KNN)
  - Matrix Factorization (ALS)
  - Content-based, Neural, Hybrid, and Ensemble models

- **Comprehensive Evaluation**
  - Precision@K, Recall@K, NDCG@K
  - Hit Rate, MRR, MAP
  - Coverage and Diversity metrics

- **ML Operations**
  - MLflow experiment tracking
  - Optuna hyperparameter tuning
  - Model monitoring with drift detection and alerts
  - Pandera data validation

- **REST API**
  - FastAPI-based recommendation service
  - Endpoints for user recommendations, similar items, and popular items

## Exploration & Visualization

An interactive Jupyter notebook is available at [`notebooks/explore.ipynb`](notebooks/explore.ipynb) that covers:

- Dataset overview: interaction types, activity over time, products by category
- User activity and product popularity distributions (power-law)
- Interaction matrix sparsity heatmap
- Train all models in one cell and compare evaluation metrics (Precision, Recall, NDCG, Hit Rate)
- Inspect recommendations for any specific user, side-by-side across models
- Similar items lookup for any product
- Top popular products chart

```bash
source venv/bin/activate
jupyter notebook notebooks/explore.ipynb
```

## Getting Started

### Prerequisites

- Python 3.10+ (3.11 or 3.12 recommended)
- WSL (Ubuntu) or native Linux

### 1. Clone and Set Up the Virtual Environment

```bash
git clone https://github.com/your-org/maple.git
cd maple

python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as an editable package (useful for development):

```bash
pip install -e ".[dev]"
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` as needed:

| Variable | Description | Default |
|----------|-------------|---------|
| `MAPLE_DATA_PATH` | Path to interaction data CSV (auto-loaded on API startup) | `data/sample/interactions.csv` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `./mlruns` |

### 4. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates sample interaction data in `data/sample/`.

### 5. Run the Quickstart Example

```bash
python examples/quickstart.py
```

### 6. Start the API

```bash
python -m src.api.main
```

Then visit http://localhost:8000/docs for interactive API documentation.

## Usage

### Basic Example

```python
from src.data.loader import DataLoader
from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemKNNRecommender, ALSRecommender

# Load data
loader = DataLoader()
loader.load_interactions("data/sample/interactions.csv")

# Create interaction matrix
matrix = loader.get_interaction_matrix(weighted=True)

# Train models
pop_model = PopularityRecommender()
pop_model.fit(matrix)

knn_model = ItemKNNRecommender(k=50)
knn_model.fit(matrix)

# Get recommendations
user_idx = loader.user_id_to_idx["user_00001"]
recs = knn_model.recommend(user_idx, n=10)

for item_idx, score in recs:
    product_id = loader.idx_to_product_id[item_idx]
    print(f"{product_id}: {score:.4f}")
```

### Evaluation

```python
from src.evaluation.metrics import evaluate_model, print_evaluation_results

# Prepare test data
test_interactions = {
    user_idx: {item_idx1, item_idx2, ...}
    for user_idx, items in test_data.items()
}

# Evaluate
results = evaluate_model(
    model,
    test_interactions,
    k_values=[5, 10, 20],
)

print_evaluation_results(results, "Item-KNN")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List loaded models |
| `/train` | POST | Train models on data |
| `/recommend/{user_id}` | GET | Get recommendations for user |
| `/similar/{product_id}` | GET | Get similar products |
| `/popular` | GET | Get popular products |

## Project Structure

```
maple/
├── src/
│   ├── api/           # REST API (FastAPI)
│   ├── data/          # Data loading, schemas, validation
│   ├── evaluation/    # Metrics and evaluation
│   ├── models/        # Recommendation models
│   ├── monitoring/    # Model monitoring and drift detection
│   ├── tracking/      # MLflow experiment tracking
│   └── tuning/        # Optuna hyperparameter tuning
├── tests/             # Unit tests
├── scripts/           # Utility scripts
├── examples/          # Usage examples
├── data/              # Data directory
└── docs/              # Documentation
```

## Running Tests

```bash
pytest tests/ -v
```

## Documentation

See [docs/ml-model-plan.md](docs/ml-model-plan.md) for:
- Fine-tunable models overview
- Phased development plan
- Training/testing/validation strategy
- Model architecture comparisons

## License

MIT
