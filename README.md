# Maple

A modular product recommendation system built with Python.

## Features

- **Multiple Recommendation Models**
  - Popularity-based recommendations
  - Item-based Collaborative Filtering (KNN)
  - User-based Collaborative Filtering (KNN)
  - Matrix Factorization (ALS)

- **Comprehensive Evaluation**
  - Precision@K, Recall@K, NDCG@K
  - Hit Rate, MRR, MAP
  - Coverage and Diversity metrics

- **REST API**
  - FastAPI-based recommendation service
  - Endpoints for user recommendations, similar items, and popular items

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/maple.git
cd maple

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### 1. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates sample interaction data in `data/sample/`.

### 2. Run the Example

```bash
python examples/quickstart.py
```

### 3. Start the API

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
│   ├── api/           # REST API
│   ├── data/          # Data loading and schemas
│   ├── evaluation/    # Metrics and evaluation
│   └── models/        # Recommendation models
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
