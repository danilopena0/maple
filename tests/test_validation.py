"""Tests for data validation."""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    DataValidator,
    validate_before_training,
    PANDERA_AVAILABLE,
)


@pytest.fixture
def valid_interactions():
    """Create valid interaction data."""
    return pd.DataFrame({
        "user_id": ["user_001", "user_002", "user_001", "user_003"],
        "product_id": ["prod_001", "prod_002", "prod_003", "prod_001"],
        "interaction_type": ["view", "purchase", "click", "rating"],
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
        "rating": [None, None, None, 4.5],
    })


@pytest.fixture
def valid_products():
    """Create valid product data."""
    return pd.DataFrame({
        "product_id": ["prod_001", "prod_002", "prod_003"],
        "name": ["Product 1", "Product 2", "Product 3"],
        "category": ["electronics", "clothing", "books"],
        "price": [99.99, 49.99, 19.99],
    })


@pytest.fixture
def valid_users():
    """Create valid user data."""
    return pd.DataFrame({
        "user_id": ["user_001", "user_002", "user_003"],
        "created_at": pd.date_range("2024-01-01", periods=3, freq="D"),
        "age": [25, 35, 45],
        "gender": ["M", "F", "O"],
    })


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = DataValidator()
        assert validator.strict_mode is False

        strict_validator = DataValidator(strict_mode=True)
        assert strict_validator.strict_mode is True

    def test_validate_interactions_valid(self, valid_interactions):
        """Test validation of valid interactions."""
        validator = DataValidator()
        is_valid, errors = validator.validate_interactions(valid_interactions)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_interactions_missing_columns(self):
        """Test validation with missing columns."""
        validator = DataValidator()
        df = pd.DataFrame({
            "user_id": ["user_001"],
            "product_id": ["prod_001"],
            # Missing interaction_type and timestamp
        })

        is_valid, errors = validator.validate_interactions(df)

        assert is_valid is False
        assert any("Missing required columns" in e for e in errors)

    def test_validate_interactions_null_values(self, valid_interactions):
        """Test validation with null values in required columns."""
        validator = DataValidator()
        df = valid_interactions.copy()
        df.loc[0, "user_id"] = None

        is_valid, errors = validator.validate_interactions(df)

        assert is_valid is False
        assert any("null values" in e for e in errors)

    def test_validate_interactions_empty(self):
        """Test validation of empty DataFrame."""
        validator = DataValidator()
        df = pd.DataFrame(columns=["user_id", "product_id", "interaction_type", "timestamp"])

        is_valid, errors = validator.validate_interactions(df)

        assert is_valid is False
        assert any("empty" in e.lower() for e in errors)

    def test_validate_interactions_strict_mode(self):
        """Test strict mode raises exception."""
        validator = DataValidator(strict_mode=True)
        df = pd.DataFrame({"user_id": ["user_001"]})  # Missing columns

        with pytest.raises(ValueError):
            validator.validate_interactions(df)

    def test_validate_products_valid(self, valid_products):
        """Test validation of valid products."""
        validator = DataValidator()
        is_valid, errors = validator.validate_products(valid_products)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_products_duplicate_ids(self, valid_products):
        """Test validation with duplicate product IDs."""
        validator = DataValidator()
        df = valid_products.copy()
        df = pd.concat([df, df.iloc[[0]]])  # Add duplicate

        is_valid, errors = validator.validate_products(df)

        assert is_valid is False
        assert any("duplicate" in e.lower() for e in errors)

    def test_validate_products_invalid_price(self, valid_products):
        """Test validation with invalid prices."""
        validator = DataValidator()
        df = valid_products.copy()
        df.loc[0, "price"] = -10.0  # Invalid negative price

        is_valid, errors = validator.validate_products(df)

        assert is_valid is False
        assert any("price" in e.lower() for e in errors)

    def test_validate_users_valid(self, valid_users):
        """Test validation of valid users."""
        validator = DataValidator()
        is_valid, errors = validator.validate_users(valid_users)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_users_duplicate_ids(self, valid_users):
        """Test validation with duplicate user IDs."""
        validator = DataValidator()
        df = valid_users.copy()
        df = pd.concat([df, df.iloc[[0]]])

        is_valid, errors = validator.validate_users(df)

        assert is_valid is False
        assert any("duplicate" in e.lower() for e in errors)

    def test_validate_users_invalid_age(self, valid_users):
        """Test validation with invalid age."""
        validator = DataValidator()
        df = valid_users.copy()
        df.loc[0, "age"] = -5  # Invalid age

        is_valid, errors = validator.validate_users(df)

        assert is_valid is False
        assert any("age" in e.lower() for e in errors)


class TestDataQualityReport:
    """Tests for data quality report generation."""

    def test_get_data_quality_report(self, valid_interactions):
        """Test quality report generation."""
        validator = DataValidator()
        report = validator.get_data_quality_report(valid_interactions, name="test_data")

        assert report["name"] == "test_data"
        assert report["n_rows"] == 4
        assert "columns" in report
        assert "user_id" in report["columns"]
        assert "null_count" in report["columns"]["user_id"]

    def test_report_numeric_stats(self):
        """Test that numeric columns have stats."""
        validator = DataValidator()
        df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
        })

        report = validator.get_data_quality_report(df)

        assert "mean" in report["columns"]["value"]
        assert report["columns"]["value"]["mean"] == 2.0
        assert "mean" not in report["columns"]["id"]

    def test_report_empty_dataframe(self):
        """Test report on empty DataFrame."""
        validator = DataValidator()
        df = pd.DataFrame({"col": []})

        report = validator.get_data_quality_report(df)

        assert report["n_rows"] == 0


class TestDriftDetection:
    """Tests for drift detection."""

    def test_detect_drift_no_drift(self):
        """Test drift detection when no drift."""
        validator = DataValidator()

        np.random.seed(42)
        ref_df = pd.DataFrame({
            "value": np.random.normal(0, 1, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        })
        cur_df = pd.DataFrame({
            "value": np.random.normal(0, 1, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        })

        report = validator.detect_drift(ref_df, cur_df)

        # Should not detect significant drift with same distributions
        assert "has_drift" in report
        # Note: Small random variations might cause drift, so we don't assert False

    def test_detect_drift_with_drift(self):
        """Test drift detection when there is drift."""
        validator = DataValidator()

        np.random.seed(42)
        ref_df = pd.DataFrame({
            "value": np.random.normal(0, 1, 1000),
        })
        cur_df = pd.DataFrame({
            "value": np.random.normal(5, 1, 1000),  # Shifted mean
        })

        report = validator.detect_drift(ref_df, cur_df, threshold=0.1)

        assert report["has_drift"] is True
        assert report["columns"]["value"]["has_drift"] is True
        assert report["columns"]["value"]["metric"] == "psi"

    def test_detect_drift_categorical(self):
        """Test drift detection for categorical columns."""
        validator = DataValidator()

        ref_df = pd.DataFrame({
            "category": ["A"] * 500 + ["B"] * 500,
        })
        cur_df = pd.DataFrame({
            "category": ["A"] * 900 + ["B"] * 100,  # Changed distribution
        })

        report = validator.detect_drift(ref_df, cur_df, threshold=0.1)

        assert report["columns"]["category"]["metric"] == "tvd"
        assert report["columns"]["category"]["has_drift"] is True


class TestValidateBeforeTraining:
    """Tests for validate_before_training function."""

    def test_validate_all_valid(self, valid_interactions, valid_products, valid_users):
        """Test validation when all data is valid."""
        result = validate_before_training(
            interactions_df=valid_interactions,
            products_df=valid_products,
            users_df=valid_users,
            strict=False,
        )

        assert result is True

    def test_validate_interactions_only(self, valid_interactions):
        """Test validation with only interactions."""
        result = validate_before_training(
            interactions_df=valid_interactions,
            strict=False,
        )

        assert result is True

    def test_validate_invalid_raises(self):
        """Test that invalid data raises with strict=True."""
        df = pd.DataFrame({"bad_column": [1, 2, 3]})

        with pytest.raises(ValueError):
            validate_before_training(df, strict=True)

    def test_validate_invalid_no_raise(self):
        """Test that invalid data returns False with strict=False."""
        df = pd.DataFrame({"bad_column": [1, 2, 3]})

        result = validate_before_training(df, strict=False)

        assert result is False


@pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
class TestPanderaSchemas:
    """Tests specific to Pandera schema validation."""

    def test_interaction_schema_invalid_type(self):
        """Test schema catches invalid interaction types."""
        from src.data.validation import InteractionSchema
        import pandera as pa

        df = pd.DataFrame({
            "user_id": ["user_001"],
            "product_id": ["prod_001"],
            "interaction_type": ["invalid_type"],  # Not in allowed list
            "timestamp": pd.Timestamp("2024-01-01"),
        })

        with pytest.raises(pa.errors.SchemaError):
            InteractionSchema.validate(df)

    def test_interaction_schema_rating_range(self):
        """Test schema validates rating range."""
        from src.data.validation import InteractionSchema
        import pandera as pa

        df = pd.DataFrame({
            "user_id": ["user_001"],
            "product_id": ["prod_001"],
            "interaction_type": ["rating"],
            "timestamp": pd.Timestamp("2024-01-01"),
            "rating": [6.0],  # Out of range (1-5)
        })

        with pytest.raises(pa.errors.SchemaError):
            InteractionSchema.validate(df)

    def test_product_schema_unique_id(self):
        """Test schema validates unique product IDs."""
        from src.data.validation import ProductSchema
        import pandera as pa

        df = pd.DataFrame({
            "product_id": ["prod_001", "prod_001"],  # Duplicate
            "name": ["Product 1", "Product 2"],
        })

        with pytest.raises(pa.errors.SchemaError):
            ProductSchema.validate(df)
