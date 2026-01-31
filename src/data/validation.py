"""Data validation schemas and utilities using Pandera."""

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    from pandera.typing import Series
    PANDERA_AVAILABLE = True
except ImportError:
    pa = None
    Column = None
    Check = None
    DataFrameSchema = None
    Series = None
    PANDERA_AVAILABLE = False


# ============================================================================
# Schema Definitions
# ============================================================================

if PANDERA_AVAILABLE:
    # Interaction Schema
    InteractionSchema = DataFrameSchema(
        {
            "user_id": Column(
                str,
                Check.str_length(min_value=1),
                nullable=False,
                description="Unique user identifier",
            ),
            "product_id": Column(
                str,
                Check.str_length(min_value=1),
                nullable=False,
                description="Unique product identifier",
            ),
            "interaction_type": Column(
                str,
                Check.isin(["view", "click", "add_to_cart", "purchase", "rating", "search"]),
                nullable=False,
                description="Type of interaction",
            ),
            "timestamp": Column(
                "datetime64[ns]",
                nullable=False,
                description="When the interaction occurred",
            ),
            "rating": Column(
                float,
                Check.in_range(1.0, 5.0),
                nullable=True,
                required=False,
                description="Rating value (1-5) for rating interactions",
            ),
            "quantity": Column(
                int,
                Check.greater_than_or_equal_to(1),
                nullable=True,
                required=False,
                description="Quantity for purchase interactions",
            ),
            "session_id": Column(
                str,
                nullable=True,
                required=False,
                description="Session identifier",
            ),
        },
        coerce=True,
        strict=False,  # Allow extra columns
        name="InteractionSchema",
        description="Schema for user-product interactions",
    )

    # Product Schema
    ProductSchema = DataFrameSchema(
        {
            "product_id": Column(
                str,
                Check.str_length(min_value=1),
                nullable=False,
                unique=True,
                description="Unique product identifier",
            ),
            "name": Column(
                str,
                Check.str_length(min_value=1),
                nullable=False,
                description="Product name",
            ),
            "category": Column(
                str,
                nullable=True,
                description="Product category",
            ),
            "price": Column(
                float,
                Check.greater_than(0),
                nullable=True,
                description="Product price (must be positive)",
            ),
            "description": Column(
                str,
                nullable=True,
                required=False,
                description="Product description",
            ),
            "brand": Column(
                str,
                nullable=True,
                required=False,
                description="Product brand",
            ),
            "created_at": Column(
                "datetime64[ns]",
                nullable=True,
                required=False,
                description="Product creation timestamp",
            ),
        },
        coerce=True,
        strict=False,
        name="ProductSchema",
        description="Schema for product catalog",
    )

    # User Schema
    UserSchema = DataFrameSchema(
        {
            "user_id": Column(
                str,
                Check.str_length(min_value=1),
                nullable=False,
                unique=True,
                description="Unique user identifier",
            ),
            "created_at": Column(
                "datetime64[ns]",
                nullable=True,
                required=False,
                description="User registration timestamp",
            ),
            "age": Column(
                int,
                Check.in_range(0, 150),
                nullable=True,
                required=False,
                description="User age",
            ),
            "gender": Column(
                str,
                Check.isin(["M", "F", "O", "U"]),  # Male, Female, Other, Unknown
                nullable=True,
                required=False,
                description="User gender",
            ),
            "country": Column(
                str,
                nullable=True,
                required=False,
                description="User country",
            ),
        },
        coerce=True,
        strict=False,
        name="UserSchema",
        description="Schema for user profiles",
    )

else:
    # Placeholders when Pandera is not available
    InteractionSchema = None
    ProductSchema = None
    UserSchema = None


# ============================================================================
# Validation Functions
# ============================================================================

class DataValidator:
    """
    Data validation utility for recommendation system data.

    Provides schema validation, data quality checks, and drift detection.

    Example:
        validator = DataValidator()

        # Validate interactions
        is_valid, errors = validator.validate_interactions(df)
        if not is_valid:
            print(f"Validation errors: {errors}")

        # Check for drift
        drift_report = validator.detect_drift(old_df, new_df)
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """
        Initialize the validator.

        Args:
            strict_mode: If True, raise exceptions on validation failure
        """
        if not PANDERA_AVAILABLE:
            logger.warning(
                "Pandera not installed. Validation will be limited. "
                "Install with: pip install pandera"
            )

        self.strict_mode = strict_mode

    def validate_interactions(
        self,
        df: pd.DataFrame,
        raise_on_error: Optional[bool] = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate interaction data.

        Args:
            df: DataFrame to validate
            raise_on_error: Override strict_mode for this call

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        should_raise = raise_on_error if raise_on_error is not None else self.strict_mode
        errors = []

        # Basic checks without Pandera
        required_cols = ["user_id", "product_id", "interaction_type", "timestamp"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

        if df.empty:
            errors.append("DataFrame is empty")

        # Check for nulls in required columns
        for col in required_cols:
            if col in df.columns and df[col].isna().any():
                null_count = df[col].isna().sum()
                errors.append(f"Column '{col}' has {null_count} null values")

        # Pandera validation if available
        if PANDERA_AVAILABLE and InteractionSchema is not None:
            try:
                InteractionSchema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as e:
                for error in e.failure_cases.to_dict("records"):
                    errors.append(
                        f"Schema error in column '{error.get('column')}': "
                        f"{error.get('check')}"
                    )

        is_valid = len(errors) == 0

        if not is_valid and should_raise:
            raise ValueError(f"Interaction validation failed: {errors}")

        if not is_valid:
            logger.warning(f"Interaction validation failed: {errors}")

        return is_valid, errors

    def validate_products(
        self,
        df: pd.DataFrame,
        raise_on_error: Optional[bool] = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate product catalog data.

        Args:
            df: DataFrame to validate
            raise_on_error: Override strict_mode for this call

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        should_raise = raise_on_error if raise_on_error is not None else self.strict_mode
        errors = []

        # Basic checks
        if "product_id" not in df.columns:
            errors.append("Missing required column: product_id")
        elif df["product_id"].isna().any():
            errors.append("product_id contains null values")
        elif df["product_id"].duplicated().any():
            dup_count = df["product_id"].duplicated().sum()
            errors.append(f"product_id has {dup_count} duplicates")

        if "name" not in df.columns:
            errors.append("Missing required column: name")

        # Check price if present
        if "price" in df.columns:
            invalid_prices = df[df["price"] <= 0]["price"].count()
            if invalid_prices > 0:
                errors.append(f"{invalid_prices} products have invalid prices (<=0)")

        # Pandera validation if available
        if PANDERA_AVAILABLE and ProductSchema is not None:
            try:
                ProductSchema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as e:
                for error in e.failure_cases.to_dict("records"):
                    errors.append(
                        f"Schema error in column '{error.get('column')}': "
                        f"{error.get('check')}"
                    )

        is_valid = len(errors) == 0

        if not is_valid and should_raise:
            raise ValueError(f"Product validation failed: {errors}")

        return is_valid, errors

    def validate_users(
        self,
        df: pd.DataFrame,
        raise_on_error: Optional[bool] = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate user data.

        Args:
            df: DataFrame to validate
            raise_on_error: Override strict_mode for this call

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        should_raise = raise_on_error if raise_on_error is not None else self.strict_mode
        errors = []

        # Basic checks
        if "user_id" not in df.columns:
            errors.append("Missing required column: user_id")
        elif df["user_id"].isna().any():
            errors.append("user_id contains null values")
        elif df["user_id"].duplicated().any():
            dup_count = df["user_id"].duplicated().sum()
            errors.append(f"user_id has {dup_count} duplicates")

        # Check age if present
        if "age" in df.columns:
            invalid_age = df[(df["age"] < 0) | (df["age"] > 150)]["age"].count()
            if invalid_age > 0:
                errors.append(f"{invalid_age} users have invalid age values")

        # Pandera validation if available
        if PANDERA_AVAILABLE and UserSchema is not None:
            try:
                UserSchema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as e:
                for error in e.failure_cases.to_dict("records"):
                    errors.append(
                        f"Schema error in column '{error.get('column')}': "
                        f"{error.get('check')}"
                    )

        is_valid = len(errors) == 0

        if not is_valid and should_raise:
            raise ValueError(f"User validation failed: {errors}")

        return is_valid, errors

    def get_data_quality_report(self, df: pd.DataFrame, name: str = "data") -> dict:
        """
        Generate a data quality report.

        Args:
            df: DataFrame to analyze
            name: Name for the report

        Returns:
            Dictionary with quality metrics
        """
        report = {
            "name": name,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": {},
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        for col in df.columns:
            col_report = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "null_pct": float(df[col].isna().mean() * 100),
                "unique_count": int(df[col].nunique()),
                "unique_pct": float(df[col].nunique() / len(df) * 100) if len(df) > 0 else 0,
            }

            # Add stats for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_report.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                })

            report["columns"][col] = col_report

        return report

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.1,
    ) -> dict:
        """
        Detect distribution drift between two datasets.

        Uses Population Stability Index (PSI) for numeric columns
        and chi-square for categorical columns.

        Args:
            reference_df: Reference (baseline) DataFrame
            current_df: Current DataFrame to compare
            threshold: PSI threshold for drift detection (default: 0.1)

        Returns:
            Dictionary with drift analysis results
        """
        drift_report = {
            "has_drift": False,
            "columns": {},
            "threshold": threshold,
        }

        # Find common columns
        common_cols = set(reference_df.columns) & set(current_df.columns)

        for col in common_cols:
            col_drift = {"has_drift": False, "metric": None, "value": None}

            ref_col = reference_df[col].dropna()
            cur_col = current_df[col].dropna()

            if len(ref_col) == 0 or len(cur_col) == 0:
                col_drift["metric"] = "insufficient_data"
                drift_report["columns"][col] = col_drift
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                # Calculate PSI for numeric columns
                psi = self._calculate_psi(ref_col, cur_col)
                col_drift["metric"] = "psi"
                col_drift["value"] = psi
                col_drift["has_drift"] = psi > threshold

            else:
                # Calculate category distribution change for categorical
                ref_dist = ref_col.value_counts(normalize=True)
                cur_dist = cur_col.value_counts(normalize=True)

                # Align distributions
                all_cats = set(ref_dist.index) | set(cur_dist.index)
                ref_aligned = ref_dist.reindex(all_cats, fill_value=0)
                cur_aligned = cur_dist.reindex(all_cats, fill_value=0)

                # Calculate total variation distance
                tvd = float((ref_aligned - cur_aligned).abs().sum() / 2)
                col_drift["metric"] = "tvd"
                col_drift["value"] = tvd
                col_drift["has_drift"] = tvd > threshold

            drift_report["columns"][col] = col_drift

            if col_drift["has_drift"]:
                drift_report["has_drift"] = True

        return drift_report

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change
        PSI > 0.25: Significant change

        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        _, bins = pd.cut(reference, bins=n_bins, retbins=True, duplicates="drop")

        # Calculate proportions in each bin
        ref_counts = pd.cut(reference, bins=bins).value_counts(normalize=True)
        cur_counts = pd.cut(current, bins=bins).value_counts(normalize=True)

        # Align and handle zeros - use set union to avoid Categorical index issues
        all_bins = list(set(ref_counts.index.tolist()) | set(cur_counts.index.tolist()))
        ref_counts = ref_counts.reindex(all_bins, fill_value=0.0001)
        cur_counts = cur_counts.reindex(all_bins, fill_value=0.0001)

        # Calculate PSI
        import numpy as np
        psi = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))

        return abs(psi)


def validate_before_training(
    interactions_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    users_df: Optional[pd.DataFrame] = None,
    strict: bool = True,
) -> bool:
    """
    Validate all data before training.

    Args:
        interactions_df: Interaction data
        products_df: Optional product catalog
        users_df: Optional user data
        strict: Raise exception on validation failure

    Returns:
        True if all validations pass

    Raises:
        ValueError: If validation fails and strict=True
    """
    validator = DataValidator(strict_mode=strict)
    all_valid = True
    all_errors = []

    # Validate interactions
    valid, errors = validator.validate_interactions(interactions_df)
    all_valid &= valid
    all_errors.extend(errors)

    # Validate products if provided
    if products_df is not None:
        valid, errors = validator.validate_products(products_df)
        all_valid &= valid
        all_errors.extend(errors)

    # Validate users if provided
    if users_df is not None:
        valid, errors = validator.validate_users(users_df)
        all_valid &= valid
        all_errors.extend(errors)

    if not all_valid:
        logger.error(f"Data validation failed with {len(all_errors)} errors")
        if strict:
            raise ValueError(f"Validation failed: {all_errors}")

    return all_valid
