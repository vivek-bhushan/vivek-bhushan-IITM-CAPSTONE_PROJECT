"""
data_prep.py
============
Phase 1 - Problem Framing and Dataset Preparation.

Responsibilities
----------------
* Load the raw retail interaction dataset and the policy documents.
* Inspect schema, missingness, target columns and class balance.
* Clean the data (explicit, documented missing-value handling).
* Engineer a small number of business-meaningful features.
* Emit a clean dataset, an ML feature table, and a machine-readable data
  dictionary that records **which columns are original vs engineered**.

Run directly:  ``python src/data_prep.py``
"""

from __future__ import annotations

import json
import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    CLEAN_INTERACTIONS,
    DATA_DICTIONARY_CSV,
    FEATURES_CSV,
    POLICIES_JSON,
    PROCESSED_DIR,
    RAW_INTERACTIONS,
    ensure_dirs,
)

# --------------------------------------------------------------------------- #
# Column provenance - the single source of truth for the data dictionary
# --------------------------------------------------------------------------- #
ORIGINAL_COLUMNS: Dict[str, str] = {
    "customer_id": "Unique customer identifier (categorical key).",
    "age": "Customer age in years (numerical).",
    "gender": "Customer gender (categorical).",
    "city_tier": "City tier of the customer: Tier-1/2/3 (categorical).",
    "state": "Customer state (categorical).",
    "income_band": "Self-reported income band: Low/Medium/High (categorical).",
    "occupation": "Customer occupation (categorical).",
    "membership_level": "Loyalty tier: Bronze/Silver/Gold/Platinum (categorical).",
    "days_since_last_purchase": "Recency in days since the customer's last purchase (numerical).",
    "avg_order_value": "Customer's historical average order value in INR (numerical).",
    "total_orders": "Lifetime number of orders placed by the customer (numerical).",
    "preferred_channel": "Preferred shopping channel: App/Web/Store (categorical).",
    "product_id": "Unique product identifier (categorical key).",
    "category": "Top-level product category (categorical).",
    "sub_category": "Product sub-category (categorical).",
    "brand": "Product brand (categorical).",
    "base_price": "List price before discount in INR (numerical).",
    "discount_percent": "Discount applied, 0-70 percent (numerical).",
    "final_price": "Price actually paid after discount in INR (numerical).",
    "avg_product_rating": "Average product rating, 1-5 (numerical).",
    "stock_level": "Units currently in stock (numerical).",
    "color": "Product colour (categorical).",
    "size": "Product size; 'NA' for non-sized items (categorical, has missing).",
    "material": "Product material (categorical).",
    "season": "Intended season: Summer/Winter/All-Season (categorical).",
    "is_returnable": "1 if the product is eligible for return (binary).",
    "purchase_flag": "TARGET - 1 if the interaction converted to a purchase (binary).",
    "quantity": "Units purchased in the interaction (numerical).",
    "payment_mode": "Payment method used (categorical).",
    "delivery_type": "Delivery speed chosen: Standard/Express (categorical).",
    "time_on_page": "Seconds spent on the product page (numerical).",
    "added_to_cart": "1 if the item was added to cart (binary).",
    "wishlist_flag": "1 if the item was wishlisted (binary).",
    "returned_flag": "TARGET - 1 if the purchased item was returned (binary, imbalanced).",
    "rating_given": "Post-purchase rating 1-5 (ordinal).",
    "review_sentiment_label": "Sentiment label of the review text (categorical).",
    "product_title": "Short product title (text).",
    "product_description": "Marketing description of the product (text).",
    "customer_review_text": "Free-text customer review (text).",
    "customer_query": "Synthetic natural-language query for the product (text).",
    "policy_doc_id": "Foreign key into the policy documents (text).",
    "policy_text_snippet": "Policy snippet attached to the record (text).",
    "image_path": "Relative path to the product image (text, optional multimodal).",
    "recommended_use_case": "Intended use case, e.g. Daily Wear (categorical).",
    "style_tags": "Comma-separated style tags (text).",
    "occasion_tags": "Comma-separated occasion tags (text).",
}

ENGINEERED_COLUMNS: Dict[str, str] = {
    "product_key": "ENGINEERED - Stable unique product identity. The source product_id is NOT unique (300 ids repeat across 1500 rows with different attributes), so product identity is derived from the full attribute signature.",
    "price_band": "ENGINEERED - Bucketed final_price into Budget/Mid/Premium/Luxury.",
    "discount_band": "ENGINEERED - Bucketed discount_percent into None/Low/Medium/High.",
    "price_sensitivity": "ENGINEERED - discount_percent / (base_price/1000), a proxy for how hard the price was cut.",
    "engagement_score": "ENGINEERED - Composite of time_on_page, added_to_cart and wishlist_flag.",
    "recency_band": "ENGINEERED - Bucketed days_since_last_purchase into Active/Warm/Cooling/Dormant.",
    "customer_value": "ENGINEERED - avg_order_value * total_orders, a lifetime-value proxy.",
    "is_discounted": "ENGINEERED - 1 if discount_percent > 0.",
    "size_missing": "ENGINEERED - 1 if the original size value was missing.",
    "product_text": "ENGINEERED - Concatenated retrieval-ready product document (title + description + attributes).",
}

TARGET_COLUMNS = ["purchase_flag", "returned_flag", "added_to_cart", "wishlist_flag", "rating_given"]


# --------------------------------------------------------------------------- #
# Loading & inspection
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_INTERACTIONS)
    return df


def load_policies() -> Dict[str, str]:
    with open(POLICIES_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def inspect(df: pd.DataFrame) -> Dict[str, object]:
    """Return a compact inspection report used by notebook 01."""
    report: Dict[str, object] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "n_duplicates": int(df.duplicated().sum()),
        "target_balance": {
            c: {str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()}
            for c in TARGET_COLUMNS
            if c in df.columns
        },
    }
    return report


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit, documented cleaning.

    Decisions
    ---------
    * ``size`` is missing for non-sized items (accessories, watches).  We do NOT
      impute a fake size; we fill with the explicit token ``"Not Applicable"``
      and keep a ``size_missing`` indicator so the model can use the signal.
    * Any other missing numeric value is median-imputed; any other missing
      categorical value is filled with ``"Unknown"``.
    * Duplicate rows are dropped.
    """
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # size: explicit sentinel + indicator
    df["size_missing"] = df["size"].isna().astype(int)
    df["size"] = df["size"].fillna("Not Applicable")

    # generic fallbacks
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")

    # sanity: prices should be positive
    df = df[df["final_price"] > 0].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _price_band(p: float) -> str:
    if p < 1000:
        return "Budget"
    if p < 3000:
        return "Mid"
    if p < 6000:
        return "Premium"
    return "Luxury"


def _discount_band(d: float) -> str:
    if d <= 0:
        return "None"
    if d <= 15:
        return "Low"
    if d <= 35:
        return "Medium"
    return "High"


def _recency_band(days: int) -> str:
    if days <= 30:
        return "Active"
    if days <= 90:
        return "Warm"
    if days <= 180:
        return "Cooling"
    return "Dormant"


def _product_signature(row: pd.Series) -> str:
    """Deterministic product identity from the attribute tuple.

    The provided dataset reuses each ``product_id`` across five rows whose
    attributes differ, so ``product_id`` cannot identify a product.  The real
    identity is the descriptive attribute tuple, which we hash into a stable
    ``product_key``.
    """
    sig = "|".join(
        str(row[c]).strip().lower()
        for c in ("product_title", "category", "sub_category", "brand",
                  "color", "material", "season", "recommended_use_case")
    )
    return "SKU_" + hashlib.md5(sig.encode("utf-8")).hexdigest()[:10]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_key"] = df.apply(_product_signature, axis=1)
    df["price_band"] = df["final_price"].apply(_price_band)
    df["discount_band"] = df["discount_percent"].apply(_discount_band)
    df["is_discounted"] = (df["discount_percent"] > 0).astype(int)
    df["price_sensitivity"] = (df["discount_percent"] / (df["base_price"] / 1000.0)).round(4)
    df["engagement_score"] = (
        (df["time_on_page"] / df["time_on_page"].max()).round(4)
        + 0.5 * df["added_to_cart"]
        + 0.5 * df["wishlist_flag"]
    ).round(4)
    df["recency_band"] = df["days_since_last_purchase"].apply(_recency_band)
    df["customer_value"] = (df["avg_order_value"] * df["total_orders"]).round(2)
    df["product_text"] = df.apply(build_product_text, axis=1)
    return df


def build_product_text(row: pd.Series) -> str:
    """Retrieval-ready product document (Phase 3 core artefact)."""
    return (
        f"{row['product_title']}. "
        f"Category: {row['category']} / {row['sub_category']}. "
        f"Brand: {row['brand']}. Colour: {row['color']}. Material: {row['material']}. "
        f"Season: {row['season']}. Use case: {row['recommended_use_case']}. "
        f"Style tags: {row['style_tags']}. Occasion tags: {row['occasion_tags']}. "
        f"Price: Rs.{row['final_price']:.0f} (base Rs.{row['base_price']:.0f}, "
        f"{row['discount_percent']}% off). Rating: {row['avg_product_rating']}. "
        f"{row['product_description']}"
    )


# --------------------------------------------------------------------------- #
# Data dictionary
# --------------------------------------------------------------------------- #
def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for col in df.columns:
        if col in ORIGINAL_COLUMNS:
            provenance, desc = "original", ORIGINAL_COLUMNS[col]
        elif col in ENGINEERED_COLUMNS:
            provenance, desc = "engineered", ENGINEERED_COLUMNS[col]
        else:
            provenance, desc = "original", "Column present in the source dataset."
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "provenance": provenance,
                "n_missing": int(df[col].isna().sum()),
                "n_unique": int(df[col].nunique(dropna=True)),
                "example": str(df[col].dropna().iloc[0])[:60] if df[col].notna().any() else "",
                "description": desc,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> Dict[str, object]:
    ensure_dirs()
    raw = load_raw()
    report = inspect(raw)

    clean_df = clean(raw)
    feat_df = engineer_features(clean_df)

    clean_df.to_csv(CLEAN_INTERACTIONS, index=False)
    feat_df.to_csv(FEATURES_CSV, index=False)

    dd = build_data_dictionary(feat_df)
    dd.to_csv(DATA_DICTIONARY_CSV, index=False)

    summary = {
        "raw_shape": list(raw.shape),
        "clean_shape": list(clean_df.shape),
        "feature_shape": list(feat_df.shape),
        "n_original_columns": int((dd["provenance"] == "original").sum()),
        "n_engineered_columns": int((dd["provenance"] == "engineered").sum()),
        "missing_after_clean": int(feat_df.isna().sum().sum()),
        "inspection": report,
    }
    with open(PROCESSED_DIR / "data_prep_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":  # pragma: no cover
    s = run()
    print(json.dumps({k: v for k, v in s.items() if k != "inspection"}, indent=2))
    print("Missing after clean:", s["missing_after_clean"])
