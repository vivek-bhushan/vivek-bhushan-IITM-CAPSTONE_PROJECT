"""
enrich_genai.py
===============
Phase 3 - Enrich Data for Generative AI.

Turns the structured retail records into **retrieval-ready documents**:

* one product document per unique product (title + description + attributes +
  price context + tags), with a stable ``doc_id``;
* one policy document per policy snippet, with a stable ``doc_id``;
* a small set of synthetic customer queries for retrieval testing.

Outputs JSONL files that the RAG / policy-QA / agent modules consume.

Run directly:  ``python src/enrich_genai.py``
"""

from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd

from config import (
    FEATURES_CSV,
    POLICY_DOCS_JSONL,
    PRODUCT_DOCS_JSONL,
    ensure_dirs,
)
from data_prep import load_policies


# --------------------------------------------------------------------------- #
# Product documents
# --------------------------------------------------------------------------- #
def build_product_documents(df: pd.DataFrame) -> List[Dict[str, object]]:
    """One document per unique product_key (the stable product identity).

    ``product_id`` is not unique in the source data, so we group on the
    engineered ``product_key`` and keep the richest row (highest rating) as the
    canonical product record.
    """
    key_col = "product_key" if "product_key" in df.columns else "product_id"
    prod = (
        df.sort_values("avg_product_rating", ascending=False)
        .drop_duplicates(subset=key_col, keep="first")
        .reset_index(drop=True)
    )

    docs: List[Dict[str, object]] = []
    for _, r in prod.iterrows():
        text = (
            f"Product: {r['product_title']}. "
            f"Category: {r['category']} > {r['sub_category']}. "
            f"Brand: {r['brand']}. Colour: {r['color']}. Material: {r['material']}. "
            f"Size: {r['size']}. Season: {r['season']}. "
            f"Recommended use: {r['recommended_use_case']}. "
            f"Style tags: {r['style_tags']}. Occasion tags: {r['occasion_tags']}. "
            f"Price: Rs.{r['final_price']:.0f} (base Rs.{r['base_price']:.0f}, "
            f"{r['discount_percent']}% discount). "
            f"Average rating: {r['avg_product_rating']} out of 5. "
            f"Stock: {r['stock_level']} units. "
            f"Description: {r['product_description']}"
        )
        docs.append(
            {
                "doc_id": f"PROD_{r['product_key']}",
                "doc_type": "product",
                "product_id": r["product_key"],
                "source_product_id": r["product_id"],
                "title": r["product_title"],
                "text": text,
                "metadata": {
                    "category": r["category"],
                    "sub_category": r["sub_category"],
                    "brand": r["brand"],
                    "color": r["color"],
                    "material": r["material"],
                    "season": r["season"],
                    "use_case": r["recommended_use_case"],
                    "style_tags": r["style_tags"],
                    "occasion_tags": r["occasion_tags"],
                    "final_price": float(r["final_price"]),
                    "base_price": float(r["base_price"]),
                    "discount_percent": int(r["discount_percent"]),
                    "avg_product_rating": float(r["avg_product_rating"]),
                    "stock_level": int(r["stock_level"]),
                    "is_returnable": int(r["is_returnable"]),
                },
            }
        )
    return docs


# --------------------------------------------------------------------------- #
# Policy documents
# --------------------------------------------------------------------------- #
POLICY_TITLES = {
    "POL_RETURNS": "Returns Policy",
    "POL_EXCHANGE": "Exchange Policy",
    "POL_SHIPPING": "Shipping & Delivery Policy",
    "POL_REFUND": "Refund Policy",
    "POL_WARRANTY": "Warranty Policy",
}


def build_policy_documents(policies: Dict[str, str]) -> List[Dict[str, object]]:
    docs: List[Dict[str, object]] = []
    for doc_id, text in policies.items():
        title = POLICY_TITLES.get(doc_id, doc_id)
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": "policy",
                "title": title,
                "text": f"{title}: {text}",
                "metadata": {"policy_id": doc_id, "title": title},
            }
        )
    return docs


# --------------------------------------------------------------------------- #
# Synthetic queries (for retrieval testing / regression checks)
# --------------------------------------------------------------------------- #
def build_synthetic_queries(df: pd.DataFrame) -> List[Dict[str, object]]:
    """A handful of realistic queries with the product they should retrieve."""
    key_col = "product_key" if "product_key" in df.columns else "product_id"
    sample = df.drop_duplicates(key_col).head(200)
    queries: List[Dict[str, object]] = []

    # derive a few queries directly from the dataset's own customer_query field
    for _, r in sample.sample(5, random_state=42).iterrows():
        queries.append(
            {
                "query": r["customer_query"],
                "expected_doc_id": f"PROD_{r[key_col]}",
                "source": "dataset",
            }
        )

    # hand-written queries covering the capstone's example set
    queries.extend(
        [
            {"query": "casual sneakers under Rs.4000 for daily wear",
             "expected_doc_id": None, "source": "handwritten"},
            {"query": "black formal leather shoes for office",
             "expected_doc_id": None, "source": "handwritten"},
            {"query": "premium watch with warranty",
             "expected_doc_id": None, "source": "handwritten"},
            {"query": "blue backpack for travel",
             "expected_doc_id": None, "source": "handwritten"},
        ]
    )
    return queries


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def run() -> Dict[str, object]:
    ensure_dirs()
    df = pd.read_csv(FEATURES_CSV)
    policies = load_policies()

    product_docs = build_product_documents(df)
    policy_docs = build_policy_documents(policies)
    queries = build_synthetic_queries(df)

    _write_jsonl(PRODUCT_DOCS_JSONL, product_docs)
    _write_jsonl(POLICY_DOCS_JSONL, policy_docs)

    summary = {
        "n_product_docs": len(product_docs),
        "n_policy_docs": len(policy_docs),
        "n_synthetic_queries": len(queries),
        "avg_product_doc_chars": round(
            sum(len(d["text"]) for d in product_docs) / max(1, len(product_docs)), 1
        ),
    }
    with open(PRODUCT_DOCS_JSONL.parent / "enrich_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2))
