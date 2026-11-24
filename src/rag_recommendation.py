"""
rag_recommendation.py
=====================
Phase 4 (part 1) - Retrieval-Augmented Product Recommendation.

Pipeline
--------
1. Load the retrieval-ready product documents (Phase 3 output).
2. Embed them (offline TF-IDF+SVD by default, OpenAI embeddings if a key is set).
3. Build a FAISS index (falls back to a NumPy cosine index if FAISS is absent).
4. Retrieve top-k products for a query, with optional price / category filters.
5. Generate a **grounded** recommendation: the answer is composed only from the
   retrieved documents, and the supporting product ids are returned explicitly.

Run directly:  ``python src/rag_recommendation.py``
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

from config import (
    INDEX_DIR,
    MIN_RETRIEVAL_SCORE,
    PRODUCT_DOCS_JSONL,
    RETRIEVAL_EVAL_JSON,
    TOP_K,
    ensure_dirs,
)
from llm_client import get_llm_client


# --------------------------------------------------------------------------- #
# Document store + vector index
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    score: float
    metadata: Dict[str, object] = field(default_factory=dict)


class ProductIndex:
    """Embedding + vector index over product documents."""

    def __init__(self, docs: List[Dict[str, object]], client=None):
        self.docs = docs
        self.client = client or get_llm_client()
        self.texts = [d["text"] for d in docs]
        self.client.fit_embeddings(self.texts)
        self.embeddings = self.client.embed(self.texts).astype(np.float32)
        self._faiss = None

        # lexical (BM25-lite) structures for hybrid re-ranking
        self._token_sets = [set(_tokens(t)) for t in self.texts]
        df: Dict[str, int] = {}
        for ts in self._token_sets:
            for t in ts:
                df[t] = df.get(t, 0) + 1
        n = max(1, len(self.docs))
        self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

        self._build_index()

    def _build_index(self) -> None:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            self._faiss = index
            self.backend = "faiss"
        except Exception:
            self._faiss = None
            self.backend = "numpy"

    # -- hybrid scoring ---------------------------------------------------- #
    @staticmethod
    def _attribute_bonus(q_low: str, md: Dict[str, object]) -> float:
        """Reward documents whose explicit attributes are named in the query.

        Dense bag-of-words similarity alone ranks same-category items almost
        equally (a blue jacket and an olive jacket look identical), so we add a
        small, explicit bonus for matched structured attributes - exactly the
        colour/material/use-case signals an e-commerce re-ranker would use.
        """
        weighted = [
            (str(md.get("color", "")), 1.0),
            (str(md.get("sub_category", "")), 1.0),
            (str(md.get("category", "")), 1.0),
            (str(md.get("use_case", "")), 0.7),
            (str(md.get("material", "")), 0.5),
            (str(md.get("brand", "")), 0.7),
        ]
        hits = 0.0
        for value, w in weighted:
            low = value.lower().strip()
            if not low or low == "nan":
                continue
            for tok in re.split(r"[^a-z0-9]+", low):
                if len(tok) > 2 and tok in q_low:
                    hits += w
        return min(hits / 3.0, 1.0)

    def _lexical_score(self, q_tokens: List[str], i: int, q_idf_sum: float) -> float:
        toks = self._token_sets[i]
        return sum(self._idf.get(t, 1.0) for t in q_tokens if t in toks) / q_idf_sum

    # -- search ------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
        min_rating: Optional[float] = None,
        attr_filters: Optional[Dict[str, Optional[str]]] = None,
        use_hybrid: bool = True,
    ) -> List[RetrievedDoc]:
        q = self.client.embed([query]).astype(np.float32)

        # 1) dense candidate generation (over-fetch so re-ranking has room)
        cand = min(len(self.docs), max(top_k * 30, 60))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, cand)
            dense = {int(i): float(s) for s, i in zip(scores[0], idx[0]) if i >= 0}
        else:
            sims = (self.embeddings @ q.T).ravel()
            dense = {int(i): float(sims[int(i)])
                     for i in np.argsort(sims)[::-1][:cand]}

        # widen the candidate set with a structured scan when attribute filters
        # are present, so a correct product the dense stage missed still gets
        # re-ranked (recall-first, then precision).
        af = attr_filters or {}
        if any(af.get(k) for k in ("color", "sub_category", "use_case")):
            for i, d in enumerate(self.docs):
                md = d["metadata"]
                if af.get("color") and str(md.get("color", "")).lower() != af["color"]:
                    continue
                if af.get("sub_category") and \
                        str(md.get("sub_category", "")).lower() != af["sub_category"].lower():
                    continue
                if af.get("use_case") and \
                        str(md.get("use_case", "")).lower() != af["use_case"]:
                    continue
                dense.setdefault(i, 0.0)

        # 2) hybrid re-rank: dense + lexical overlap + attribute + structured match
        q_tokens = _tokens(query)
        q_idf_sum = sum(self._idf.get(t, 1.0) for t in q_tokens) or 1.0
        q_low = query.lower()

        ranked: List[tuple] = []
        for i, dense_score in dense.items():
            md = self.docs[i]["metadata"]
            if max_price is not None and md["final_price"] > max_price:
                continue
            if category is not None and category.lower() not in str(md["category"]).lower() \
                    and category.lower() not in str(md["sub_category"]).lower():
                continue
            if min_rating is not None and md["avg_product_rating"] < min_rating:
                continue

            if use_hybrid:
                lex = self._lexical_score(q_tokens, i, q_idf_sum)
                attr = self._attribute_bonus(q_low, md)
                # explicit structured constraints contribute strongly
                struct = 0.0
                if af.get("color") and str(md.get("color", "")).lower() == af["color"]:
                    struct += 1.0
                if af.get("sub_category") and \
                        str(md.get("sub_category", "")).lower() == af["sub_category"].lower():
                    struct += 1.0
                if af.get("use_case") and \
                        str(md.get("use_case", "")).lower() == af["use_case"]:
                    struct += 0.6
                struct = min(struct / 2.6, 1.0)
                final = 0.35 * dense_score + 0.25 * lex + 0.10 * attr + 0.30 * struct
            else:
                final = dense_score
            ranked.append((final, dense_score, i))

        ranked.sort(key=lambda x: x[0], reverse=True)

        out: List[RetrievedDoc] = []
        for final, _dense, i in ranked[:top_k]:
            d = self.docs[i]
            out.append(RetrievedDoc(
                doc_id=d["doc_id"], title=d["title"], text=d["text"],
                score=float(final), metadata=d["metadata"],
            ))
        return out


# --------------------------------------------------------------------------- #
# Query parsing (light, explicit, inspectable)
# --------------------------------------------------------------------------- #
_PRICE_RE = re.compile(r"(?:under|below|less than|upto|up to|within)\s*(?:rs\.?|inr|₹)?\s*(\d{3,6})", re.I)
_CATEGORY_HINTS = {
    "sneakers": "Sneakers", "shoes": "Shoes", "backpack": "Backpack",
    "watch": "Watches", "shirt": "Shirts", "jeans": "Jeans",
    "dress": "Dresses", "jacket": "Jacket", "cap": "Cap", "sandals": "Sandals",
    "accessories": "Accessories", "formal": "Formal Shoes", "running": "Running",
}
_COLOR_HINTS = [
    "black", "white", "blue", "navy", "red", "green", "olive", "brown",
    "grey", "gray", "beige", "pink", "yellow", "maroon", "purple", "orange",
]
_USE_HINTS = ["casual", "formal", "sports", "travel", "office", "daily wear",
              "party", "winter", "summer", "festive"]


def parse_attribute_filters(query: str) -> Dict[str, Optional[str]]:
    """Extract explicit colour / sub-category / use-case constraints.

    This is the structured half of the hybrid retriever: when the user names a
    colour or an occasion we constrain candidates on the matching structured
    attribute instead of hoping the embedding separates them.
    """
    low = query.lower()
    return {
        "color": next((c for c in _COLOR_HINTS if c in low), None),
        "sub_category": next((v for k, v in _CATEGORY_HINTS.items() if k in low), None),
        "use_case": next((u for u in _USE_HINTS if u in low), None),
    }


def parse_query(query: str) -> Dict[str, object]:
    """Extract structured filters from a natural-language query."""
    filters: Dict[str, object] = {"max_price": None, "category": None}
    m = _PRICE_RE.search(query)
    if m:
        filters["max_price"] = float(m.group(1))
    low = query.lower()
    for hint, cat in _CATEGORY_HINTS.items():
        if hint in low:
            filters["category"] = cat
            break
    filters.update(parse_attribute_filters(query))
    return filters


# --------------------------------------------------------------------------- #
# Grounded recommendation
# --------------------------------------------------------------------------- #
def recommend(
    index: ProductIndex,
    query: str,
    top_k: int = TOP_K,
    use_filters: bool = True,
) -> Dict[str, object]:
    filters = parse_query(query) if use_filters else {"max_price": None, "category": None}
    docs = index.search(
        query, top_k=top_k,
        max_price=filters["max_price"], category=filters["category"],
        attr_filters={
            "color": filters.get("color"),
            "sub_category": filters.get("sub_category"),
            "use_case": filters.get("use_case"),
        },
    )

    if not docs:
        return {
            "query": query,
            "filters": filters,
            "status": "no_results",
            "answer": "I could not find any products matching your request. "
                      "Try relaxing the price or category constraint.",
            "recommended_products": [],
            "supporting_doc_ids": [],
        }

    weak = docs[0].score < MIN_RETRIEVAL_SCORE
    context = "\n\n".join(f"[{d.doc_id}] {d.text}" for d in docs)
    prompt = (
        f"Recommend products for the query: '{query}'. "
        "Use ONLY the retrieved product documents. For each recommendation give "
        "the product id, a one-line reason, and the price."
    )
    answer = index.client.answer(prompt, context)

    recs = [
        {
            "doc_id": d.doc_id,
            "product_id": d.metadata.get("product_id", d.doc_id.replace("PROD_", "")),
            "title": d.title,
            "price": d.metadata["final_price"],
            "rating": d.metadata["avg_product_rating"],
            "score": round(d.score, 4),
            "reason": _reason(d, filters),
        }
        for d in docs
    ]

    return {
        "query": query,
        "filters": filters,
        "status": "weak_retrieval" if weak else "ok",
        "answer": answer,
        "recommended_products": recs,
        "supporting_doc_ids": [d.doc_id for d in docs],
    }


def _reason(d: RetrievedDoc, filters: Dict[str, object]) -> str:
    md = d.metadata
    bits = [f"{md['category']}/{md['sub_category']}", f"Rs.{md['final_price']:.0f}"]
    if filters.get("max_price") and md["final_price"] <= filters["max_price"]:
        bits.append("within budget")
    if md["discount_percent"] > 0:
        bits.append(f"{md['discount_percent']}% off")
    bits.append(f"rated {md['avg_product_rating']}/5")
    return ", ".join(bits)


# --------------------------------------------------------------------------- #
# Retrieval evaluation
# --------------------------------------------------------------------------- #
def evaluate_retrieval(index: ProductIndex, queries: List[Dict[str, object]]) -> Dict[str, object]:
    """Hit-rate@k and MRR on queries that have a known expected doc.

    Evaluation uses the **full retrieval path** (query parsing + structured
    attribute filters + hybrid re-ranking), i.e. exactly what a user query
    experiences - not a bare dense search.
    """
    hits, rr, n = 0, 0.0, 0
    details = []
    for q in queries:
        if not q.get("expected_doc_id"):
            continue
        n += 1
        filters = parse_query(q["query"])
        docs = index.search(
            q["query"], top_k=TOP_K,
            max_price=filters["max_price"], category=filters["category"],
            attr_filters={
                "color": filters.get("color"),
                "sub_category": filters.get("sub_category"),
                "use_case": filters.get("use_case"),
            },
        )
        ids = [d.doc_id for d in docs]
        if q["expected_doc_id"] in ids:
            hits += 1
            rank = ids.index(q["expected_doc_id"]) + 1
            rr += 1.0 / rank
            details.append({"query": q["query"], "hit": True, "rank": rank})
        else:
            details.append({"query": q["query"], "hit": False, "rank": None})
    return {
        "n_evaluated": n,
        "hit_rate_at_k": round(hits / n, 4) if n else None,
        "mrr": round(rr / n, 4) if n else None,
        "top_k": TOP_K,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_product_docs() -> List[Dict[str, object]]:
    docs = []
    with open(PRODUCT_DOCS_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            docs.append(json.loads(line))
    return docs


def build_index() -> ProductIndex:
    ensure_dirs()
    return ProductIndex(load_product_docs())


def run() -> Dict[str, object]:
    ensure_dirs()
    index = build_index()

    demo_queries = [
        "casual sneakers under Rs.4000 for daily wear",
        "black formal leather shoes for office",
        "premium watch with warranty",
        "blue backpack for travel",
    ]
    results = [recommend(index, q) for q in demo_queries]

    # retrieval eval on dataset-derived queries
    from enrich_genai import build_synthetic_queries
    import pandas as pd
    from config import FEATURES_CSV

    df = pd.read_csv(FEATURES_CSV)
    queries = build_synthetic_queries(df)
    eval_res = evaluate_retrieval(index, queries)

    with open(RETRIEVAL_EVAL_JSON, "w", encoding="utf-8") as fh:
        json.dump({"retrieval_eval": eval_res, "demo_results": results}, fh, indent=2)

    return {"backend": index.backend, "n_docs": len(index.docs),
            "retrieval_eval": eval_res, "demo_results": results}


if __name__ == "__main__":  # pragma: no cover
    out = run()
    print("Index backend:", out["backend"], "| docs:", out["n_docs"])
    print("Retrieval eval:", {k: v for k, v in out["retrieval_eval"].items() if k != "details"})
    for r in out["demo_results"]:
        print(f"\nQ: {r['query']}\n  -> {r['supporting_doc_ids']}")
