"""
agent_controller.py
===================
Phase 5 - Agent / Controller.

A deliberately **simple, explicit and inspectable** routing layer.  It is a
hybrid rule-based + keyword-scoring controller (no opaque framework), which is
exactly what the capstone asks for: "keep routing bounded and simple".

Tools
-----
* ``product_search``   - RAG product recommendation (Phase 4).
* ``policy_qa``        - grounded policy question answering (Phase 4).
* ``analytics``        - customer-segment / price-sensitivity analytics (Phase 2).
* ``review_summary``   - summarise review sentiment for a product/category.
* ``calculator``       - safe arithmetic for price/discount questions.

Every call returns a **trace**: the detected intent, the tool chosen, the
reason, the arguments, and the tool result.  Routing is bounded to a single
tool per query (no unbounded loops).

Run directly:  ``python src/agent_controller.py``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import pandas as pd

from config import (
    AGENT_TRACE_JSON,
    FEATURES_CSV,
    SEGMENTS_CSV,
    ensure_dirs,
)
from policy_qa import build_index as build_policy_index, answer_policy_question
from rag_recommendation import build_index as build_product_index, recommend


# --------------------------------------------------------------------------- #
# Intent detection
# --------------------------------------------------------------------------- #
POLICY_KEYWORDS = [
    "return", "refund", "exchange", "shipping", "delivery", "warranty",
    "policy", "cancel", "replace", "guarantee", "damaged", "defective",
]
ANALYTICS_KEYWORDS = [
    "segment", "segments", "cluster", "price-sensitive", "price sensitive",
    "which customers", "customer group", "persona", "churn", "loyal",
]
REVIEW_KEYWORDS = ["review", "reviews", "sentiment", "feedback", "opinion", "summarise", "summarize"]
CALC_KEYWORDS = ["calculate", "how much", "total price", "discount of", "percent off", "gst", "tax"]
PRODUCT_KEYWORDS = [
    "recommend", "suggest", "show", "find", "buy", "looking for", "need",
    "shoes", "sneakers", "backpack", "watch", "shirt", "jeans", "dress",
    "jacket", "sandals", "accessories", "under", "premium", "casual", "formal",
]


def detect_intent(query: str) -> Dict[str, object]:
    """Score each intent by keyword overlap; return the winner + scores."""
    low = query.lower()
    scores = {
        "policy_qa": sum(1 for k in POLICY_KEYWORDS if k in low),
        "analytics": sum(1 for k in ANALYTICS_KEYWORDS if k in low),
        "review_summary": sum(1 for k in REVIEW_KEYWORDS if k in low),
        "calculator": sum(1 for k in CALC_KEYWORDS if k in low),
        "product_search": sum(1 for k in PRODUCT_KEYWORDS if k in low),
    }
    # policy questions often contain "return"/"refund" AND a product word;
    # policy intent wins when it has any signal, because it is the riskier path.
    if scores["policy_qa"] > 0:
        winner = "policy_qa"
    else:
        winner = max(scores, key=scores.get)
        if scores[winner] == 0:
            winner = "product_search"  # safe default
    return {"intent": winner, "scores": scores}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class Tools:
    def __init__(self):
        self.df = pd.read_csv(FEATURES_CSV)
        self.segments = pd.read_csv(SEGMENTS_CSV)
        self.product_index = build_product_index()
        self.policy_index = build_policy_index()

    # -- product search ---------------------------------------------------- #
    def product_search(self, query: str, **kw) -> Dict[str, object]:
        return recommend(self.product_index, query, top_k=kw.get("top_k", 5))

    # -- policy QA --------------------------------------------------------- #
    def policy_qa(self, query: str, **kw) -> Dict[str, object]:
        return answer_policy_question(self.policy_index, query)

    # -- analytics --------------------------------------------------------- #
    def analytics(self, query: str, **kw) -> Dict[str, object]:
        low = query.lower()
        if "price" in low and "sensitiv" in low:
            prof = (
                self.segments.groupby("segment")
                .agg(price_sensitivity=("price_sensitivity", "mean"),
                     avg_discount=("discount_percent", "mean"),
                     n=("customer_id", "count"))
                .round(3).reset_index()
            )
            top = prof.sort_values("price_sensitivity", ascending=False).iloc[0]
            return {
                "analysis": "price_sensitivity_by_segment",
                "most_price_sensitive_segment": int(top["segment"]),
                "table": prof.to_dict(orient="records"),
                "answer": (
                    f"Segment {int(top['segment'])} is the most price-sensitive "
                    f"(mean price-sensitivity {top['price_sensitivity']}, "
                    f"avg discount {top['avg_discount']}%)."
                ),
            }
        # default: segment overview
        prof = (
            self.segments.groupby("segment")
            .agg(n=("customer_id", "count"),
                 avg_order_value=("avg_order_value", "mean"),
                 recency=("days_since_last_purchase", "mean"),
                 purchase_rate=("purchase_rate", "mean"))
            .round(3).reset_index()
        )
        return {
            "analysis": "segment_overview",
            "table": prof.to_dict(orient="records"),
            "answer": f"There are {len(prof)} customer segments in the data.",
        }

    # -- review summary ---------------------------------------------------- #
    def review_summary(self, query: str, **kw) -> Dict[str, object]:
        df = self.df
        # try to narrow by a category keyword
        cat = None
        for c in df["category"].dropna().unique():
            if str(c).lower() in query.lower():
                cat = c
                break
        sub = df[df["category"] == cat] if cat else df
        dist = sub["review_sentiment_label"].value_counts(normalize=True).round(3).to_dict()
        top_reviews = sub["customer_review_text"].dropna().head(3).tolist()
        return {
            "scope": cat or "all categories",
            "sentiment_distribution": dist,
            "sample_reviews": top_reviews,
            "answer": (
                f"Across {len(sub)} interactions"
                + (f" in {cat}" if cat else "")
                + f", sentiment is {dist}. "
                + "Representative review: " + (top_reviews[0] if top_reviews else "n/a")
            ),
        }

    # -- calculator -------------------------------------------------------- #
    def calculator(self, query: str, **kw) -> Dict[str, object]:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", query)]
        low = query.lower()
        if len(nums) >= 2 and ("discount" in low or "off" in low or "%" in low):
            price, pct = nums[0], nums[1]
            final = price * (1 - pct / 100.0)
            return {
                "operation": "discount",
                "inputs": {"price": price, "discount_percent": pct},
                "result": round(final, 2),
                "answer": f"Rs.{price:.0f} with {pct:.0f}% off is Rs.{final:.2f}.",
            }
        if len(nums) >= 2:
            return {
                "operation": "sum",
                "inputs": nums,
                "result": round(sum(nums), 2),
                "answer": f"The sum is {sum(nums):.2f}.",
            }
        return {"operation": "none", "answer": "I could not find numbers to calculate with."}


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
@dataclass
class Trace:
    query: str
    intent: str
    intent_scores: Dict[str, int]
    tool: str
    reason: str
    arguments: Dict[str, object]
    status: str
    result_summary: str


class AgentController:
    def __init__(self, tools: Optional[Tools] = None):
        self.tools = tools or Tools()
        self.registry: Dict[str, Callable] = {
            "product_search": self.tools.product_search,
            "policy_qa": self.tools.policy_qa,
            "analytics": self.tools.analytics,
            "review_summary": self.tools.review_summary,
            "calculator": self.tools.calculator,
        }

    def route(self, query: str) -> Dict[str, object]:
        det = detect_intent(query)
        intent = det["intent"]
        tool = self.registry[intent]
        reason = f"keyword scores {det['scores']} -> highest-signal intent '{intent}'"

        try:
            result = tool(query)
            status = "ok"
        except Exception as exc:  # bounded, explicit failure handling
            result = {"error": str(exc)}
            status = "error"

        summary = self._summarise(intent, result)
        trace = Trace(
            query=query, intent=intent, intent_scores=det["scores"],
            tool=intent, reason=reason, arguments={"query": query},
            status=status, result_summary=summary,
        )
        return {"trace": asdict(trace), "result": result}

    @staticmethod
    def _summarise(intent: str, result: Dict[str, object]) -> str:
        if intent == "product_search":
            ids = result.get("supporting_doc_ids", [])
            return f"retrieved {len(ids)} products: {ids}"
        if intent == "policy_qa":
            return f"answered from {result.get('primary_source')} (conf {result.get('confidence')})"
        if intent == "analytics":
            return result.get("answer", "analytics computed")
        if intent == "review_summary":
            return f"sentiment {result.get('sentiment_distribution')}"
        if intent == "calculator":
            return result.get("answer", "calculated")
        return "done"


# --------------------------------------------------------------------------- #
# Demo queries (the capstone's example set + extras)
# --------------------------------------------------------------------------- #
DEMO_QUERIES = [
    "Recommend casual sneakers under Rs.4000 for daily wear.",
    "Can I return a discounted product after 7 days?",
    "Suggest products for a premium customer who prefers black formal shoes.",
    "Which customer segment seems most price-sensitive?",
    "Summarise the top three retrieved products and explain why they fit the query.",
    "How long does standard delivery take?",
    "What is the sentiment in reviews for Shoes?",
    "Calculate the price of 2500 with 20% discount.",
]


def run() -> Dict[str, object]:
    ensure_dirs()
    agent = AgentController()
    traces = []
    for q in DEMO_QUERIES:
        out = agent.route(q)
        traces.append(out)

    with open(AGENT_TRACE_JSON, "w", encoding="utf-8") as fh:
        json.dump(traces, fh, indent=2, default=str)

    routing = {}
    for t in traces:
        routing[t["trace"]["intent"]] = routing.get(t["trace"]["intent"], 0) + 1
    return {"n_queries": len(traces), "routing_distribution": routing, "traces": traces}


if __name__ == "__main__":  # pragma: no cover
    out = run()
    print("Routing distribution:", out["routing_distribution"])
    for t in out["traces"]:
        tr = t["trace"]
        print(f"\nQ: {tr['query']}\n  intent={tr['intent']} tool={tr['tool']} "
              f"status={tr['status']}\n  -> {tr['result_summary']}")
