"""
evaluation.py
=============
Final Phase - Evaluation and Packaging.

Consolidates every result into a single evaluation summary and runs
**regression-like checks** so the whole system can be validated in CI:

* ML metric thresholds (ROC-AUC, F1) for the supervised model.
* Retrieval hit-rate threshold.
* Policy-QA citation accuracy threshold.
* Agent routing correctness on a labelled query set.
* Grounding check: policy answers must be substrings of retrieved evidence.

Run directly:  ``python src/evaluation.py``
"""

from __future__ import annotations

import json
from typing import Dict, List

from config import (
    AGENT_TRACE_JSON,
    EVALUATION_JSON,
    EXAMPLE_QUERIES_JSON,
    ML_METRICS_JSON,
    POLICY_QA_JSON,
    RETRIEVAL_EVAL_JSON,
    SEGMENT_SUMMARY_JSON,
    ensure_dirs,
)
from agent_controller import AgentController, DEMO_QUERIES
from policy_qa import POLICY_TEST_QUESTIONS, build_index as build_policy_index, answer_policy_question
from rag_recommendation import build_index as build_product_index, recommend


# --------------------------------------------------------------------------- #
# Regression thresholds
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "supervised_roc_auc": 0.60,
    "supervised_f1": 0.55,
    "retrieval_hit_rate": 0.50,
    "policy_citation_accuracy": 0.80,
    "agent_routing_accuracy": 0.75,
}

# labelled routing set: query -> expected intent
ROUTING_LABELS = {
    "Recommend casual sneakers under Rs.4000 for daily wear.": "product_search",
    "Can I return a discounted product after 7 days?": "policy_qa",
    "Suggest products for a premium customer who prefers black formal shoes.": "product_search",
    "Which customer segment seems most price-sensitive?": "analytics",
    "How long does standard delivery take?": "policy_qa",
    "What is the sentiment in reviews for Shoes?": "review_summary",
    "Calculate the price of 2500 with 20% discount.": "calculator",
    "Is physical damage covered under warranty?": "policy_qa",
}


def _load(path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run() -> Dict[str, object]:
    ensure_dirs()

    ml = _load(ML_METRICS_JSON)
    seg = _load(SEGMENT_SUMMARY_JSON)
    ret = _load(RETRIEVAL_EVAL_JSON)
    pol = _load(POLICY_QA_JSON)

    # ---- agent routing accuracy ----------------------------------------- #
    agent = AgentController()
    routing_correct, routing_details = 0, []
    for q, expected in ROUTING_LABELS.items():
        out = agent.route(q)
        got = out["trace"]["intent"]
        ok = got == expected
        routing_correct += int(ok)
        routing_details.append({"query": q, "expected": expected, "got": got, "ok": ok})
    routing_acc = routing_correct / len(ROUTING_LABELS)

    # ---- grounding check ------------------------------------------------- #
    pol_index = build_policy_index()
    grounding_ok, grounding_details = 0, []
    for q in POLICY_TEST_QUESTIONS:
        r = answer_policy_question(pol_index, q["question"])
        if r["status"] != "answered":
            grounding_details.append({"question": q["question"], "grounded": False,
                                      "reason": "no evidence"})
            continue
        # the answer must be a substring of the retrieved evidence text
        evidence_text = " ".join(e["text"] for e in r["evidence"])
        grounded = r["answer"].strip() in evidence_text
        grounding_ok += int(grounded)
        grounding_details.append({"question": q["question"], "grounded": grounded,
                                  "source": r.get("primary_source")})
    grounding_rate = grounding_ok / len(POLICY_TEST_QUESTIONS)

    # ---- curated example queries with outputs ---------------------------- #
    prod_index = build_product_index()
    example_queries = []
    for q in DEMO_QUERIES:
        out = agent.route(q)
        example_queries.append({
            "query": q,
            "intent": out["trace"]["intent"],
            "tool": out["trace"]["tool"],
            "status": out["trace"]["status"],
            "summary": out["trace"]["result_summary"],
            "result": out["result"],
        })
    with open(EXAMPLE_QUERIES_JSON, "w", encoding="utf-8") as fh:
        json.dump(example_queries, fh, indent=2, default=str)

    # ---- assemble checks ------------------------------------------------- #
    checks = [
        {"name": "supervised_roc_auc",
         "value": ml["supervised"]["best_roc_auc"],
         "threshold": THRESHOLDS["supervised_roc_auc"],
         "passed": ml["supervised"]["best_roc_auc"] >= THRESHOLDS["supervised_roc_auc"]},
        {"name": "supervised_f1",
         "value": ml["supervised"]["models"][ml["supervised"]["best_model"]]["f1"],
         "threshold": THRESHOLDS["supervised_f1"],
         "passed": ml["supervised"]["models"][ml["supervised"]["best_model"]]["f1"] >= THRESHOLDS["supervised_f1"]},
        {"name": "retrieval_hit_rate",
         "value": ret["retrieval_eval"]["hit_rate_at_k"],
         "threshold": THRESHOLDS["retrieval_hit_rate"],
         "passed": (ret["retrieval_eval"]["hit_rate_at_k"] or 0) >= THRESHOLDS["retrieval_hit_rate"]},
        {"name": "policy_citation_accuracy",
         "value": pol["citation_accuracy"],
         "threshold": THRESHOLDS["policy_citation_accuracy"],
         "passed": pol["citation_accuracy"] >= THRESHOLDS["policy_citation_accuracy"]},
        {"name": "agent_routing_accuracy",
         "value": round(routing_acc, 4),
         "threshold": THRESHOLDS["agent_routing_accuracy"],
         "passed": routing_acc >= THRESHOLDS["agent_routing_accuracy"]},
        {"name": "policy_grounding_rate",
         "value": round(grounding_rate, 4),
         "threshold": 1.0,
         "passed": grounding_rate >= 1.0},
    ]

    summary = {
        "ml": {
            "best_model": ml["supervised"]["best_model"],
            "best_roc_auc": ml["supervised"]["best_roc_auc"],
            "metrics": ml["supervised"]["models"][ml["supervised"]["best_model"]],
            "secondary_returned_flag": ml["secondary"],
        },
        "segmentation": {
            "chosen_k": seg["chosen_k"],
            "silhouette": seg["chosen_silhouette"],
            "descriptions": seg["segment_descriptions"],
        },
        "retrieval": ret["retrieval_eval"],
        "policy_qa": {"citation_accuracy": pol["citation_accuracy"]},
        "agent": {"routing_accuracy": round(routing_acc, 4), "details": routing_details},
        "grounding": {"rate": round(grounding_rate, 4), "details": grounding_details},
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "n_example_queries": len(example_queries),
    }

    with open(EVALUATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


if __name__ == "__main__":  # pragma: no cover
    s = run()
    print("ALL CHECKS PASSED:", s["all_passed"])
    for c in s["checks"]:
        flag = "PASS" if c["passed"] else "FAIL"
        print(f"  [{flag}] {c['name']}: {c['value']} (>= {c['threshold']})")
