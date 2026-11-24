"""
policy_qa.py
============
Phase 4 (part 2) - Policy-Aware Question Answering.

A small, fully grounded QA path over the retail policy documents:

1. Embed the policy documents and index them (same embedder as products).
2. Retrieve the most relevant policy evidence for a question.
3. Answer **only** from the retrieved evidence, quoting the exact sentence.
4. Always return the source ``doc_id`` (citation) and a confidence signal.
5. Refuse gracefully when retrieval is weak (no unsupported answers).

Run directly:  ``python src/policy_qa.py``
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

import numpy as np

from config import (
    MIN_RETRIEVAL_SCORE,
    POLICY_DOCS_JSONL,
    POLICY_QA_JSON,
    TOP_K,
    ensure_dirs,
)
from llm_client import get_llm_client


class PolicyIndex:
    def __init__(self, docs: List[Dict[str, object]], client=None):
        self.docs = docs
        self.client = client or get_llm_client()
        self.texts = [d["text"] for d in docs]
        self.client.fit_embeddings(self.texts)
        self.embeddings = self.client.embed(self.texts).astype(np.float32)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, object]]:
        q = self.client.embed([query]).astype(np.float32)
        sims = (self.embeddings @ q.T).ravel()
        order = np.argsort(sims)[::-1][:top_k]
        return [
            {**self.docs[int(i)], "score": float(sims[int(i)])}
            for i in order
        ]


def _best_sentence(question: str, text: str) -> str:
    """Pick the sentence in the policy text most overlapping with the question."""
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    best, best_overlap = sentences[0], -1
    for s in sentences:
        s_tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best, best_overlap = s, overlap
    return best.strip()


def answer_policy_question(index: PolicyIndex, question: str, top_k: int = 3) -> Dict[str, object]:
    hits = index.search(question, top_k=top_k)
    if not hits or hits[0]["score"] < MIN_RETRIEVAL_SCORE:
        return {
            "question": question,
            "status": "insufficient_evidence",
            "answer": "I do not have enough policy evidence to answer that confidently. "
                      "Please contact customer support.",
            "citations": [],
            "evidence": [],
        }

    top = hits[0]
    evidence_sentence = _best_sentence(question, top["text"])

    context = "\n\n".join(f"[{h['doc_id']}] {h['text']}" for h in hits)
    prompt = (
        f"Answer the customer question using ONLY the policy context. "
        f"Quote the exact policy sentence and cite the doc id.\nQuestion: {question}"
    )
    generated = index.client.answer(prompt, context)

    return {
        "question": question,
        "status": "answered",
        "answer": evidence_sentence,
        "generated_answer": generated,
        "citations": [h["doc_id"] for h in hits],
        "primary_source": top["doc_id"],
        "primary_source_title": top["title"],
        "confidence": round(top["score"], 4),
        "evidence": [
            {"doc_id": h["doc_id"], "title": h["title"],
             "score": round(h["score"], 4), "text": h["text"]}
            for h in hits
        ],
    }


def load_policy_docs() -> List[Dict[str, object]]:
    docs = []
    with open(POLICY_DOCS_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            docs.append(json.loads(line))
    return docs


def build_index() -> PolicyIndex:
    ensure_dirs()
    return PolicyIndex(load_policy_docs())


# --------------------------------------------------------------------------- #
# Curated policy test questions (used by the Final Phase evaluation)
# --------------------------------------------------------------------------- #
POLICY_TEST_QUESTIONS = [
    {"question": "Can I return a discounted product after 7 days?",
     "expected_doc_id": "POL_RETURNS"},
    {"question": "How long does standard delivery take?",
     "expected_doc_id": "POL_SHIPPING"},
    {"question": "When will I get my refund for a prepaid order?",
     "expected_doc_id": "POL_REFUND"},
    {"question": "Can I exchange a product for a different size?",
     "expected_doc_id": "POL_EXCHANGE"},
    {"question": "Is physical damage covered under warranty?",
     "expected_doc_id": "POL_WARRANTY"},
]


def run() -> Dict[str, object]:
    ensure_dirs()
    index = build_index()
    results = [answer_policy_question(index, q["question"]) for q in POLICY_TEST_QUESTIONS]

    # citation accuracy: did the top source match the expected policy?
    correct = sum(
        1 for q, r in zip(POLICY_TEST_QUESTIONS, results)
        if r.get("primary_source") == q["expected_doc_id"]
    )
    summary = {
        "n_questions": len(results),
        "citation_accuracy": round(correct / len(results), 4),
        "results": results,
    }
    with open(POLICY_QA_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":  # pragma: no cover
    out = run()
    print("Citation accuracy:", out["citation_accuracy"])
    for r in out["results"]:
        print(f"\nQ: {r['question']}\n  A: {r['answer']}\n  src: {r.get('primary_source')}")
