"""
build_notebooks.py
==================
Programmatically builds the five capstone notebooks with nbformat.

Each notebook is a self-contained walkthrough of one phase: markdown cells
explain the reasoning, code cells import the corresponding ``src`` module and
display the artefacts it produced.  Heavy computation is done once by
``src/run_all.py``; the notebooks read the resulting CSV/JSON so they execute
in seconds and stay reproducible.

Run:  python build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

# Common preamble injected into every notebook so `src` is importable and the
# working directory is the project root regardless of where Jupyter is launched.
PREAMBLE = """\
import os, sys, json
from pathlib import Path

# Locate the project root (the folder that contains `src/`) and make it the CWD
# so every relative path in the notebooks and in `src` resolves correctly.
_here = Path.cwd()
_root = _here
for _p in [_here, *_here.parents]:
    if (_p / "src" / "config.py").exists():
        _root = _p
        break
os.chdir(_root)
sys.path.insert(0, str(_root / "src"))

import pandas as pd
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
print("Project root:", _root)
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write(nb, name: str):
    path = NB_DIR / name
    nbf.write(nb, str(path))
    print("wrote", path)


# --------------------------------------------------------------------------- #
# 01 - Data preparation
# --------------------------------------------------------------------------- #
def nb01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 01 - Data Preparation (Phase 1)
### Intelligent Retail Decision Assistant

**Business goal.** Define the retail use case and prepare a well-documented base
dataset for the downstream ML and GenAI workflows.

**What this notebook does**
1. Loads the raw retail interaction dataset and inspects schema, missingness,
   target columns and class balance.
2. Applies explicit, documented cleaning (missing-value handling, dedup).
3. Engineers business-meaningful features.
4. Produces a clean data dictionary that records **which columns are original
   vs engineered**.
"""),
        code(PREAMBLE),
        md("## 1.1 Load the raw dataset and inspect it"),
        code("""
from data_prep import load_raw, inspect, load_policies

raw = load_raw()
print("Shape:", raw.shape)
raw.head(3)
"""),
        code("""
rep = inspect(raw)
print("Rows:", rep["n_rows"], "| Cols:", rep["n_cols"])
print("Duplicate rows:", rep["n_duplicates"])
print("\\nMissing values:")
print(rep["missing"])
print("\\nTarget balance:")
for k, v in rep["target_balance"].items():
    print(f"  {k}: {v}")
"""),
        md("""
**Observation.** The only column with missing values is `size` (270 rows) -
these are non-sized items such as accessories and watches. `purchase_flag` is
close to balanced (~51/49), while `returned_flag` is imbalanced (~18% positive),
which we handle explicitly in Phase 2.

> **Data-integrity note.** `product_id` is **not unique** in the source data:
> 300 ids repeat across the 1,500 rows, and the five rows sharing an id carry
> *different* product attributes. `product_id` therefore cannot identify a
> product, so we derive a stable identity, `product_key`, from the full
> attribute signature. All downstream product-level work (retrieval documents,
> retrieval evaluation) keys on `product_key`.
"""),
        md("## 1.2 Clean the data"),
        code("""
from data_prep import clean, engineer_features

clean_df = clean(raw)
print("After cleaning:", clean_df.shape)
print("Missing after cleaning:", int(clean_df.isna().sum().sum()))
"""),
        md("""
**Cleaning decisions**
* `size` missing -> filled with the explicit token `"Not Applicable"` and a
  `size_missing` indicator is kept (we do **not** invent a size).
* Any other missing numeric -> median imputed; categorical -> `"Unknown"`.
* Duplicate rows dropped; non-positive prices removed.
"""),
        md("## 1.3 Feature engineering"),
        code("""
feat = engineer_features(clean_df)
new_cols = [c for c in feat.columns if c not in raw.columns]
print("Engineered columns:", new_cols)
feat[["customer_id", "final_price", "price_band", "discount_band",
      "engagement_score", "recency_band", "customer_value"]].head()
"""),
        md("## 1.4 Data dictionary (original vs engineered)"),
        code("""
from data_prep import build_data_dictionary

dd = build_data_dictionary(feat)
print("Original columns :", (dd.provenance == "original").sum())
print("Engineered columns:", (dd.provenance == "engineered").sum())
dd[["column", "dtype", "provenance", "n_missing", "description"]]
"""),
        md("## 1.5 Policy documents"),
        code("""
pol = load_policies()
for k, v in pol.items():
    print(f"[{k}] {v}\\n")
"""),
        md("""
**Phase 1 summary.** We have a clean, documented dataset of 1,500
customer-product interactions with 46 original columns plus engineered
features (including the `product_key` identity), a machine-readable data
dictionary, and five policy documents ready for the GenAI phases.
"""),
    ]
    write(nb, "01_data_preparation.ipynb")


# --------------------------------------------------------------------------- #
# 02 - Classical ML
# --------------------------------------------------------------------------- #
def nb02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 02 - Classical ML Pipeline (Phase 2)

**Business goal.** Use structured retail data to generate predictive insights
and segmentation for business decision-making.

**Supervised task.** Predict `purchase_flag` (conversion). Justification: it is
the core conversion KPI, directly actionable, and balanced enough to model
reliably.

**Unsupervised task.** K-Means customer segmentation on behavioural + value
features.
"""),
        code(PREAMBLE),
        md("## 2.1 Load the feature table"),
        code("""
from classical_ml import load_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

df = load_features()
print("Shape:", df.shape)
print("Numeric features   :", len(NUMERIC_FEATURES))
print("Categorical features:", len(CATEGORICAL_FEATURES))
df[["purchase_flag", "returned_flag"]].apply(pd.Series.value_counts)
"""),
        md("""
## 2.2 Preprocessing pipeline

* **Numerical** -> median imputation + `StandardScaler`.
* **Categorical** -> most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`.
* Wrapped in a `ColumnTransformer` so the exact same transform is applied at
  train and inference time (no leakage).
"""),
        code("""
from classical_ml import build_preprocessor
pre = build_preprocessor()
print(pre)
"""),
        md("## 2.3 Train and evaluate supervised baselines"),
        code("""
from classical_ml import train_supervised
sup = train_supervised(df)
print("Best model:", sup["best_model"], "| ROC-AUC:", sup["best_roc_auc"])
print("Imbalance handling:", sup["imbalance_handling"])
"""),
        code("""
import pandas as pd
rows = []
for name, m in sup["models"].items():
    rows.append({"model": name, **{k: m[k] for k in
        ["accuracy","precision","recall","f1","roc_auc","pr_auc","cv_roc_auc_mean"]}})
pd.DataFrame(rows).set_index("model").round(4)
"""),
        code("""
print(sup["models"][sup["best_model"]]["classification_report"])
"""),
        md("## 2.4 Feature importance and interpretation"),
        code("""
pd.DataFrame(sup["top_features"]).head(15)
"""),
        md("""
**Interpretation.** Engagement signals (`time_on_page`, `added_to_cart`,
`wishlist_flag`), price context (`final_price`, `discount_percent`) and
customer value (`avg_order_value`, `customer_value`) dominate. In business
terms: *how engaged a shopper is on the page and how the price is positioned
matter more than raw demographics* - which supports personalising the product
page and the discount strategy.
"""),
        md("## 2.5 Secondary (imbalanced) target: returned_flag"),
        code("""
from classical_ml import train_secondary
sec = train_secondary(df)
print(json.dumps(sec, indent=2))
"""),
        md("""
**Note.** `returned_flag` is heavily imbalanced (~18% positive). We use
`class_weight="balanced_subsample"` and report **PR-AUC** as the honest metric
rather than accuracy, which would be misleadingly high.
"""),
        md("## 2.6 Unsupervised: customer segmentation"),
        code("""
from classical_ml import train_segmentation
seg = train_segmentation(df)
print("Chosen k:", seg["chosen_k"], "| silhouette:", seg["chosen_silhouette"])
pd.DataFrame(seg["diagnostics"])
"""),
        code("""
pd.DataFrame(seg["segment_profile"])
"""),
        code("""
for s, d in seg["segment_descriptions"].items():
    print(f"Segment {s}: {d}\\n")
"""),
        md("""
**Segment meaning (business language).** The clusters separate customers along
value, recency and price behaviour - e.g. *high-value loyalists*, *active
regulars*, *discount-driven bargain hunters* and *dormant / at-risk* customers.
These map directly to retention, win-back and discount-targeting campaigns.
"""),
        md("## 2.7 Figures"),
        code("""
from IPython.display import Image, display
from config import FIGURES_DIR
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f.name)
    display(Image(filename=str(f)))
"""),
    ]
    write(nb, "02_classical_ml.ipynb")


# --------------------------------------------------------------------------- #
# 03 - RAG recommendation
# --------------------------------------------------------------------------- #
def nb03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 03 - RAG and Grounded Product Recommendation (Phase 3 + 4a)

**Business goal.** Enable grounded product recommendation using retrieved
context.

**Pipeline.** Product records -> retrieval-ready text documents -> embeddings
-> FAISS index -> top-k retrieval -> grounded answer with supporting product ids.
"""),
        code(PREAMBLE),
        md("## 3.1 Retrieval-ready product documents (Phase 3)"),
        code("""
from rag_recommendation import load_product_docs
docs = load_product_docs()
print("Number of product documents:", len(docs))
print("\\nExample document:\\n")
print(docs[0]["text"])
"""),
        md("## 3.2 Build the embedding index"),
        code("""
from rag_recommendation import build_index
index = build_index()
print("Backend:", index.backend)
print("Documents indexed:", len(index.docs))
print("Embedding shape:", index.embeddings.shape)
"""),
        md("""
**Embedding + re-ranking choice.** By default we use a deterministic, fit-free
hashing bag-of-words embedding (L2-normalised, so dot product = cosine
similarity). If `OPENAI_API_KEY` is set, the same interface transparently uses
OpenAI embeddings. The index is FAISS when available, with a NumPy cosine
fallback.

Similarity alone cannot distinguish products of the same category with
different attributes (a *blue* jacket and an *olive* jacket score almost
identically), so retrieval is **hybrid**: dense candidate generation followed by
a re-rank that blends dense similarity, a lexical (IDF-weighted) overlap term,
and an **attribute-match bonus** for colour / sub-category / use-case / brand
words named in the query - the standard e-commerce re-ranking pattern.
"""),
        md("## 3.3 Query parsing (explicit, inspectable filters)"),
        code("""
from rag_recommendation import parse_query
for q in ["casual sneakers under Rs.4000 for daily wear",
          "black formal leather shoes for office",
          "premium watch with warranty"]:
    print(q, "->", parse_query(q))
"""),
        md("## 3.4 Grounded recommendation"),
        code("""
from rag_recommendation import recommend
res = recommend(index, "casual sneakers under Rs.4000 for daily wear")
print("Status:", res["status"])
print("Filters:", res["filters"])
print("\\nAnswer:\\n", res["answer"])
print("\\nSupporting doc ids:", res["supporting_doc_ids"])
pd.DataFrame(res["recommended_products"])
"""),
        code("""
for q in ["black formal leather shoes for office",
          "blue backpack for travel",
          "premium watch with warranty"]:
    r = recommend(index, q)
    print(f"\\nQ: {q}\\n  status={r['status']}  docs={r['supporting_doc_ids']}")
    for p in r["recommended_products"][:3]:
        print(f"    - {p['title']} | Rs.{p['price']:.0f} | {p['reason']}")
"""),
        md("## 3.5 Retrieval quality evaluation"),
        code("""
from rag_recommendation import evaluate_retrieval
from enrich_genai import build_synthetic_queries
qs = build_synthetic_queries(pd.read_csv("data/processed/ml_features.csv"))
ev = evaluate_retrieval(index, qs)
print("Evaluated queries:", ev["n_evaluated"])
print("Hit-rate@k:", ev["hit_rate_at_k"])
print("MRR:", ev["mrr"])
"""),
        md("""
**Grounding guarantee.** The answer is composed only from the retrieved
documents, and every recommendation carries its `doc_id` and `product_id`, so
the response is fully traceable. Weak retrieval is reported explicitly
(`status="weak_retrieval"`) rather than answered speculatively.
"""),
    ]
    write(nb, "03_rag_recommendation.ipynb")


# --------------------------------------------------------------------------- #
# 04 - Policy QA
# --------------------------------------------------------------------------- #
def nb04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 04 - Policy-Aware Question Answering (Phase 4b)

**Business goal.** Answer retail policy questions using **retrieved evidence
only** - never unsupported free-form answers.

**Guarantees**
* Every answer quotes the exact policy sentence.
* Every answer returns the source `doc_id` (citation).
* Weak retrieval -> graceful refusal, not a guess.
"""),
        code(PREAMBLE),
        md("## 4.1 Policy documents"),
        code("""
from policy_qa import load_policy_docs
pol = load_policy_docs()
for d in pol:
    print(f"[{d['doc_id']}] {d['title']}")
    print("   ", d["text"], "\\n")
"""),
        md("## 4.2 Build the policy index"),
        code("""
from policy_qa import build_index
pindex = build_index()
print("Policy docs indexed:", len(pindex.docs))
"""),
        md("## 4.3 Grounded answers with citations"),
        code("""
from policy_qa import answer_policy_question
r = answer_policy_question(pindex, "Can I return a discounted product after 7 days?")
print("Status:", r["status"])
print("Answer:", r["answer"])
print("Primary source:", r["primary_source"], "-", r["primary_source_title"])
print("Confidence:", r["confidence"])
print("Citations:", r["citations"])
"""),
        code("""
for q in ["How long does standard delivery take?",
          "When will I get my refund for a prepaid order?",
          "Can I exchange a product for a different size?",
          "Is physical damage covered under warranty?"]:
    r = answer_policy_question(pindex, q)
    print(f"\\nQ: {q}")
    print(f"  A: {r['answer']}")
    print(f"  src: {r.get('primary_source')} (conf {r.get('confidence')})")
"""),
        md("## 4.4 Graceful handling of out-of-scope questions"),
        code("""
r = answer_policy_question(pindex, "Do you offer a loyalty points programme for pets?")
print("Status:", r["status"])
print("Answer:", r["answer"])
"""),
        md("""
**Why this matters.** The refusal path is what makes the assistant safe: it
never fabricates a policy. This is the behaviour the capstone guardrail
"do not allow unsupported policy answers" demands.
"""),
        md("## 4.5 Citation accuracy on the curated test set"),
        code("""
from policy_qa import POLICY_TEST_QUESTIONS
correct = 0
for q in POLICY_TEST_QUESTIONS:
    r = answer_policy_question(pindex, q["question"])
    ok = r.get("primary_source") == q["expected_doc_id"]
    correct += ok
    print(f"[{'OK' if ok else 'XX'}] {q['question']} -> {r.get('primary_source')} (expected {q['expected_doc_id']})")
print(f"\\nCitation accuracy: {correct}/{len(POLICY_TEST_QUESTIONS)} = {correct/len(POLICY_TEST_QUESTIONS):.0%}")
"""),
    ]
    write(nb, "04_policy_qa.ipynb")


# --------------------------------------------------------------------------- #
# 05 - Agentic assistant
# --------------------------------------------------------------------------- #
def nb05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 05 - Agentic Assistant / Controller (Phase 5)

**Business goal.** Build a simple but clear routing layer that chooses the
right tool for the user's request, with an **inspectable trace**.

**Tools**
| Tool | Purpose |
|------|---------|
| `product_search` | RAG product recommendation |
| `policy_qa` | grounded policy question answering |
| `analytics` | customer-segment / price-sensitivity analytics |
| `review_summary` | review sentiment summarisation |
| `calculator` | safe arithmetic for price/discount questions |

Routing is a hybrid rule-based + keyword-scoring controller - explicit and
bounded to a single tool per query (no unbounded loops).
"""),
        code(PREAMBLE),
        md("## 5.1 Intent detection"),
        code("""
from agent_controller import detect_intent
for q in ["Recommend casual sneakers under Rs.4000 for daily wear.",
          "Can I return a discounted product after 7 days?",
          "Which customer segment seems most price-sensitive?",
          "Calculate the price of 2500 with 20% discount."]:
    print(q, "->", detect_intent(q))
"""),
        md("## 5.2 Build the controller"),
        code("""
from agent_controller import AgentController
agent = AgentController()
print("Registered tools:", list(agent.registry.keys()))
"""),
        md("## 5.3 Route the capstone's example queries (with trace)"),
        code("""
from agent_controller import DEMO_QUERIES
for q in DEMO_QUERIES:
    out = agent.route(q)
    tr = out["trace"]
    print(f"\\nQ: {tr['query']}")
    print(f"   intent={tr['intent']}  tool={tr['tool']}  status={tr['status']}")
    print(f"   reason: {tr['reason']}")
    print(f"   result: {tr['result_summary']}")
"""),
        md("## 5.4 Full result for a product query"),
        code("""
out = agent.route("Recommend casual sneakers under Rs.4000 for daily wear.")
print("Answer:", out["result"]["answer"])
pd.DataFrame(out["result"]["recommended_products"])
"""),
        md("## 5.5 Full result for a policy query"),
        code("""
out = agent.route("Can I return a discounted product after 7 days?")
print("Answer:", out["result"]["answer"])
print("Source:", out["result"]["primary_source"])
"""),
        md("## 5.6 Full result for an analytics query"),
        code("""
out = agent.route("Which customer segment seems most price-sensitive?")
print("Answer:", out["result"]["answer"])
pd.DataFrame(out["result"]["table"])
"""),
        md("## 5.7 Routing distribution across the demo set"),
        code("""
from collections import Counter
dist = Counter(agent.route(q)["trace"]["intent"] for q in DEMO_QUERIES)
print(dict(dist))
"""),
        md("""
**Design note.** The controller is deliberately simple: keyword scoring picks
one intent, one tool runs, and a trace records the decision. This satisfies the
capstone's "keep routing bounded and simple" and "return a trace showing what
the system decided to do" requirements without an opaque framework.
"""),
    ]
    write(nb, "05_agentic_assistant.ipynb")


if __name__ == "__main__":
    nb01(); nb02(); nb03(); nb04(); nb05()
    print("\nAll notebooks built.")
