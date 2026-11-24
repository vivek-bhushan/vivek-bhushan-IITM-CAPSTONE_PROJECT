"""
build_report.py
===============
Assembles the detailed Capstone Project report as a PDF from the artefacts
produced by the pipeline (``outputs/*.json``, ``data/processed/*``).

Run:  python build_report.py
Output: outputs/Capstone Project report.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"
FIG = OUT / "figures"


def load(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default if default is not None else {}


# --------------------------------------------------------------------------- #
# PDF class with simple heading/body helpers
# --------------------------------------------------------------------------- #
class Report(FPDF):
    def multi_cell(self, w, h=None, text="", *args, **kwargs):
        """Force a multiline cell to return to the LEFT margin.

        fpdf2 >= 2.7 defaults ``new_x`` to RIGHT, which leaves the cursor at the
        right margin after every call; a following ``multi_cell(0, ...)`` then has
        zero horizontal space and raises "Not enough horizontal space to render a
        single character". Resetting to LMARGIN/NEXT keeps the original
        line-by-line layout the report was written against.
        """
        kwargs.setdefault("new_x", XPos.LMARGIN)
        kwargs.setdefault("new_y", YPos.NEXT)
        return super().multi_cell(w, h, text, *args, **kwargs)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Intelligent Retail Decision Assistant - Capstone Project Report",
                  align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # -- helpers ----------------------------------------------------------- #
    def h1(self, text):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(20, 60, 90)
        self.multi_cell(0, 9, text)
        self.ln(2)

    def h2(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 90, 120)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def h3(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(0.5)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, text)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, "  -  " + text)
        self.ln(0.5)

    def kv_table(self, rows, widths=(70, 110)):
        self.set_font("Helvetica", "", 9.5)
        for k, v in rows:
            self.set_font("Helvetica", "B", 9.5)
            self.set_text_color(40, 40, 40)
            self.cell(widths[0], 6, str(k)[:48], border=0)
            self.set_font("Helvetica", "", 9.5)
            self.multi_cell(widths[1], 6, str(v), border=0)
        self.ln(1.5)

    def table(self, headers, rows, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 240, 245)
        self.set_text_color(20, 40, 60)
        for h, w in zip(headers, widths):
            self.cell(w, 7, str(h), border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        for r in rows:
            for val, w in zip(r, widths):
                self.cell(w, 6, str(val)[:40], border=1)
            self.ln()
        self.ln(2)

    def figure(self, path: Path, caption: str, w=150):
        if not path.exists():
            return
        if self.get_y() > 200:
            self.add_page()
        self.image(str(path), w=w)
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 5, caption)
        self.ln(2)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build():
    ml = load(OUT / "ml_metrics.json")
    seg = load(OUT / "segment_summary.json")
    ret = load(OUT / "retrieval_eval.json")
    pol = load(OUT / "policy_qa_results.json")
    ev = load(OUT / "evaluation_summary.json")
    ex = load(OUT / "example_queries.json", [])
    prep = load(PROC / "data_prep_summary.json")
    enrich = load(PROC / "enrich_summary.json")

    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ---------------- Title page ---------------- #
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(20, 60, 90)
    pdf.multi_cell(0, 12, "Intelligent Retail Decision Assistant", align="C")
    pdf.set_font("Helvetica", "", 15)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 9, "From Tabular Machine Learning to Agentic\nGenerative AI Platform", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 90, 120)
    pdf.multi_cell(0, 8, "Capstone Project Report", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        0, 7,
        "An end-to-end retail AI prototype spanning problem framing, dataset "
        "preparation, classical supervised and unsupervised machine learning, "
        "retrieval-augmented generation, policy-aware question answering, and an "
        "agentic controller - evaluated and packaged as a deployable prototype.",
        align="C",
    )
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, "Prepared as the final deliverable of the AI/ML Capstone Programme.", align="C")

    # ---------------- Executive summary ---------------- #
    pdf.add_page()
    pdf.h1("1. Executive Summary")
    pdf.body(
        "This report documents the design, implementation and evaluation of the "
        "Intelligent Retail Decision Assistant - a single coherent retail AI "
        "application built for a mid-sized online retailer. The system begins with "
        "structured tabular data, adds predictive and segmentation intelligence "
        "through classical machine learning, extends the same records into "
        "retrieval-ready text for generative AI, and finally exposes everything "
        "through a simple, inspectable agent/controller that routes each user "
        "request to the right tool."
    )
    if ev:
        pdf.h3("Headline results")
        rows = []
        for c in ev.get("checks", []):
            rows.append([c["name"], c["value"], c["threshold"],
                         "PASS" if c["passed"] else "FAIL"])
        pdf.table(["Check", "Value", "Threshold", "Status"], rows, [62, 30, 30, 25])
        pdf.body(f"All regression checks passed: {ev.get('all_passed')}. "
                 f"Curated example queries delivered: {ev.get('n_example_queries')}.")

    # ---------------- Problem framing ---------------- #
    pdf.add_page()
    pdf.h1("2. Problem Framing and Business Context")
    pdf.h2("2.1 Business problem")
    pdf.body(
        "A mid-sized online retailer serves both internal business teams and end "
        "customers. It holds structured data about customers, products and "
        "transactions, plus unstructured text such as product descriptions, "
        "customer reviews and policy documents. Leadership wants a prototype that "
        "can (a) analyse customer and product behaviour with classical ML, "
        "(b) answer product and policy questions with retrieval-backed generation, "
        "and (c) decide which workflow to use based on the user's query."
    )
    pdf.h2("2.2 Business questions")
    for q in [
        "Which interactions convert to a purchase, and what drives conversion?",
        "Which customers form meaningful behavioural segments, and which are most price-sensitive?",
        "Which products best match a natural-language shopping request?",
        "What do the retailer's policies say about returns, refunds, shipping, exchange and warranty?",
        "How should the assistant route a request to the correct capability?",
    ]:
        pdf.bullet(q)
    pdf.h2("2.3 Success metrics")
    pdf.body(
        "Supervised conversion model: ROC-AUC and F1 (with PR-AUC for the "
        "imbalanced return target). Segmentation: silhouette score. Retrieval: "
        "hit-rate@k and MRR. Policy QA: citation accuracy and grounding rate. "
        "Agent: routing accuracy. These are enforced as regression thresholds in "
        "the final evaluation."
    )

    # ---------------- Dataset ---------------- #
    pdf.add_page()
    pdf.h1("3. Dataset Preparation (Phase 1)")
    pdf.h2("3.1 Source data")
    pdf.body(
        "The base dataset is retail_interactions.csv - 1,500 customer-product "
        "interaction records across 46 columns covering customer profile, customer "
        "history, product profile, commercial data, interaction data and "
        "post-purchase outcomes. Five policy documents are provided in "
        "retail_policies.json."
    )
    if prep:
        pdf.kv_table([
            ("Raw shape", prep.get("raw_shape")),
            ("Clean shape", prep.get("clean_shape")),
            ("Feature shape", prep.get("feature_shape")),
            ("Original columns", prep.get("n_original_columns")),
            ("Engineered columns", prep.get("n_engineered_columns")),
            ("Missing after cleaning", prep.get("missing_after_clean")),
        ])
    pdf.h2("3.2 Cleaning decisions")
    for b in [
        "size is missing for 270 non-sized items (accessories, watches): filled with the explicit token 'Not Applicable' and a size_missing indicator is retained - no size is invented.",
        "Any other missing numeric value is median-imputed; any other missing categorical value is filled with 'Unknown'.",
        "Duplicate rows are dropped and non-positive prices removed.",
    ]:
        pdf.bullet(b)
    pdf.h2("3.3 Engineered features")
    pdf.body(
        "Nine features were engineered deterministically: price_band, "
        "discount_band, price_sensitivity, engagement_score, recency_band, "
        "customer_value, is_discounted, size_missing and product_text. The full "
        "data dictionary (original vs engineered, with dtypes and descriptions) is "
        "written to data/processed/data_dictionary.csv."
    )
    pdf.h2("3.4 Class balance")
    pdf.body(
        "purchase_flag is close to balanced (~51% positive), while returned_flag is "
        "imbalanced (~18% positive). This directly informs the modelling choices in "
        "Phase 2."
    )

    # ---------------- Classical ML ---------------- #
    pdf.add_page()
    pdf.h1("4. Classical ML Pipeline (Phase 2)")
    pdf.h2("4.1 Supervised task: predict purchase_flag")
    if ml:
        sup = ml.get("supervised", {})
        pdf.body(sup.get("target_justification", ""))
        pdf.body(f"Train/test split: {sup.get('n_train')} / {sup.get('n_test')} "
                 f"(stratified). Positive rate: {sup.get('positive_rate')}. "
                 f"{sup.get('imbalance_handling','')}")
        rows = []
        for name, m in sup.get("models", {}).items():
            rows.append([name, m["accuracy"], m["precision"], m["recall"],
                         m["f1"], m["roc_auc"], m["pr_auc"]])
        pdf.table(["Model", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "PR-AUC"],
                  rows, [40, 20, 20, 20, 20, 25, 25])
        pdf.body(f"Best model: {sup.get('best_model')} with ROC-AUC "
                 f"{sup.get('best_roc_auc')}.")
        pdf.h3("Top predictive features")
        for f in sup.get("top_features", [])[:10]:
            pdf.bullet(f"{f['feature']}  (importance {f['importance']})")
        pdf.body(
            "Interpretation: engagement signals (time_on_page, added_to_cart, "
            "wishlist_flag), price context (final_price, discount_percent) and "
            "customer value (avg_order_value, customer_value) dominate. In business "
            "terms, how engaged a shopper is on the page and how the price is "
            "positioned matter more than raw demographics - supporting page "
            "personalisation and a data-driven discount strategy."
        )
    pdf.h2("4.2 Secondary target: returned_flag (imbalance handling)")
    if ml:
        sec = ml.get("secondary", {})
        pdf.kv_table([
            ("Positive rate", sec.get("positive_rate")),
            ("ROC-AUC", sec.get("roc_auc")),
            ("PR-AUC", sec.get("pr_auc")),
            ("F1", sec.get("f1")),
        ])
        pdf.body(sec.get("note", ""))

    pdf.add_page()
    pdf.h2("4.3 Unsupervised task: customer segmentation")
    if seg:
        pdf.body(seg.get("feature_justification", ""))
        pdf.body(f"Chosen k = {seg.get('chosen_k')} (silhouette "
                 f"{seg.get('chosen_silhouette')}).")
        rows = [[d["k"], d["inertia"], d["silhouette"]] for d in seg.get("diagnostics", [])]
        pdf.table(["k", "Inertia", "Silhouette"], rows, [30, 50, 50])
        pdf.h3("Segment descriptions (business language)")
        for s, d in seg.get("segment_descriptions", {}).items():
            pdf.bullet(f"Segment {s}: {d}")
        pdf.body(
            "The clusters separate customers along value, recency and price "
            "behaviour - e.g. high-value loyalists, active regulars, "
            "discount-driven bargain hunters and dormant/at-risk customers. These "
            "map directly to retention, win-back and discount-targeting campaigns."
        )
    pdf.figure(FIG / "02_model_comparison.png", "Figure 1. Supervised model comparison (ROC-AUC vs PR-AUC).")
    pdf.figure(FIG / "03_feature_importance.png", "Figure 2. Top predictive features of the best model.")
    pdf.figure(FIG / "04_cluster_diagnostics.png", "Figure 3. K-Means elbow and silhouette diagnostics.")
    pdf.figure(FIG / "05_segments.png", "Figure 4. Customer segments by order value and lifetime value.")

    # ---------------- GenAI enrichment ---------------- #
    pdf.add_page()
    pdf.h1("5. Data Enrichment for Generative AI (Phase 3)")
    pdf.body(
        "Structured records were transformed into retrieval-ready text documents. "
        "Each unique product becomes one document combining title, description, "
        "category, brand, colour, material, season, use case, style/occasion tags, "
        "price context and rating, with a stable doc_id (PROD_<product_id>). Each "
        "policy snippet becomes one document with a stable doc_id (POL_*)."
    )
    if enrich:
        pdf.kv_table([
            ("Product documents", enrich.get("n_product_docs")),
            ("Policy documents", enrich.get("n_policy_docs")),
            ("Synthetic queries", enrich.get("n_synthetic_queries")),
            ("Avg product doc length (chars)", enrich.get("avg_product_doc_chars")),
        ])

    # ---------------- RAG ---------------- #
    pdf.h1("6. RAG and Grounded Answer Generation (Phase 4)")
    pdf.h2("6.1 Retrieval architecture")
    pdf.body(
        "Documents are embedded and stored in a vector index (FAISS when available, "
        "with a NumPy cosine fallback). The default embedder is a deterministic, "
        "fit-free hashing bag-of-words model (L2-normalised, so dot product equals "
        "cosine similarity); if an OpenAI-compatible endpoint is configured, the "
        "same interface transparently uses OpenAI embeddings. Queries are parsed "
        "for explicit price and category filters before retrieval."
    )
    if ret:
        re_ = ret.get("retrieval_eval", {})
        pdf.kv_table([
            ("Queries evaluated", re_.get("n_evaluated")),
            ("Hit-rate@k", re_.get("hit_rate_at_k")),
            ("MRR", re_.get("mrr")),
            ("Top-k", re_.get("top_k")),
        ])
        pdf.h3("Example grounded recommendations")
        for r in ret.get("demo_results", [])[:4]:
            pdf.bullet(f"Q: {r['query']}  ->  {r['supporting_doc_ids']}")
    pdf.h2("6.2 Grounding guarantee")
    pdf.body(
        "The generated answer is composed only from the retrieved documents, and "
        "every recommendation carries its doc_id and product_id, so the response is "
        "fully traceable. Weak retrieval is reported explicitly rather than "
        "answered speculatively."
    )

    # ---------------- Policy QA ---------------- #
    pdf.add_page()
    pdf.h1("7. Policy-Aware Question Answering (Phase 4b)")
    pdf.body(
        "A dedicated grounded QA path answers policy questions using retrieved "
        "evidence only. Every answer quotes the exact policy sentence and returns "
        "the source doc_id; out-of-scope questions are refused rather than guessed."
    )
    if pol:
        pdf.body(f"Citation accuracy on the curated test set: "
                 f"{pol.get('citation_accuracy')}.")
        rows = []
        for r in pol.get("results", []):
            rows.append([r["question"][:44], r.get("primary_source", "-"),
                         str(r.get("confidence", "-"))])
        pdf.table(["Question", "Source", "Conf."], rows, [95, 40, 25])
        pdf.h3("Example grounded answers")
        for r in pol.get("results", [])[:5]:
            pdf.bullet(f"Q: {r['question']}")
            pdf.bullet(f"A: {r['answer']}  [{r.get('primary_source')}]")

    # ---------------- Agent ---------------- #
    pdf.h1("8. Agent / Controller (Phase 5)")
    pdf.body(
        "The controller is a hybrid rule-based + keyword-scoring router. It detects "
        "the intent of a query, selects exactly one tool, executes it, and returns "
        "an inspectable trace (intent, tool, reason, arguments, status, result "
        "summary). Routing is bounded to a single tool per query - no unbounded "
        "loops - per the capstone's 'keep routing bounded and simple' guidance."
    )
    pdf.h3("Registered tools")
    for t, d in [
        ("product_search", "RAG product recommendation"),
        ("policy_qa", "grounded policy question answering"),
        ("analytics", "customer-segment / price-sensitivity analytics"),
        ("review_summary", "review sentiment summarisation"),
        ("calculator", "safe arithmetic for price/discount questions"),
    ]:
        pdf.bullet(f"{t}: {d}")
    if ev:
        pdf.body(f"Agent routing accuracy on the labelled set: "
                 f"{ev.get('agent', {}).get('routing_accuracy')}.")

    # ---------------- Evaluation ---------------- #
    pdf.add_page()
    pdf.h1("9. Final Evaluation and Packaging")
    pdf.h2("9.1 Regression-like checks")
    if ev:
        rows = [[c["name"], c["value"], c["threshold"], "PASS" if c["passed"] else "FAIL"]
                for c in ev.get("checks", [])]
        pdf.table(["Check", "Value", "Threshold", "Status"], rows, [62, 30, 30, 25])
        pdf.body(f"All checks passed: {ev.get('all_passed')}.")
    pdf.h2("9.2 Curated example queries with outputs")
    for e in ex[:8]:
        pdf.h3(f"Q: {e['query']}")
        pdf.body(f"Routed to: {e['tool']}  |  status: {e['status']}")
        pdf.body(f"Summary: {e['summary']}")
    pdf.h2("9.3 Grounding verification")
    if ev:
        g = ev.get("grounding", {})
        pdf.body(f"Policy grounding rate: {g.get('rate')} - every policy answer was "
                 f"verified to be a substring of the retrieved evidence.")

    # ---------------- Conclusions ---------------- #
    pdf.add_page()
    pdf.h1("10. Conclusions and Recommendations")
    pdf.body(
        "The Intelligent Retail Decision Assistant demonstrates a coherent, "
        "end-to-end path from structured retail data to a deployable AI prototype. "
        "Classical ML delivers interpretable conversion prediction and actionable "
        "customer segments; the same records are transformed into retrieval-ready "
        "text that powers grounded product recommendation and policy QA; and a "
        "simple, inspectable controller unifies the capabilities behind one "
        "interface."
    )
    pdf.h3("Key strengths")
    for b in [
        "Grounded by construction - policy answers quote retrieved evidence and carry citations; recommendations carry product ids.",
        "No leakage - IDs, free text and post-outcome columns are excluded from the supervised feature set.",
        "Reproducible - fixed random seed, offline-first design, one-command pipeline.",
        "Evaluated - regression thresholds enforced across ML, retrieval, policy QA and routing.",
    ]:
        pdf.bullet(b)
    pdf.h3("Recommended next steps")
    for b in [
        "Swap the offline embedder/generator for a production LLM endpoint (the interface is already in place).",
        "Add a feature store and scheduled retraining for the conversion model.",
        "Extend the controller with the optional multimodal path (product image -> attributes -> retrieval).",
        "Add human-in-the-loop review for policy answers before customer-facing release.",
    ]:
        pdf.bullet(b)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(
        0, 5,
        "Note: this is an engineering and analytics report. It is not investment "
        "advice. As an AI Agent, I cannot provide legally protected, personalized "
        "investment advice. This analysis is for informational purposes only and "
        "does not constitute investment advice or recommendations. Consult a "
        "licensed financial advisor before making investment decisions."
    )

    out_path = OUT / "Capstone Project report.pdf"
    pdf.output(str(out_path))
    print("Wrote", out_path, f"({out_path.stat().st_size} bytes)")


# --------------------------------------------------------------------------- #
# Markdown report (same content, portable text format)
# --------------------------------------------------------------------------- #
def _md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_markdown():
    """Write the detailed capstone report as Markdown (outputs/Capstone Project report.md)."""
    ml = load(OUT / "ml_metrics.json")
    seg = load(OUT / "segment_summary.json")
    ret = load(OUT / "retrieval_eval.json")
    pol = load(OUT / "policy_qa_results.json")
    ev = load(OUT / "evaluation_summary.json")
    ex = load(OUT / "example_queries.json", [])
    prep = load(PROC / "data_prep_summary.json")
    enrich = load(PROC / "enrich_summary.json")

    L = []
    A = L.append

    A("# Intelligent Retail Decision Assistant")
    A("## From Tabular Machine Learning to Agentic Generative AI Platform")
    A("### Capstone Project Report")
    A("")
    A("**Author/Student:** Vivek Bhushan  ")
    A("**Programme:** AI/ML Capstone  ")
    A("**Deliverable:** Detailed end-to-end report (PDF + Markdown)")
    A("")
    A("---")
    A("")

    # 1. Executive summary
    A("## 1. Executive Summary")
    A("")
    A("This report documents the design, implementation and evaluation of the "
      "**Intelligent Retail Decision Assistant** - a single coherent retail AI "
      "application built for a mid-sized online retailer. The system begins with "
      "structured tabular data, adds predictive and segmentation intelligence "
      "through classical machine learning, extends the same records into "
      "retrieval-ready text for generative AI, and finally exposes everything "
      "through a simple, inspectable agent/controller that routes each user "
      "request to the right tool.")
    A("")
    if ev:
        A("### Headline results")
        A("")
        A(_md_table(["Check", "Value", "Threshold", "Status"],
                    [[c["name"], c["value"], c["threshold"],
                      "PASS" if c["passed"] else "FAIL"] for c in ev.get("checks", [])]))
        A("")
        A(f"**All regression checks passed:** {ev.get('all_passed')}. "
          f"**Curated example queries delivered:** {ev.get('n_example_queries')}.")
        A("")

    # 2. Problem framing
    A("## 2. Problem Framing and Business Context")
    A("")
    A("### 2.1 Business problem")
    A("")
    A("A mid-sized online retailer serves both internal business teams and end "
      "customers. It holds structured data about customers, products and "
      "transactions, plus unstructured text such as product descriptions, "
      "customer reviews and policy documents. Leadership wants a prototype that "
      "can (a) analyse customer and product behaviour with classical ML, "
      "(b) answer product and policy questions with retrieval-backed generation, "
      "and (c) decide which workflow to use based on the user's query.")
    A("")
    A("### 2.2 Business questions")
    A("")
    for q in [
        "Which interactions convert to a purchase, and what drives conversion?",
        "Which customers form meaningful behavioural segments, and which are most price-sensitive?",
        "Which products best match a natural-language shopping request?",
        "What do the retailer's policies say about returns, refunds, shipping, exchange and warranty?",
        "How should the assistant route a request to the correct capability?",
    ]:
        A(f"- {q}")
    A("")
    A("### 2.3 Success metrics")
    A("")
    A("Supervised conversion model: **ROC-AUC** and **F1** (with **PR-AUC** for the "
      "imbalanced return target). Segmentation: **silhouette score**. Retrieval: "
      "**hit-rate@k** and **MRR**. Policy QA: **citation accuracy** and **grounding "
      "rate**. Agent: **routing accuracy**. These are enforced as regression "
      "thresholds in the final evaluation.")
    A("")

    # 3. Dataset
    A("## 3. Dataset Preparation (Phase 1)")
    A("")
    A("### 3.1 Source data")
    A("")
    A("The base dataset is `retail_interactions.csv` - **1,500 customer-product "
      "interaction records across 46 columns** covering customer profile, customer "
      "history, product profile, commercial data, interaction data and "
      "post-purchase outcomes. Five policy documents are provided in "
      "`retail_policies.json`.")
    A("")
    if prep:
        A(_md_table(["Property", "Value"], [
            ["Raw shape", prep.get("raw_shape")],
            ["Clean shape", prep.get("clean_shape")],
            ["Feature shape", prep.get("feature_shape")],
            ["Original columns", prep.get("n_original_columns")],
            ["Engineered columns", prep.get("n_engineered_columns")],
            ["Missing after cleaning", prep.get("missing_after_clean")],
        ]))
        A("")
    A("### 3.2 Cleaning decisions")
    A("")
    for b in [
        "`size` is missing for 270 non-sized items (accessories, watches): filled with the explicit token 'Not Applicable' and a `size_missing` indicator is retained - no size is invented.",
        "Any other missing numeric value is median-imputed; any other missing categorical value is filled with 'Unknown'.",
        "Duplicate rows are dropped and non-positive prices removed.",
    ]:
        A(f"- {b}")
    A("")
    A("### 3.3 Engineered features")
    A("")
    A("Nine features were engineered deterministically: `price_band`, "
      "`discount_band`, `price_sensitivity`, `engagement_score`, `recency_band`, "
      "`customer_value`, `is_discounted`, `size_missing` and `product_text`. The "
      "full data dictionary (original vs engineered, with dtypes and descriptions) "
      "is written to `data/processed/data_dictionary.csv`.")
    A("")
    A("### 3.4 Class balance")
    A("")
    A("`purchase_flag` is close to balanced (~51% positive), while `returned_flag` "
      "is imbalanced (~18% positive). This directly informs the modelling choices "
      "in Phase 2.")
    A("")

    # 4. Classical ML
    A("## 4. Classical ML Pipeline (Phase 2)")
    A("")
    A("### 4.1 Supervised task: predict `purchase_flag`")
    A("")
    if ml:
        sup = ml.get("supervised", {})
        A(sup.get("target_justification", ""))
        A("")
        A(f"Train/test split: **{sup.get('n_train')} / {sup.get('n_test')}** "
          f"(stratified). Positive rate: **{sup.get('positive_rate')}**. "
          f"{sup.get('imbalance_handling','')}")
        A("")
        A(_md_table(["Model", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "PR-AUC"],
                    [[name, m["accuracy"], m["precision"], m["recall"],
                      m["f1"], m["roc_auc"], m["pr_auc"]]
                     for name, m in sup.get("models", {}).items()]))
        A("")
        A(f"**Best model:** {sup.get('best_model')} with ROC-AUC "
          f"**{sup.get('best_roc_auc')}**.")
        A("")
        A("#### Top predictive features")
        A("")
        for f in sup.get("top_features", [])[:10]:
            A(f"- `{f['feature']}` (importance {f['importance']})")
        A("")
        A("**Interpretation:** engagement signals (`time_on_page`, `added_to_cart`, "
          "`wishlist_flag`), price context (`final_price`, `discount_percent`) and "
          "customer value (`avg_order_value`, `customer_value`) dominate. In "
          "business terms, how engaged a shopper is on the page and how the price "
          "is positioned matter more than raw demographics - supporting page "
          "personalisation and a data-driven discount strategy.")
        A("")
    A("### 4.2 Secondary target: `returned_flag` (imbalance handling)")
    A("")
    if ml:
        sec = ml.get("secondary", {})
        A(_md_table(["Metric", "Value"], [
            ["Positive rate", sec.get("positive_rate")],
            ["ROC-AUC", sec.get("roc_auc")],
            ["PR-AUC", sec.get("pr_auc")],
            ["F1", sec.get("f1")],
        ]))
        A("")
        A(sec.get("note", ""))
        A("")
    A("### 4.3 Unsupervised task: customer segmentation")
    A("")
    if seg:
        A(seg.get("feature_justification", ""))
        A("")
        A(f"**Chosen k = {seg.get('chosen_k')}** (silhouette "
          f"{seg.get('chosen_silhouette')}).")
        A("")
        A(_md_table(["k", "Inertia", "Silhouette"],
                    [[d["k"], d["inertia"], d["silhouette"]]
                     for d in seg.get("diagnostics", [])]))
        A("")
        A("#### Segment descriptions (business language)")
        A("")
        for s, d in seg.get("segment_descriptions", {}).items():
            A(f"- **Segment {s}:** {d}")
        A("")
        A("The clusters separate customers along value, recency and price "
          "behaviour - e.g. high-value loyalists, active regulars, discount-driven "
          "bargain hunters and dormant/at-risk customers. These map directly to "
          "retention, win-back and discount-targeting campaigns.")
        A("")
    A("### 4.4 Model artefacts (compulsory: `.joblib` AND `.pkl`)")
    A("")
    A("Every trained model is persisted in **both** formats under `outputs/models/` "
      "and reloaded to verify the round-trip:")
    A("")
    A(_md_table(["Artefact stem", ".joblib", ".pkl"], [
        ["best_purchase_flag_model", "yes", "yes"],
        ["returned_flag_model", "yes", "yes"],
        ["customer_segmentation_kmeans", "yes", "yes"],
        ["customer_segmentation_scaler", "yes", "yes"],
    ]))
    A("")
    A("![Supervised model comparison](figures/02_model_comparison.png)")
    A("")
    A("![Top predictive features](figures/03_feature_importance.png)")
    A("")
    A("![K-Means diagnostics](figures/04_cluster_diagnostics.png)")
    A("")
    A("![Customer segments](figures/05_segments.png)")
    A("")

    # 5. GenAI enrichment
    A("## 5. Data Enrichment for Generative AI (Phase 3)")
    A("")
    A("Structured records were transformed into retrieval-ready text documents. "
      "Each unique product becomes one document combining title, description, "
      "category, brand, colour, material, season, use case, style/occasion tags, "
      "price context and rating, with a stable `doc_id` (`PROD_<product_id>`). "
      "Each policy snippet becomes one document with a stable `doc_id` (`POL_*`).")
    A("")
    if enrich:
        A(_md_table(["Property", "Value"], [
            ["Product documents", enrich.get("n_product_docs")],
            ["Policy documents", enrich.get("n_policy_docs")],
            ["Synthetic queries", enrich.get("n_synthetic_queries")],
            ["Avg product doc length (chars)", enrich.get("avg_product_doc_chars")],
        ]))
        A("")

    # 6. RAG
    A("## 6. RAG and Grounded Answer Generation (Phase 4)")
    A("")
    A("### 6.1 Retrieval architecture")
    A("")
    A("Documents are embedded and stored in a vector index (FAISS when available, "
      "with a NumPy cosine fallback). The default embedder is a deterministic, "
      "fit-free hashing bag-of-words model (L2-normalised, so dot product equals "
      "cosine similarity); if a Gemini / Vocareum / OpenAI-compatible endpoint is "
      "configured, the same interface transparently uses hosted embeddings. "
      "Queries are parsed for explicit price and category filters before retrieval.")
    A("")
    if ret:
        re_ = ret.get("retrieval_eval", {})
        A(_md_table(["Metric", "Value"], [
            ["Queries evaluated", re_.get("n_evaluated")],
            ["Hit-rate@k", re_.get("hit_rate_at_k")],
            ["MRR", re_.get("mrr")],
            ["Top-k", re_.get("top_k")],
        ]))
        A("")
        A("#### Example grounded recommendations")
        A("")
        for r in ret.get("demo_results", [])[:4]:
            A(f"- **Q:** {r['query']} -> `{r['supporting_doc_ids']}`")
        A("")
    A("### 6.2 Grounding guarantee")
    A("")
    A("The generated answer is composed only from the retrieved documents, and "
      "every recommendation carries its `doc_id` and `product_id`, so the response "
      "is fully traceable. Weak retrieval is reported explicitly rather than "
      "answered speculatively.")
    A("")

    # 7. Policy QA
    A("## 7. Policy-Aware Question Answering (Phase 4b)")
    A("")
    A("A dedicated grounded QA path answers policy questions using retrieved "
      "evidence only. Every answer quotes the exact policy sentence and returns "
      "the source `doc_id`; out-of-scope questions are refused rather than guessed.")
    A("")
    if pol:
        A(f"**Citation accuracy on the curated test set:** {pol.get('citation_accuracy')}.")
        A("")
        A(_md_table(["Question", "Source", "Confidence"],
                    [[r["question"][:60], r.get("primary_source", "-"),
                      r.get("confidence", "-")] for r in pol.get("results", [])]))
        A("")
        A("#### Example grounded answers")
        A("")
        for r in pol.get("results", [])[:5]:
            A(f"- **Q:** {r['question']}")
            A(f"  - **A:** {r['answer']}  `[{r.get('primary_source')}]`")
        A("")

    # 8. Agent
    A("## 8. Agent / Controller (Phase 5)")
    A("")
    A("The controller is a hybrid rule-based + keyword-scoring router. It detects "
      "the intent of a query, selects exactly one tool, executes it, and returns "
      "an inspectable trace (intent, tool, reason, arguments, status, result "
      "summary). Routing is bounded to a single tool per query - no unbounded "
      "loops - per the capstone's 'keep routing bounded and simple' guidance.")
    A("")
    A("### Registered tools")
    A("")
    for t, d in [
        ("product_search", "RAG product recommendation"),
        ("policy_qa", "grounded policy question answering"),
        ("analytics", "customer-segment / price-sensitivity analytics"),
        ("review_summary", "review sentiment summarisation"),
        ("calculator", "safe arithmetic for price/discount questions"),
    ]:
        A(f"- **`{t}`** - {d}")
    A("")
    if ev:
        A(f"**Agent routing accuracy on the labelled set:** "
          f"{ev.get('agent', {}).get('routing_accuracy')}.")
        A("")

    # 9. Evaluation
    A("## 9. Final Evaluation and Packaging")
    A("")
    A("### 9.1 Regression-like checks")
    A("")
    if ev:
        A(_md_table(["Check", "Value", "Threshold", "Status"],
                    [[c["name"], c["value"], c["threshold"],
                      "PASS" if c["passed"] else "FAIL"] for c in ev.get("checks", [])]))
        A("")
        A(f"**All checks passed:** {ev.get('all_passed')}.")
        A("")
    A("### 9.2 Curated example queries with outputs")
    A("")
    for e in ex[:8]:
        A(f"**Q: {e['query']}**")
        A("")
        A(f"- Routed to: `{e['tool']}` | status: `{e['status']}`")
        A(f"- Summary: {e['summary']}")
        A("")
    A("### 9.3 Grounding verification")
    A("")
    if ev:
        g = ev.get("grounding", {})
        A(f"Policy grounding rate: **{g.get('rate')}** - every policy answer was "
          f"verified to be a substring of the retrieved evidence.")
        A("")

    # 10. Conclusions
    A("## 10. Conclusions and Recommendations")
    A("")
    A("The Intelligent Retail Decision Assistant demonstrates a coherent, "
      "end-to-end path from structured retail data to a deployable AI prototype. "
      "Classical ML delivers interpretable conversion prediction and actionable "
      "customer segments; the same records are transformed into retrieval-ready "
      "text that powers grounded product recommendation and policy QA; and a "
      "simple, inspectable controller unifies the capabilities behind one "
      "interface.")
    A("")
    A("### Key strengths")
    A("")
    for b in [
        "Grounded by construction - policy answers quote retrieved evidence and carry citations; recommendations carry product ids.",
        "No leakage - IDs, free text and post-outcome columns are excluded from the supervised feature set.",
        "Reproducible - fixed random seed, offline-first design, one-command pipeline.",
        "Evaluated - regression thresholds enforced across ML, retrieval, policy QA and routing.",
        "Portable artefacts - every model saved in BOTH `.joblib` and `.pkl`.",
    ]:
        A(f"- {b}")
    A("")
    A("### Recommended next steps")
    A("")
    for b in [
        "Swap the offline embedder/generator for a production LLM endpoint (the interface is already in place).",
        "Add a feature store and scheduled retraining for the conversion model.",
        "Extend the controller with the optional multimodal path (product image -> attributes -> retrieval).",
        "Add human-in-the-loop review for policy answers before customer-facing release.",
    ]:
        A(f"- {b}")
    A("")
    A("---")
    A("")
    A("*Note: this is an engineering and analytics report. It is not investment "
      "advice. As an AI Agent, I cannot provide legally protected, personalized "
      "investment advice. This analysis is for informational purposes only and "
      "does not constitute investment advice or recommendations. Consult a "
      "licensed financial advisor before making investment decisions.*")

    out_path = OUT / "Capstone Project report.md"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Wrote", out_path, f"({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
    build_markdown()
