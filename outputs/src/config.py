"""
config.py
=========
Central configuration for the Intelligent Retail Decision Assistant capstone.

All paths are resolved relative to the project root so that the code runs
identically from a notebook, from the command line, or from Colab (after the
project folder is unzipped).

Design notes
------------
* The project is designed to run **fully offline** out of the box (no API key
  required) so that every notebook is reproducible and every metric is stable.
* If an OpenAI-compatible endpoint is available, set the environment variables
  ``OPENAI_API_KEY`` and (optionally) ``OPENAI_BASE_URL`` / ``OPENAI_MODEL`` and
  the LLM client will automatically switch to real LLM generation.  The
  retrieval, grounding and evaluation logic is identical in both modes.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
POLICIES_DIR = DATA_DIR / "policies"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
INDEX_DIR = OUTPUTS_DIR / "index"

# --------------------------------------------------------------------------- #
# Model artefact naming
# --------------------------------------------------------------------------- #
# Every trained model is persisted in BOTH .joblib and .pkl - both formats are
# compulsory deliverables - so downstream consumers can standardise on either.
MODEL_STEMS = [
    "best_purchase_flag_model",       # champion supervised classifier (purchase_flag)
    "returned_flag_model",            # secondary imbalanced classifier (returned_flag)
    "customer_segmentation_kmeans",   # K-Means customer segmentation model
    "customer_segmentation_scaler",   # StandardScaler fitted before K-Means
]


def model_artifact_paths(stem: str) -> dict:
    """Return {'joblib': Path, 'pkl': Path} for a model artefact stem."""
    return {
        "joblib": MODELS_DIR / f"{stem}.joblib",
        "pkl": MODELS_DIR / f"{stem}.pkl",
    }


RAW_INTERACTIONS = RAW_DIR / "retail_interactions.csv"
POLICIES_JSON = POLICIES_DIR / "retail_policies.json"

# Processed artefacts
CLEAN_INTERACTIONS = PROCESSED_DIR / "retail_interactions_clean.csv"
FEATURES_CSV = PROCESSED_DIR / "ml_features.csv"
SEGMENTS_CSV = PROCESSED_DIR / "customer_segments.csv"
PRODUCT_DOCS_JSONL = PROCESSED_DIR / "product_documents.jsonl"
POLICY_DOCS_JSONL = PROCESSED_DIR / "policy_documents.jsonl"
DATA_DICTIONARY_CSV = PROCESSED_DIR / "data_dictionary.csv"

# Output artefacts
ML_METRICS_JSON = OUTPUTS_DIR / "ml_metrics.json"
SEGMENT_SUMMARY_JSON = OUTPUTS_DIR / "segment_summary.json"
RETRIEVAL_EVAL_JSON = OUTPUTS_DIR / "retrieval_eval.json"
POLICY_QA_JSON = OUTPUTS_DIR / "policy_qa_results.json"
AGENT_TRACE_JSON = OUTPUTS_DIR / "agent_traces.json"
EVALUATION_JSON = OUTPUTS_DIR / "evaluation_summary.json"
EXAMPLE_QUERIES_JSON = OUTPUTS_DIR / "example_queries.json"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
SUPERVISED_TARGET = "purchase_flag"          # primary supervised target
SECONDARY_TARGET = "returned_flag"           # secondary (imbalanced) target
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Retrieval / GenAI
# --------------------------------------------------------------------------- #
EMBEDDING_DIM = 256          # dimension of the offline hashing/SVD embedding
TOP_K = 5                    # default number of retrieved documents
MIN_RETRIEVAL_SCORE = 0.05   # below this we treat retrieval as "weak"

# --------------------------------------------------------------------------- #
# LLM / embeddings providers (all OPTIONAL - offline fallback otherwise)
# --------------------------------------------------------------------------- #
# The notebooks configure BOTH a Gemini key and a Vocareum (OpenAI-compatible)
# key from environment variables.  Nothing here requires a key: if none is
# present the deterministic offline backend is used, so every notebook stays
# reproducible.

# --- Google Gemini (google-genai SDK) -------------------------------------- #
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY", "")
    or os.environ.get("GOOGLE_API_KEY", "")
).strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004").strip()

# --- Vocareum (OpenAI-compatible gateway / Bedrock-style) ------------------ #
VOCAREUM_API_KEY = (
    os.environ.get("VOCAREUM_API_KEY", "")
    or os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
).strip()
VOCAREUM_BASE_URL = (os.environ.get("VOCAREUM_BASE_URL", "").strip()) or None
VOCAREUM_MODEL = os.environ.get(
    "VOCAREUM_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
).strip()
VOCAREUM_EMBED_MODEL = os.environ.get(
    "VOCAREUM_EMBED_MODEL", "amazon.titan-embed-text-v2:0"
).strip()

# --- Generic OpenAI-compatible endpoint (backwards compatibility) ----------- #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip() or None
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()


def active_provider() -> str:
    """Return the highest-priority configured remote provider (or 'offline').

    Priority: Gemini -> Vocareum -> OpenAI -> offline.  A learner can set either
    key (or both) and the pipeline picks it up automatically.
    """
    if GEMINI_API_KEY:
        return "gemini"
    if VOCAREUM_API_KEY:
        return "vocareum"
    if OPENAI_API_KEY:
        return "openai"
    return "offline"


LLM_PROVIDER = active_provider()
USE_REMOTE_LLM = LLM_PROVIDER != "offline"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        POLICIES_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        INDEX_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":  # pragma: no cover
    ensure_dirs()
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("USE_REMOTE_LLM:", USE_REMOTE_LLM)
    for name in ("RAW_INTERACTIONS", "POLICIES_JSON", "PROCESSED_DIR", "OUTPUTS_DIR"):
        print(f"{name:18s}:", globals()[name])
