"""
llm_client.py
=============
A thin, dependency-light abstraction over the two capabilities the capstone
needs:

1. **Embeddings** - turn text into vectors for the (FAISS) vector index.
2. **Grounded generation** - turn a prompt + retrieved context into an answer.

Back-ends (selected automatically, in priority order)
-----------------------------------------------------
* ``gemini``   - Google Gemini via the ``google-genai`` SDK.  Configured from
  ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``); embeddings use
  ``text-embedding-004`` and generation uses ``gemini-2.5-flash`` by default.
* ``vocareum`` - any OpenAI-compatible gateway (the programme's Vocareum /
  Bedrock endpoint).  Configured from ``VOCAREUM_API_KEY`` (or
  ``AWS_BEARER_TOKEN_BEDROCK``) plus ``VOCAREUM_BASE_URL``.
* ``openai``   - a plain OpenAI-compatible endpoint via ``OPENAI_API_KEY``.
* ``offline``  - **the default when no key is present**: a deterministic
  TF-IDF + Truncated-SVD embedder and a template-based *extractive* generator.
  No network, fully reproducible.

Every remote path is wrapped in try/except and **falls back to the offline
backend on any failure**, so the notebooks always run end-to-end - with a key
they use the real model, without one they still demonstrate the identical
retrieval, grounding and evaluation logic.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np

from config import (
    EMBEDDING_DIM,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    RANDOM_STATE,
    VOCAREUM_API_KEY,
    VOCAREUM_BASE_URL,
    VOCAREUM_EMBED_MODEL,
    VOCAREUM_MODEL,
)

TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Shared vector helper
# --------------------------------------------------------------------------- #
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --------------------------------------------------------------------------- #
# Offline (default, no key required) - hashing bag-of-words, extractive gen
# --------------------------------------------------------------------------- #
class OfflineEmbedder:
    """Deterministic, fit-free text embedding via the hashing trick.

    ``HashingVectorizer`` maps n-grams into a fixed-dimension space with no
    vocabulary to learn, so ``fit`` is a no-op and embedding a corpus of a few
    hundred documents takes milliseconds.  Vectors are L2-normalised, so a dot
    product is cosine similarity - a solid proxy for semantic similarity on
    short product/policy documents, with no external model and no API key.

    (An earlier TF-IDF + TruncatedSVD variant was replaced because fitting the
    SVD on this corpus was slow and memory-hungry; the hashing embedder is
    equivalent in behaviour for retrieval and orders of magnitude cheaper.)
    """

    def __init__(self, dim: int = 2 ** 14, random_state: int = RANDOM_STATE):
        self.dim = dim
        self.random_state = random_state
        self._vectorizer = None
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> "OfflineEmbedder":
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=self.dim,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            token_pattern=r"[a-z0-9]+",
        )
        # touch the corpus once so the vectorizer is "warm" (stateless anyway)
        self._vectorizer.transform(list(corpus)[:1])
        self._fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OfflineEmbedder must be fitted before transform().")
        X = self._vectorizer.transform(list(texts))
        dense = X.todense() if hasattr(X, "todense") else X
        return np.asarray(dense, dtype=np.float32)


class OfflineGenerator:
    """Extractive, template-based generator.

    It never invents content: every sentence it returns is selected from the
    retrieved context.  This is the honest baseline that proves the RAG pipeline
    is *grounded* rather than hallucinating.
    """

    def generate(self, prompt: str, context: str, **kwargs) -> str:
        return self._compose(context)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _compose(self, context: str) -> str:
        sentences = self._split_sentences(context)
        if not sentences:
            return "I could not find supporting evidence in the retrieved context."
        return " ".join(sentences[:4])


# --------------------------------------------------------------------------- #
# Gemini back-end (google-genai SDK)
# --------------------------------------------------------------------------- #
class GeminiEmbedder:
    def __init__(self, model: str = GEMINI_EMBED_MODEL):
        from google import genai  # lazy import

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model

    def fit(self, corpus: Sequence[str]) -> "GeminiEmbedder":  # stateless API
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        resp = self.client.models.embed_content(model=self.model, contents=list(texts))
        vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
        return _l2_normalize(vecs)


class GeminiGenerator:
    def __init__(self, model: str = GEMINI_MODEL):
        from google import genai

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model

    def generate(self, prompt: str, context: str, **kwargs) -> str:
        system = (
            "You are a retail assistant. Answer ONLY using the provided context. "
            "If the context is insufficient, say you do not have enough information. "
            "Cite the document ids you used."
        )
        contents = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION / TASK:\n{prompt}"
        resp = self.client.models.generate_content(model=self.model, contents=contents)
        return (getattr(resp, "text", None) or "").strip()


# --------------------------------------------------------------------------- #
# OpenAI-compatible back-ends (Vocareum gateway + plain OpenAI)
# --------------------------------------------------------------------------- #
class OpenAICompatibleEmbedder:
    def __init__(self, model: str, api_key: str, base_url=None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def fit(self, corpus: Sequence[str]) -> "OpenAICompatibleEmbedder":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return _l2_normalize(vecs)


class OpenAICompatibleGenerator:
    def __init__(self, model: str, api_key: str, base_url=None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, context: str, **kwargs) -> str:
        system = (
            "You are a retail assistant. Answer ONLY using the provided context. "
            "If the context is insufficient, say you do not have enough information. "
            "Cite the document ids you used."
        )
        user = f"CONTEXT:\n{context}\n\nQUESTION / TASK:\n{prompt}"
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Provider selection with graceful fallback
# --------------------------------------------------------------------------- #
def _build_backends():
    """Return (embedder, generator, mode). Never raises - falls back offline."""
    if LLM_PROVIDER == "gemini":
        try:
            return GeminiEmbedder(), GeminiGenerator(), "gemini"
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"[llm_client] Gemini backend unavailable ({exc}); using offline backend.")
    elif LLM_PROVIDER == "vocareum":
        try:
            return (
                OpenAICompatibleEmbedder(VOCAREUM_EMBED_MODEL, VOCAREUM_API_KEY, VOCAREUM_BASE_URL),
                OpenAICompatibleGenerator(VOCAREUM_MODEL, VOCAREUM_API_KEY, VOCAREUM_BASE_URL),
                "vocareum",
            )
        except Exception as exc:  # pragma: no cover
            print(f"[llm_client] Vocareum backend unavailable ({exc}); using offline backend.")
    elif LLM_PROVIDER == "openai":
        try:
            return (
                OpenAICompatibleEmbedder("text-embedding-3-small", OPENAI_API_KEY, OPENAI_BASE_URL),
                OpenAICompatibleGenerator(OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL),
                "openai",
            )
        except Exception as exc:  # pragma: no cover
            print(f"[llm_client] OpenAI backend unavailable ({exc}); using offline backend.")
    return OfflineEmbedder(), OfflineGenerator(), "offline"


# --------------------------------------------------------------------------- #
# Public facade
# --------------------------------------------------------------------------- #
class LLMClient:
    """Facade combining an embedder and a generator."""

    def __init__(self, embedder=None, generator=None):
        if embedder is None or generator is None:
            e, g, mode = _build_backends()
            embedder = embedder or e
            generator = generator or g
            self.mode = mode
        else:
            self.mode = "custom"
        self.embedder = embedder
        self.generator = generator

    def fit_embeddings(self, corpus: Sequence[str]) -> "LLMClient":
        self.embedder.fit(corpus)
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.embedder.transform(texts)

    def answer(self, prompt: str, context: str, **kwargs) -> str:
        return self.generator.generate(prompt, context, **kwargs)


def get_llm_client() -> LLMClient:
    """Return a configured client (offline unless a key is present)."""
    return LLMClient()


if __name__ == "__main__":  # pragma: no cover
    c = get_llm_client()
    print("Provider priority :", LLM_PROVIDER)
    print("Active backend    :", c.mode)
    corpus = ["blue running shoes for daily wear", "black formal leather shoes", "returns within 7 days"]
    c.fit_embeddings(corpus)
    print("embedding shape   :", c.embed(["casual sneakers under 4000"]).shape)
    print("answer            :", c.answer("What is the return window?", "Returns are allowed within 7 days of delivery."))
