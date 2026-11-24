"""
nb_setup.py
===========
Small helper module imported by every notebook so the five notebooks share ONE
environment-setup, API-configuration and asset-display implementation.

It deliberately imports nothing heavy at module import time, so a notebook can
call it before any pip install has happened.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# 1. Project paths
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Resolve the project root whether the notebook runs from ./notebooks or
the project root (also works in Colab after unzipping the package)."""
    cwd = Path.cwd()
    root = cwd.parent if cwd.name == "notebooks" else cwd
    # If we were handed a path that is not the project root, walk up until we
    # find the marker file (requirements.txt) - robust to Colab restarts.
    if not (root / "requirements.txt").exists():
        for cand in [root, *root.parents]:
            if (cand / "requirements.txt").exists() and (cand / "src").exists():
                root = cand
                break
    return root


def bootstrap():
    """Put ``src/`` on the path and chdir to the project root."""
    root = project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(root)
    return root


# --------------------------------------------------------------------------- #
# 2. Environment / version report
# --------------------------------------------------------------------------- #
def environment_report() -> dict:
    """Print and return a version report for the packages the capstone uses."""
    from importlib import metadata as _md

    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable.split("/")[-1],
        "platform": platform.platform(),
        "in_colab": "google.colab" in sys.modules or "COLAB_GPU" in os.environ,
    }
    pkgs = [
        "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "joblib",
        "faiss-cpu", "openai", "google-genai", "xgboost", "lightgbm",
    ]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = _md.version(p)
        except Exception:
            versions[p] = "not installed"
    info["packages"] = versions

    print("Environment")
    print("-" * 60)
    print(f"  python     : {info['python']}")
    print(f"  platform   : {info['platform']}")
    print(f"  colab      : {info['in_colab']}")
    for k, v in versions.items():
        print(f"  {k:12s}: {v}")
    return info


def install_hint() -> None:
    """Print the exact command to install missing dependencies."""
    print("\nTo install every dependency, run (from the project root):")
    print("    pip install -r requirements.txt")
    print("In Google Colab, prefix with '!':  !pip install -r requirements.txt")


# --------------------------------------------------------------------------- #
# 3. API configuration (Gemini AND Vocareum, env-var based, graceful fallback)
# --------------------------------------------------------------------------- #
GEMINI_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
VOCAREUM_ENV_NAMES = ("VOCAREUM_API_KEY", "AWS_BEARER_TOKEN_BEDROCK")


def configure_apis(verbose: bool = True) -> dict:
    """Configure the Gemini and Vocareum clients from environment variables.

    Both are OPTIONAL.  Returns a report dict describing what was found and
    which backend the pipeline will actually use.  Never raises and never
    prints or stores the key value itself - only whether it is present.
    """
    import config  # lives in src/, already on sys.path

    def _present(names):
        for n in names:
            v = os.environ.get(n, "").strip()
            if v:
                return n
        return None

    gemini_var = _present(GEMINI_ENV_NAMES)
    vocareum_var = _present(VOCAREUM_ENV_NAMES)
    openai_var = "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY", "").strip() else None

    report = {
        "gemini_key_found": gemini_var,
        "vocareum_key_found": vocareum_var,
        "openai_key_found": openai_var,
        "gemini_model": config.GEMINI_MODEL,
        "gemini_embed_model": config.GEMINI_EMBED_MODEL,
        "vocareum_base_url": config.VOCAREUM_BASE_URL,
        "provider_priority": config.LLM_PROVIDER,
    }

    # Try to *construct* the Gemini client if a key is present (this validates
    # the SDK is installed); construction makes no network call.
    gemini_client = None
    if gemini_var:
        try:
            from google import genai  # noqa: F401
            from google import genai as _g

            gemini_client = _g.Client(api_key=os.environ.get(gemini_var))
            report["gemini_client"] = "initialised"
        except Exception as exc:
            report["gemini_client"] = f"unavailable ({type(exc).__name__})"
    else:
        report["gemini_client"] = "no key - not initialised"

    # Try to construct the Vocareum (OpenAI-compatible) client if a key exists.
    vocareum_client = None
    if vocareum_var:
        try:
            from openai import OpenAI

            vocareum_client = OpenAI(
                api_key=os.environ.get(vocareum_var), base_url=config.VOCAREUM_BASE_URL
            )
            report["vocareum_client"] = "initialised"
        except Exception as exc:
            report["vocareum_client"] = f"unavailable ({type(exc).__name__})"
    else:
        report["vocareum_client"] = "no key - not initialised"

    # Which backend will the shared LLM facade actually use?
    from llm_client import get_llm_client

    client = get_llm_client()
    report["active_backend"] = client.mode

    if verbose:
        print("API configuration (all optional - offline fallback is automatic)")
        print("-" * 60)
        print(f"  Gemini key      : {report['gemini_key_found'] or 'not set'}")
        print(f"  Gemini client   : {report['gemini_client']}")
        print(f"  Vocareum key    : {report['vocareum_key_found'] or 'not set'}")
        print(f"  Vocareum client : {report['vocareum_client']}")
        print(f"  OpenAI key      : {report['openai_key_found'] or 'not set'}")
        print(f"  -> active backend: {report['active_backend'].upper()}")
        if report["active_backend"] == "offline":
            print(
                "     (No key detected: using the deterministic offline embedder + "
                "extractive generator.\n      Set GEMINI_API_KEY and/or VOCAREUM_API_KEY "
                "to switch to a hosted model - the logic is identical.)"
            )
    report["client"] = client
    return report


# --------------------------------------------------------------------------- #
# 4. Asset display
# --------------------------------------------------------------------------- #
def asset_path(name: str) -> Path:
    """Absolute path to a provided asset under ``assets/attached_images``."""
    return project_root() / "assets" / "attached_images" / name


def display_asset(name: str, caption: str | None = None, width: int | None = None) -> bool:
    """Render a provided PNG inline (works after the notebook is executed).

    Returns True if the image was displayed, False if it was missing (in which
    case a readable warning is printed instead of a crash).
    """
    from IPython.display import Image, Markdown, display

    p = asset_path(name)
    if p.exists():
        if caption:
            display(Markdown(f"**{caption}**"))
        display(Image(filename=str(p), width=width) if width else Image(filename=str(p)))
        return True
    display(Markdown(f"*[asset `{name}` not found at `{p}` - skipped]*"))
    return False


def read_links(path_name: str = "links_to_attach.txt") -> list:
    """Read the provided 'Links to attach.txt' file into a list of URLs."""
    p = project_root() / "assets" / path_name
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
