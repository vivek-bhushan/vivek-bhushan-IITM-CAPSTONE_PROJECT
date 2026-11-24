"""
gpu_utils.py
============
Shared GPU / Google-Colab helpers used by **all five** capstone notebooks.

Why this module exists
----------------------
The capstone must run in *two* very different places:

* **Google Colab with an NVIDIA T4 GPU (16 GB)**  - the recommended runtime.
  Notebooks should exploit the GPU, print what hardware they landed on, and use
  mixed precision without over-running the 16 GB of VRAM.
* **A plain CPU machine / CI runner**             - no CUDA at all.  Everything
  must still work, just slower, with zero code changes.

This module centralises that logic so the notebooks stay short and readable:

1. ``is_colab()``            - detect the runtime (``google.colab``).
2. ``colab_install()``       - Colab-guarded ``pip install`` (never runs off-Colab).
3. ``detect_gpu()``          - ``torch.cuda.is_available()``, device **name**
                               (expects ``Tesla T4``), **VRAM**, **CUDA version**.
4. ``get_device()``          - ``torch.device("cuda")`` when available, else CPU.
5. ``empty_cache()``         - ``torch.cuda.empty_cache()`` guarded for CPU.
6. ``autocast_context()``    - ``torch.cuda.amp`` mixed precision on GPU.
7. ``vram_snapshot()``       - allocated / reserved VRAM after a step.
8. ``t4_memory_tips()``      - the printed guidance for staying inside 16 GB.

Nothing here imports ``torch`` at module import time, so the module can be
imported before/without PyTorch being installed.  Every function degrades
gracefully to a CPU-only code path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

__all__ = [
    "IN_COLAB",
    "is_colab",
    "colab_install",
    "torch_available",
    "detect_gpu",
    "get_device",
    "device_name",
    "empty_cache",
    "autocast_context",
    "vram_snapshot",
    "recommend_batch_size",
    "t4_memory_tips",
    "setup_gpu",
]

# --------------------------------------------------------------------------- #
# 1. Runtime detection
# --------------------------------------------------------------------------- #
IN_COLAB: bool = (
    "google.colab" in sys.modules
    or bool(os.environ.get("COLAB_GPU"))
    or bool(os.environ.get("COLAB_RELEASE_TAG"))
    or os.path.exists("/content")
)


def is_colab() -> bool:
    """Return ``True`` when running inside Google Colab.

    Detected three ways so it is robust whether Colab exports its env vars or
    not: the ``google.colab`` module, the ``COLAB_*`` environment variables, or
    the presence of the ``/content`` mount that only Colab has.
    """
    return (
        "google.colab" in sys.modules
        or bool(os.environ.get("COLAB_GPU"))
        or bool(os.environ.get("COLAB_RELEASE_TAG"))
        or os.path.exists("/content")
    )


def recommend_runtime_note() -> None:
    """Print the exact Colab menu path to obtain a T4 GPU."""
    print("  Runtime is NOT a GPU runtime.  To enable the T4 GPU:")
    print("    Runtime  >  Change runtime type  >  Hardware accelerator  >  T4 GPU  >  Save")


# --------------------------------------------------------------------------- #
# 2. Colab-guarded installs
# --------------------------------------------------------------------------- #
def colab_install(requirements: str = "requirements.txt", quiet: bool = True) -> bool:
    """Install the project requirements **only inside Google Colab**.

    Guarded with a ``try/except ImportError`` style check on ``google.colab`` so
    that a local/CI run never touches the network or the user's environment.
    Returns ``True`` when an install was attempted, ``False`` when skipped.
    """
    if not is_colab():
        print("[gpu_utils] Not in Colab - skipping auto-install.")
        print("            If a package is missing, run:  pip install -r requirements.txt")
        return False

    print("[gpu_utils] Google Colab detected - installing requirements ...")
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd += ["-r", requirements]
    try:
        subprocess.run(cmd, check=False)
        print("[gpu_utils] requirements install finished.")
    except Exception as exc:  # pragma: no cover - network issues
        print("[gpu_utils] install skipped:", type(exc).__name__, exc)
    return True


def torch_available() -> bool:
    """Return ``True`` when PyTorch can be imported."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 3. GPU detection & reporting
# --------------------------------------------------------------------------- #
def detect_gpu(verbose: bool = True) -> Dict[str, object]:
    """Detect the accelerator and return a report dictionary.

    Keys
    ----
    ``torch``              - torch version (or ``None`` if torch is absent)
    ``cuda_available``     - ``torch.cuda.is_available()``
    ``device_name``        - e.g. ``"Tesla T4"``
    ``vram_gb``            - total VRAM in GB (rounded to 1 dp)
    ``cuda_version``       - the CUDA build torch was compiled against
    ``sm``                 - compute capability, e.g. ``"7.5"`` (T4 = 7.5)
    ``is_t4``              - whether the GPU is a Tesla T4
    ``device``             - the chosen device string: ``"cuda"`` or ``"cpu"``
    ``in_colab``           - whether we are running in Colab
    """
    report: Dict[str, object] = {
        "torch": None,
        "cuda_available": False,
        "device_name": None,
        "vram_gb": None,
        "cuda_version": None,
        "sm": None,
        "is_t4": False,
        "device": "cpu",
        "in_colab": is_colab(),
    }

    if verbose:
        print("=" * 74)
        print("GPU / ACCELERATOR SETUP")
        print("=" * 74)
        print("  Colab runtime            :", "yes (google.colab)" if report["in_colab"] else "no")

    try:
        import torch
    except Exception as exc:
        if verbose:
            print("  PyTorch                  : NOT INSTALLED ->", type(exc).__name__)
            print("  Falling back to            CPU (the pipeline is device-agnostic).")
            if report["in_colab"]:
                print("  In Colab, install with    : !pip install torch")
        return report

    report["torch"] = torch.__version__
    report["cuda_version"] = getattr(torch.version, "cuda", None)

    cuda_ok = bool(torch.cuda.is_available())
    report["cuda_available"] = cuda_ok

    if verbose:
        print("  PyTorch                  :", torch.__version__)
        print("  torch.cuda.is_available():", cuda_ok)

    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 1)
        sm = f"{props.major}.{props.minor}"
        report.update(
            device_name=name,
            vram_gb=vram_gb,
            sm=sm,
            is_t4=("T4" in name.upper()),
            device="cuda",
        )
        if verbose:
            print("  Device name              :", name)
            print("  VRAM (total)             : %.1f GB" % vram_gb)
            print("  CUDA version (torch)     :", report["cuda_version"])
            print("  Compute capability       : sm_%s" % sm)
            print("  Device count             :", torch.cuda.device_count())
            if report["is_t4"]:
                print("  -> NVIDIA Tesla T4 detected: mixed precision (float16) is supported (Turing).")
            else:
                print("  -> GPU detected (not a T4); the same code path applies.")
        report["device"] = "cuda"
    else:
        if verbose:
            print("  CUDA not available       : running on CPU (device-agnostic fallback).")
            recommend_runtime_note()

    if verbose:
        print("  Selected torch device    :", report["device"])
        print("=" * 74)
    return report


def get_device():
    """Return a ``torch.device`` - ``cuda`` if a GPU is present, else ``cpu``.

    Falls back to the *string* ``"cpu"`` if torch is not installed, so callers
    can always pass the result to ``.to(...)`` without crashing.
    """
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


def device_name() -> str:
    """Human-readable description of the active device (``"cuda:Tesla T4"`` / ``"cpu"``)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:" + torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# 4. VRAM hygiene (16 GB T4)
# --------------------------------------------------------------------------- #
def empty_cache() -> None:
    """Call ``torch.cuda.empty_cache()`` when a GPU is present; no-op on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return
    except Exception:
        pass


def vram_snapshot(tag: str = "", verbose: bool = True) -> Optional[Dict[str, float]]:
    """Print (and return) allocated/reserved VRAM in GB. ``None`` on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if verbose:
            print(
                "  VRAM%s: allocated %.2f GB | reserved %.2f GB | peak %.2f GB"
                % ((" [" + tag + "]") if tag else "", alloc, reserved, peak)
            )
        return {"allocated_gb": round(alloc, 3), "reserved_gb": round(reserved, 3), "peak_gb": round(peak, 3)}
    except Exception:
        return None


def recommend_batch_size(fallback: int = 32) -> int:
    """Suggest a safe starting batch size for the detected VRAM.

    A deliberate *conservative* heuristic: on a 16 GB T4 start around 64, and
    scale down on smaller cards.  This is a starting point to be tuned, not a
    guarantee - see :func:`t4_memory_tips`.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return fallback
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram >= 15:
            return 64
        if vram >= 10:
            return 32
        if vram >= 6:
            return 16
        return 8
    except Exception:
        return fallback


# --------------------------------------------------------------------------- #
# 5. Mixed precision (torch.cuda.amp)
# --------------------------------------------------------------------------- #
class _NullAutocast:
    """A no-op context manager used on CPU so call sites stay identical."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def autocast_context(device=None, enabled: bool = True):
    """Return an autocast context manager for mixed precision.

    On a CUDA device returns ``torch.autocast(device_type="cuda",
    dtype=torch.float16)`` (the modern ``torch.cuda.amp`` replacement); on CPU it
    returns a no-op context so the same ``with`` block runs everywhere.
    """
    try:
        import torch

        dev = device if device is not None else get_device()
        dev_type = getattr(dev, "type", str(dev))
        if enabled and dev_type == "cuda" and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        pass
    return _NullAutocast()


def amp_grad_scaler(enabled: bool = True):
    """Return a ``torch.cuda.amp.GradScaler`` when on GPU, else a no-op scaler.

    Provided for completeness (gradient scaling for float16 training); the
    capstone's models are CPU/sklearn, so this is documented rather than used.
    """
    try:
        import torch

        if enabled and torch.cuda.is_available():
            return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        pass

    class _NoOpScaler:
        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return None

    return _NoOpScaler()


# --------------------------------------------------------------------------- #
# 6. T4 (16 GB) memory guidance
# --------------------------------------------------------------------------- #
def t4_memory_tips(verbose: bool = True) -> str:
    """Return (and print) the practical guidance for staying inside 16 GB."""
    tips = """
MEMORY & VRAM TIPS  -  NVIDIA T4 (16 GB)
--------------------------------------------------------------------------
1. Batch sizing        Start at batch_size=32-64 for a 16 GB T4 and grow until
                       `nvidia-smi` (or vram_snapshot()) shows ~80% use. Halve
                       the batch the moment you see a CUDA OOM.
2. Mixed precision     Wrap the forward pass in `with autocast_context(device):`
                       (torch.cuda.amp / torch.autocast, float16). On a Turing
                       T4 this ~halves activation memory and speeds training up,
                       with no accuracy loss for most workloads.
3. Gradient accumulation  If the batch you *want* does not fit, keep a small
                       physical batch and accumulate gradients over N steps to
                       emulate the large batch (effective_batch = micro_batch*N).
4. Free memory         Call `empty_cache()` at the end of an epoch/experiment.
                       Delete big tensors and let the Python GC run:
                       `del x; empty_cache()`.
5. Avoid OOM           * Never keep two full copies of a dataset on the GPU.
                       * Prefer float32 -> float16 only where safe; keep the
                         loss/optimizer in float32 when using AMP.
                       * Use `torch.inference_mode()` (not just `no_grad`) for
                         evaluation/inference to cut memory further.
                       * Reduce sequence length / image resolution before you
                         reduce model quality.
                       * Set `torch.cuda.set_per_process_memory_fraction(0.9)`
                         to fail fast instead of thrashing.
6. Persist the runtime Colab T4 sessions time out and the VM is wiped on
                       disconnect. Write checkpoints/artefacts to Drive
                       (`from google.colab import drive; drive.mount('/content/drive')`)
                       or re-run the idempotent pipeline from `src/`.
7. CPU fallback is safe Every cell in this capstone runs on CPU too - the device
                       is chosen once at the top and threads through the code.
--------------------------------------------------------------------------"""
    if verbose:
        print(tips)
    return tips


# --------------------------------------------------------------------------- #
# 7. One-call convenience
# --------------------------------------------------------------------------- #
def setup_gpu(requirements: str = "requirements.txt", verbose: bool = True) -> Dict[str, object]:
    """Run the full Colab GPU setup sequence and return the GPU report.

    Order: Colab-guarded install -> detect GPU/print device report -> print the
    16 GB T4 memory tips.  Returns the same dict as :func:`detect_gpu`, with an
    extra ``"batch_size"`` suggestion.
    """
    colab_install(requirements)
    report = detect_gpu(verbose=verbose)
    report["batch_size"] = recommend_batch_size()
    if verbose:
        print("  Suggested starting batch size:", report["batch_size"])
        t4_memory_tips(verbose=True)
    return report


# --------------------------------------------------------------------------- #
# 8. Device-agnostic torch smoke test
# --------------------------------------------------------------------------- #
def torch_smoke_test(verbose: bool = True) -> Dict[str, object]:
    """Prove the torch code is device-agnostic (``.to(device)`` + mixed precision).

    Builds two small tensors **on the selected device**, runs a matrix multiply
    under :func:`autocast_context` when a GPU is present, and reports the result.
    Returns a small report dict; never raises (returns ``{"torch": False}`` when
    PyTorch is missing).
    """
    try:
        import torch
    except Exception as exc:
        if verbose:
            print("[smoke] torch not installed ->", type(exc).__name__, "(CPU path unaffected)")
        return {"torch": False, "device": "cpu"}

    device = get_device()
    dev_type = getattr(device, "type", str(device))
    if verbose:
        print("[smoke] selected device      :", device, "(", device_name(), ")")

    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    if verbose:
        print("[smoke] tensor device/dtype  :", a.device, "/", a.dtype)

    with autocast_context(device):
        c = a @ b                      # float16 on GPU, float32 on CPU - same call
    if verbose:
        print("[smoke] matmul OK, result    :", tuple(c.shape),
              "dtype:", c.dtype, "mean: %.5f" % float(c.mean()))
        vram_snapshot("after smoke test")

    empty_cache()
    return {"torch": True, "device": dev_type, "result_dtype": str(c.dtype),
            "shape": tuple(c.shape)}


# --------------------------------------------------------------------------- #
# 9. Device-agnostic torch building blocks (used by the notebooks)
# --------------------------------------------------------------------------- #
def torch_embed_texts(texts, dim: int = 256, verbose: bool = True):
    """Deterministic, **device-agnostic** text embedding via a torch hashing trick.

    Runs on the GPU when one is present, otherwise on the CPU, and returns an
    ``(n, dim)`` float32 NumPy array (L2-normalised).  This is the torch analogue
    of the offline TF-IDF/SVD embedder and shows the ``.to(device)`` pattern on a
    real tensor workload.  Returns ``None`` if torch is unavailable.
    """
    try:
        import torch
    except Exception as exc:
        if verbose:
            print("[torch_embed] torch unavailable ->", type(exc).__name__, "(skipping)")
        return None

    import hashlib

    device = get_device()
    n = len(texts)
    mat = torch.zeros((n, dim), dtype=torch.float32, device=device)
    for i, t in enumerate(texts):
        for tok in str(t).lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            mat[i, h % dim] += 1.0 if (h >> 8) % 2 == 0 else -1.0
    mat = mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    out = mat.detach().cpu().numpy().astype("float32")
    if verbose:
        print("[torch_embed] %d texts -> %s on device %s" % (n, tuple(out.shape), device))
    empty_cache()
    return out


def torch_mlp_classifier(X_train, y_train, X_test, y_test,
                         hidden: int = 64, epochs: int = 60, lr: float = 1e-3,
                         batch_size: Optional[int] = None, verbose: bool = True):
    """A small **device-agnostic** MLP classifier in PyTorch.

    Demonstrates the full GPU-friendly pattern the brief asks for:
    ``torch.device`` selection, ``.to(device)``, mini-batch training, mixed
    precision (``autocast`` + ``GradScaler``), ``torch.inference_mode()`` for
    evaluation and ``empty_cache()`` afterwards.  Returns a metrics dict, or
    ``None`` when torch is unavailable (the sklearn models remain the baseline).
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        if verbose:
            print("[torch_mlp] torch unavailable ->", type(exc).__name__, "(skipping)")
        return None

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    device = get_device()
    bs = batch_size or recommend_batch_size()

    Xtr = torch.tensor(np.asarray(X_train, dtype="float32"), device=device)
    ytr = torch.tensor(np.asarray(y_train, dtype="float32").reshape(-1, 1), device=device)
    Xte = torch.tensor(np.asarray(X_test, dtype="float32"), device=device)
    yte = np.asarray(y_test).reshape(-1)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, max(8, hidden // 2)), nn.ReLU(),
        nn.Linear(max(8, hidden // 2), 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    scaler = amp_grad_scaler(enabled=True)

    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            with autocast_context(device):
                loss = lossf(model(Xtr[idx]), ytr[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    model.eval()
    with torch.inference_mode():
        proba = torch.sigmoid(model(Xte)).detach().cpu().numpy().reshape(-1)
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "device": str(device),
        "batch_size": bs,
        "epochs": epochs,
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
    }
    if verbose:
        print("[torch_mlp] device=%s batch=%d -> acc=%.4f f1=%.4f roc_auc=%.4f"
              % (device, bs, metrics["accuracy"], metrics["f1"], metrics["roc_auc"]))
        vram_snapshot("after MLP")
    empty_cache()
    return metrics


if __name__ == "__main__":  # pragma: no cover
    setup_gpu()
    torch_smoke_test()
