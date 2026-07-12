"""
Load the local finegrained-fp8 Triton kernel from disk and inject it into the
transformers kernel cache so FP8 weights run without network access.

Qwen3-4B-Thinking-2507-FP8 stores W8A8 FP8 weights. At inference, transformers
lazily fetches the matching Triton kernel ("finegrained-fp8") from the HF Hub.
To run fully offline we load the vendored kernel sources from a local directory
and register them under the same cache key transformers looks up, so the Hub is
never contacted.

This mirrors the recipe from the original dot.mocr demo: the kernel folder is
loaded as a Python module and placed in
``transformers.integrations.hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"]``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

__all__ = ["load_local_fp8_kernel", "inject_fp8_kernel"]

# Resolve the default kernel directory once: env override first, then the
# checked-in copy shipped with this repo (src/ -> kernels/...).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KERNEL_DIR = _REPO_ROOT / "kernels" / "local_kernels_qwen3_8B_FP8"


def load_local_fp8_kernel(kernel_dir: str | os.PathLike | None = None):
    """Load the kernel folder at ``kernel_dir`` as a Python module.

    The folder is expected to contain an ``__init__.py`` that imports the
    matmul / batched / grouped / act_quant submodules. Registering it in
    ``sys.modules`` first lets those internal relative imports resolve.

    Args:
        kernel_dir: Path to the local kernel folder. If ``None``, falls back to
            the ``QWEN3_FP8_KERNEL_DIR`` env var, then to the vendored copy.

    Returns:
        The loaded kernel module.
    """
    kernel_dir = Path(kernel_dir or os.environ.get("QWEN3_FP8_KERNEL_DIR") or _DEFAULT_KERNEL_DIR)
    if not kernel_dir.exists():
        raise FileNotFoundError(
            f"Local FP8 kernel not found at: {kernel_dir}\n"
            f"Set QWEN3_FP8_KERNEL_DIR or run downloader/download_kernels.sh."
        )

    init_file = kernel_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "finegrained_fp8_local",
        init_file,
        submodule_search_locations=[str(kernel_dir)],
    )
    module = importlib.util.module_from_spec(spec)

    # Register before exec so the kernel's own internal imports resolve.
    sys.modules["finegrained_fp8_local"] = module
    # Running __init__.py pulls in matmul.py, batched.py, etc.
    spec.loader.exec_module(module)
    return module


def inject_fp8_kernel(kernel_dir: str | os.PathLike | None = None):
    """Load the local FP8 kernel and register it in the transformers cache.

    After this call, when transformers' ``lazy_load_kernel`` asks for the
    ``"finegrained-fp8"`` kernel, it finds it pre-populated and returns our
    local module instead of hitting the Hub.

    Must be called BEFORE importing the model / ``AutoModelForCausalLM``.
    """
    import transformers.integrations.hub_kernels as hub_kernels  # noqa: PLC0415

    local_fp8_kernel = load_local_fp8_kernel(kernel_dir)
    hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"] = local_fp8_kernel
    return local_fp8_kernel
