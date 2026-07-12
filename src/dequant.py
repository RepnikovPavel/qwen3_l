"""
Dequantize FP8 (e4m3, block-wise) weights to bf16 so the model can run on CPU.

Qwen3-4B-Thinking-2507-FP8 stores weights as ``float8_e4m3fn`` with block-wise
scales of [128, 128] (see config.json → quantization_config.weight_block_size).
The Triton FP8 kernel only runs on CUDA, so for CPU inference we walk the model
and replace every ``FP8Linear`` with a plain ``nn.Linear`` whose ``weight`` is
the dequantized bf16 tensor: ``W_bf16 = W_fp8 * scale_block``.

The dequantization is exact (no approximation): FP8 block-quantized values are
reconstructed by multiplying each 128×128 block by its float32 scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Block size from config.json: quantization_config.weight_block_size
DEFAULT_BLOCK = [128, 128]


def _dequantize_fp8_weight(weight: torch.Tensor, scale_inv: torch.Tensor,
                           block_size=DEFAULT_BLOCK) -> torch.Tensor:
    """Dequantize an FP8 weight tensor to bf16 using block-wise scales.

    Args:
        weight: (out_features, in_features) float8_e4m3fn tensor.
        scale_inv: (out//block_n, in//block_k) float32 per-block scales.
        block_size: [block_n, block_k] quantization block (default [128,128]).

    Returns:
        (out_features, in_features) bfloat16 dequantized weight.
    """
    out_features, in_features = weight.shape
    block_n, block_k = block_size

    # upcast FP8 → float32, then reshape into blocks.
    w = weight.to(torch.float32)
    # Pad to a multiple of block size (Qwen3 dims are already multiples of 128).
    pad_n = (block_n - out_features % block_n) % block_n
    pad_k = (block_k - in_features % block_k) % block_k
    if pad_n or pad_k:
        w = torch.nn.functional.pad(w, (0, pad_k, 0, pad_n))
    n_blocks_n = w.shape[0] // block_n
    n_blocks_k = w.shape[1] // block_k

    # (n_blocks_n, block_n, n_blocks_k, block_k)
    blocked = w.view(n_blocks_n, block_n, n_blocks_k, block_k)
    # scale_inv: (n_blocks_n, n_blocks_k) → broadcast over each block.
    scales = scale_inv.to(torch.float32)
    # Crop scales if padding added block rows/cols that shouldn't exist.
    scales = scales[:n_blocks_n, :n_blocks_k]
    dequant_blocked = blocked * scales.unsqueeze(1).unsqueeze(-1)

    dequant = dequant_blocked.view(w.shape[0], w.shape[1])
    # Remove padding.
    dequant = dequant[:out_features, :in_features]
    return dequant.to(torch.bfloat16)


def _make_linear_from_fp8(fp8_linear: nn.Module, block_size=DEFAULT_BLOCK) -> nn.Linear:
    """Build a plain nn.Linear with dequantized bf16 weights from an FP8Linear.

    Preserves in/out features and bias (if any).
    """
    in_features = fp8_linear.in_features
    out_features = fp8_linear.out_features

    linear = nn.Linear(
        in_features,
        out_features,
        bias=fp8_linear.has_bias,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        linear.weight.copy_(
            _dequantize_fp8_weight(fp8_linear.weight.data, fp8_linear.weight_scale_inv.data, block_size)
        )
        if fp8_linear.has_bias and fp8_linear.bias is not None:
            linear.bias.copy_(fp8_linear.bias.data.to(torch.bfloat16))
    return linear


def dequantize_model(model: nn.Module, block_size=DEFAULT_BLOCK) -> nn.Module:
    """Replace every FP8Linear in ``model`` with a dequantized bf16 nn.Linear.

    Walks the module tree in place. After this call the model contains no FP8
    layers and can run on CPU (or any device) without the FP8 Triton kernel.
    """
    # Import lazily — only present in transformers with FP8 support.
    from transformers.integrations.finegrained_fp8 import FP8Linear  # noqa: PLC0415

    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, FP8Linear):
                new_linear = _make_linear_from_fp8(child, block_size)
                new_linear = new_linear.to(next(model.parameters()).dtype
                                           if any(True for _ in model.parameters()) else torch.bfloat16)
                setattr(module, child_name, new_linear)
                replaced += 1
    return replaced


def model_has_fp8(model: nn.Module) -> bool:
    """True if ``model`` still contains any FP8Linear layers."""
    from transformers.integrations.finegrained_fp8 import FP8Linear  # noqa: PLC0415

    return any(isinstance(m, FP8Linear) for m in model.modules())
