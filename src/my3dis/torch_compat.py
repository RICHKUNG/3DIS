"""Compatibility shims for running SAM2 on older PyTorch versions.

This module patches missing features at runtime without touching external
dependencies such as the SAM2 repository.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

_PATCHED = False


def patch_torch_compat() -> None:
    """Apply runtime patches for torch APIs used by SAM2."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _patch_torch_load_weights_only()
    _patch_scaled_dot_product_attention()


def _patch_torch_load_weights_only() -> None:
    """Allow torch.load(..., weights_only=True) on older PyTorch releases."""
    try:
        import inspect

        if "weights_only" in inspect.signature(torch.load).parameters:
            return
    except (ValueError, TypeError):
        # If we cannot introspect, fall back to installing the shim.
        pass

    original_load = torch.load
    if getattr(original_load, "__my3dis_weights_only_shim__", False):
        return

    def load_with_optional_weights_only(*args: Any, **kwargs: Any):
        kwargs.pop("weights_only", None)
        return original_load(*args, **kwargs)

    load_with_optional_weights_only.__my3dis_weights_only_shim__ = True  # type: ignore[attr-defined]
    torch.load = load_with_optional_weights_only  # type: ignore[assignment]


def _patch_scaled_dot_product_attention() -> None:
    """Provide a functional scaled-dot-product attention fallback when missing."""
    if hasattr(F, "scaled_dot_product_attention"):
        return

    def _scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(
                    (q.size(-2), k.size(-2)),
                    device=q.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        if dropout_p:
            weights = F.dropout(weights, p=dropout_p, training=True)

        return torch.matmul(weights, v)

    F.scaled_dot_product_attention = _scaled_dot_product_attention  # type: ignore[attr-defined]

