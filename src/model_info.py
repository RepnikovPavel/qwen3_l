"""KV-cache budget math + dtype helpers, shared by inference/demo/manager.

Kept in src/ (not demo/) so both the benchmark and the web demo use the same
context-accounting logic.
"""
from __future__ import annotations


class ModelInfo:
    """Static model facts used for the context budget computation."""

    def __init__(self, num_layers, num_kv_heads, head_dim,
                 max_position_embeddings, dtype_bytes=2, num_experts=0,
                 moe_intermediate_size=0, hidden_size=0, is_moe=False):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.dtype_bytes = dtype_bytes
        # MoE bookkeeping (zero for dense models).
        self.num_experts = num_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_size = hidden_size
        self.is_moe = is_moe

    @classmethod
    def from_model(cls, model) -> "ModelInfo":
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = cfg.hidden_size // cfg.num_attention_heads
        dtype = next(model.parameters()).dtype
        dtype_bytes = 2 if dtype.itemsize <= 2 else 4
        is_moe = cfg.model_type.endswith("_moe") or hasattr(cfg, "num_experts")
        return cls(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=getattr(cfg, "max_position_embeddings", 32768),
            dtype_bytes=dtype_bytes,
            num_experts=getattr(cfg, "num_experts", 0),
            moe_intermediate_size=getattr(cfg, "moe_intermediate_size", 0),
            hidden_size=cfg.hidden_size,
            is_moe=is_moe,
        )

    @property
    def bytes_per_token(self) -> int:
        """KV-cache bytes per token: K+V across all layers & KV heads.

        For MoE this is the SAME as dense — only attention allocates a KV cache;
        expert FFNs are stateless (params only, reused per token).
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def max_kv_bytes(self) -> int:
        return self.bytes_per_token * self.max_position_embeddings
