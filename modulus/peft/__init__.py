from .lora import LoRAConfig, apply_lora_grad_hook, orth_lora_grad_jax

try:
    from .lora import LoRADense
except Exception:
    LoRADense = None  # Optional when flax is not installed

__all__ = [
    "LoRAConfig",
    "LoRADense",
    "orth_lora_grad_jax",
    "apply_lora_grad_hook",
]
