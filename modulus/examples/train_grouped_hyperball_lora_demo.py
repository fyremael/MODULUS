"""
modulus.examples.train_grouped_hyperball_lora_demo

End-to-end example demonstrating:
1) Hyperball integrated into a MODULUS-style harness
2) Parameter-group configs: attention vs MLP different target angles
3) LoRA tangent-steering: orthogonalize LoRA factor gradients

Run:
  python -m modulus.examples.train_grouped_hyperball_lora_demo

Requires: flax
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from modulus.optim.groups import make_grouped_hyperball_tx, make_llm_default_labels
from modulus.optim.masks import default_llm_hyperball_mask
from modulus.optim.schedules import WarmupCosine
from modulus.peft.lora import LoRAConfig, LoRADense
from modulus.training.harness import make_train_step


@dataclass(frozen=True)
class Config:
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    seq_len: int = 64
    batch_size: int = 32
    steps: int = 200

    lr: float = 1e-3
    wd: float = 0.01

    # Hyperball angles
    alpha_attn_peak: float = 0.04
    alpha_mlp_peak: float = 0.06
    warmup_steps: int = 20
    total_steps: int = 200

    # LoRA
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 8.0


class TinyBlock(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        h = nn.LayerNorm(use_bias=False)(x)

        # Attention path
        h = nn.SelfAttention(
            num_heads=self.cfg.n_heads,
            qkv_features=self.cfg.d_model,
            out_features=self.cfg.d_model,
            deterministic=deterministic,
            use_bias=False,
            name="attn",
        )(h)
        x = x + h

        # MLP path with optional LoRA
        h = nn.LayerNorm(use_bias=False)(x)
        if self.cfg.use_lora:
            lora = LoRAConfig(rank=self.cfg.lora_rank, alpha=self.cfg.lora_alpha, dropout=0.0)
            h = LoRADense(self.cfg.d_model * 4, use_bias=False, lora=lora, name="mlp_fc1")(
                h, deterministic=deterministic
            )
            h = nn.gelu(h)
            h = LoRADense(self.cfg.d_model, use_bias=False, lora=lora, name="mlp_fc2")(
                h, deterministic=deterministic
            )
        else:
            h = nn.Dense(self.cfg.d_model * 4, use_bias=False, name="mlp_fc1")(h)
            h = nn.gelu(h)
            h = nn.Dense(self.cfg.d_model, use_bias=False, name="mlp_fc2")(h)

        return x + h


class TinyLM(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, tokens, deterministic: bool = True):
        x = nn.Embed(self.cfg.vocab, self.cfg.d_model, name="embed")(tokens)
        for i in range(self.cfg.n_layers):
            x = TinyBlock(self.cfg, name=f"block_{i}")(x, deterministic=deterministic)
        x = nn.LayerNorm(use_bias=False)(x)
        logits = nn.Dense(self.cfg.vocab, use_bias=False, name="lm_head")(x)
        return logits


def make_batch(rng, cfg: Config) -> Tuple[jnp.ndarray, jnp.ndarray]:
    tokens = jax.random.randint(rng, (cfg.batch_size, cfg.seq_len), 0, cfg.vocab)
    targets = jnp.roll(tokens, shift=-1, axis=1)
    return tokens, targets


def apply_fn(params, tokens):
    # Bound later via closure in train_step for speed (flax apply needs module)
    raise NotImplementedError


def loss_fn(apply_fn, params, batch):
    tokens, targets = batch
    logits = apply_fn(params, tokens)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    loss = jnp.mean(nll)
    return loss, logits


def main():
    cfg = Config()
    rng = jax.random.PRNGKey(0)

    model = TinyLM(cfg)
    dummy = jnp.zeros((cfg.batch_size, cfg.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy, deterministic=True)["params"]

    # Parameter labels (attn vs mlp vs others)
    labels_fn = make_llm_default_labels()

    # Base optimizer per group (could differ)
    base_attn = optax.adamw(cfg.lr, weight_decay=cfg.wd)
    base_mlp = optax.adamw(cfg.lr, weight_decay=cfg.wd)
    base_other = optax.adamw(cfg.lr, weight_decay=cfg.wd)

    # Angle schedules per group
    attn_sched = WarmupCosine(
        cfg.warmup_steps, cfg.total_steps, cfg.alpha_attn_peak, cfg.alpha_attn_peak * 0.2
    )
    mlp_sched = WarmupCosine(
        cfg.warmup_steps, cfg.total_steps, cfg.alpha_mlp_peak, cfg.alpha_mlp_peak * 0.2
    )

    # Hyperball settings: row-sphere, tangent projection
    hb_common = dict(
        radius=1.0,
        mode="sphere",
        proj_tangent=True,
        granularity="row",
        update_norm_clip=(None, None),
        emit_metrics=True,
    )

    # Mask within each group is not needed here since multi_transform does group masking,
    # but we *still* want to exclude LN/bias inside each group. We'll use default mask callable.
    mask_fn = default_llm_hyperball_mask(
        include_embeddings=False, exclude_lora=True, exclude_1d=True
    )

    hb_kwargs_by_group = {
        "attn": dict(**hb_common, target_angle=attn_sched, mask=mask_fn),
        "mlp": dict(**hb_common, target_angle=mlp_sched, mask=mask_fn),
        "other": {},  # fall back to base optimizer only
        "embed": {},  # you can enable Hyperball here as an ablation
        "norm": {},  # no constraint
        "bias": {},  # no constraint
    }

    base_by_group = {
        "attn": base_attn,
        "mlp": base_mlp,
        "other": base_other,
        "embed": base_other,
        "norm": base_other,
        "bias": base_other,
    }

    # Build grouped tx
    tx_builder = make_grouped_hyperball_tx(
        base_by_group=base_by_group,
        hyperball_kwargs_by_group=hb_kwargs_by_group,
        labels_fn=labels_fn,
        default_group="other",
    )
    tx = tx_builder(params)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Wrap apply_fn so harness can call it with only params,tokens
    def apply_only(params, tokens):
        return model.apply({"params": params}, tokens, deterministic=True)

    train_step = make_train_step(
        apply_fn=apply_only,
        loss_fn=loss_fn,
        use_lora_grad_hook=cfg.use_lora,
        lora_hook_kwargs={"a_name": "lora_A", "b_name": "lora_B", "eps": 1e-6},
    )

    for step in range(cfg.steps):
        rng, brng = jax.random.split(rng)
        batch = make_batch(brng, cfg)
        state, metrics = train_step(state, batch)

        if step % 20 == 0:
            msg = f"step={step:04d} loss={metrics['loss']:.4f}"
            # Hyperball metrics may be present from groups (state.opt_state is MultiTransformState)
            # We won't always have keys, but print if present.
            angle_key = "hyperball/angle_mean"
            if angle_key in metrics:
                msg += f" angle={metrics[angle_key]:.4f}"
            print(msg)

    print("Done.")


if __name__ == "__main__":
    main()
