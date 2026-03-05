"""
modulus.training.harness

A compact, JAX-first training harness that is:
- JIT-friendly and pmap-ready
- compatible with Hyperball metrics extraction
- supports ablations via config switches
- supports LoRA gradient hooks (orthogonalization / tangent steering)

This is intentionally framework-light:
- If `flax` is available, use flax.training.train_state
- Otherwise, provides a simple TrainState replacement.

We prefer explicitness over magic: everything is functional and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from modulus.optim.hyperball import get_last_metrics as hyperball_last_metrics
from modulus.peft.lora import apply_lora_grad_hook

PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass
class SimpleTrainState:
    """
    A minimal TrainState for environments without flax.

    Fields:
      - params: model parameters pytree
      - tx: optax transformation
      - opt_state: optimizer state
      - step: global step counter
    """

    params: PyTree
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    step: jnp.ndarray

    @classmethod
    def create(cls, *, params: PyTree, tx: optax.GradientTransformation):
        return cls(
            params=params, tx=tx, opt_state=tx.init(params), step=jnp.asarray(0, dtype=jnp.int32)
        )

    def apply_gradients(self, *, grads: PyTree) -> "SimpleTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return SimpleTrainState(
            params=new_params, tx=self.tx, opt_state=new_opt_state, step=self.step + 1
        )

    def tree_flatten(self):
        # Optimizer transform itself is static Python state; dynamic leaves are arrays/pytrees.
        children = (self.params, self.opt_state, self.step)
        aux_data = self.tx
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, opt_state, step = children
        return cls(params=params, tx=aux_data, opt_state=opt_state, step=step)


def default_loss_and_logits(
    apply_fn: Callable, params: PyTree, batch: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Default next-token prediction loss helper.
    Expects:
      apply_fn(params, tokens) -> logits [B,T,V]
      batch = (tokens, targets)
    """
    tokens, targets = batch
    logits = apply_fn(params, tokens)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    loss = jnp.mean(nll)
    return loss, logits


def make_train_step(
    *,
    apply_fn: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    loss_fn: Callable[[Callable, PyTree, Any], Tuple[jnp.ndarray, Any]] = default_loss_and_logits,
    use_lora_grad_hook: bool = False,
    lora_hook_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[Any, Any], Tuple[Any, Dict[str, jnp.ndarray]]]:
    """
    Build a JIT-able train_step(state, batch) -> (new_state, metrics).

    `state` can be flax TrainState or our SimpleTrainState as long as it has:
      - params
      - apply_gradients(grads=...)
      - opt_state
    """
    lora_hook_kwargs = lora_hook_kwargs or {}

    def step_fn(state, batch):
        def _loss(params):
            loss, aux = loss_fn(apply_fn, params, batch)
            return loss, aux

        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)

        if use_lora_grad_hook:
            grads = apply_lora_grad_hook(state.params, grads, **lora_hook_kwargs)

        new_state = state.apply_gradients(grads=grads)

        metrics: Dict[str, jnp.ndarray] = {"loss": loss}

        # Hyperball emits metrics into optimizer state if present
        try:
            mets = hyperball_last_metrics(new_state.opt_state)
            for k, v in mets.items():
                metrics[k] = jnp.asarray(v)
        except Exception:
            pass

        return new_state, metrics

    return jax.jit(step_fn)
