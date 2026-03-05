"""
modulus.optim.hyperball

A robust, Optax-compatible optimizer wrapper that enforces hyperspherical
(or hyperball-like) dynamics on selected parameter leaves.

Core idea:
- Base optimizer proposes an update u
- Hyperball projects to tangent space (optional)
- Scales update by target angular step or target update norm (optional)
- Applies retraction (normalize) to keep norms fixed (sphere) or clamped (ball)

This module is designed for nGPT / MODULUS-style training:
- Direction learning is primary (rotation on spheres)
- Norm drift is controlled/eliminated
- Angular learning rate becomes an explicit knob
- Compatible with arbitrary base optimizers and parameter trees

Authoring style: production-grade, ablatable, heavily documented.

Dependencies:
- jax
- optax
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax

PyTree = Any
Schedule = Union[float, Callable[[jnp.ndarray], jnp.ndarray], Callable[[int], float]]

Granularity = Literal["leaf", "row", "col", "channel"]
Mode = Literal["sphere", "ball"]

# -----------------------------
# Utilities: schedules & typing
# -----------------------------


def _as_schedule(x: Optional[Schedule]) -> Optional[Callable[[jnp.ndarray], jnp.ndarray]]:
    """Normalize a schedule-like input into a function f(step)->scalar (jnp)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        const = float(x)

        def f(step: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(const, dtype=jnp.float32)

        return f
    if callable(x):
        # Accept either step: int or step: jnp.ndarray
        def f(step: jnp.ndarray) -> jnp.ndarray:
            try:
                return jnp.asarray(x(step), dtype=jnp.float32)
            except TypeError:
                # Might be int-based
                return jnp.asarray(x(int(step)), dtype=jnp.float32)

        return f
    raise TypeError(f"Unsupported schedule type: {type(x)}")


def _tree_zeros_like(tree: PyTree) -> PyTree:
    return jax.tree.map(lambda x: jnp.zeros_like(x), tree)


def _is_array(x: Any) -> bool:
    return isinstance(x, jnp.ndarray)


# ---------------------------------------
# Norm axes selection (granularity policy)
# ---------------------------------------


def default_norm_axes_for_leaf(x: jnp.ndarray, granularity: Granularity) -> Tuple[int, ...]:
    """
    Choose reduction axes for per-group norms within a leaf.

    Given a leaf tensor x with shape:
      - leaf: norm over all elements  -> axes = all axes
      - row: per-row groups           -> axes = (1..ndim-1) for ndim>=2
      - col: per-col groups           -> axes = (0,) for matrices; more generally (0..ndim-2)
      - channel: per-output channel   -> axes = (1..ndim-1), same as row for conv-style shapes

    Important: these axes are used with keepdims=True so that broadcasting works.
    """
    ndim = x.ndim
    if granularity == "leaf":
        return tuple(range(ndim))
    if granularity in ("row", "channel"):
        if ndim <= 1:
            return tuple(range(ndim))
        return tuple(range(1, ndim))
    if granularity == "col":
        if ndim <= 1:
            return tuple(range(ndim))
        # For (d_out, d_in), per-column norm reduces axis 0 (keep axis 1).
        # For general tensors, keep last axis and reduce everything else.
        return tuple(range(0, ndim - 1))
    raise ValueError(f"Unknown granularity: {granularity}")


AxesFn = Callable[[Tuple[Any, ...], jnp.ndarray], Tuple[int, ...]]


def make_axes_fn(granularity: Granularity = "row") -> AxesFn:
    """
    Build an axes function that can be path-aware in the future.

    Signature: axes_fn(path, leaf) -> reduction_axes
    """

    def axes_fn(path: Tuple[Any, ...], leaf: jnp.ndarray) -> Tuple[int, ...]:
        return default_norm_axes_for_leaf(leaf, granularity)

    return axes_fn


# -----------------------
# Core geometric operators
# -----------------------


def group_dot(
    u: jnp.ndarray, w: jnp.ndarray, axes: Tuple[int, ...], *, dtype_accum=jnp.float32
) -> jnp.ndarray:
    """Compute group-wise dot product ⟨u, w⟩ reduced over axes, keeping dims."""
    return jnp.sum(u * w, axis=axes, keepdims=True, dtype=dtype_accum)


def group_norm(
    x: jnp.ndarray, axes: Tuple[int, ...], eps: float, *, dtype_accum=jnp.float32
) -> jnp.ndarray:
    """Compute group-wise ||x|| with keepdims=True."""
    return jnp.sqrt(jnp.sum(x * x, axis=axes, keepdims=True, dtype=dtype_accum) + eps)


def project_tangent(
    u: jnp.ndarray, w: jnp.ndarray, axes: Tuple[int, ...], eps: float, *, dtype_accum=jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Decompose u into tangent + radial components w.r.t. w:
      u_par = ⟨u,w⟩/||w||^2 * w
      u_perp = u - u_par
    """
    dot = group_dot(u, w, axes, dtype_accum=dtype_accum)
    w2 = jnp.sum(w * w, axis=axes, keepdims=True, dtype=dtype_accum) + eps
    u_par = (dot / w2) * w
    u_perp = u - u_par
    return u_perp, u_par


def retract_to_radius(
    x: jnp.ndarray,
    axes: Tuple[int, ...],
    radius: jnp.ndarray,
    eps: float,
    *,
    dtype_accum=jnp.float32,
) -> jnp.ndarray:
    """
    Normalize x to have group norm == radius.
    radius must be broadcastable to x with keepdims=True shape.
    """
    n = group_norm(x, axes, eps, dtype_accum=dtype_accum)
    return (radius * x) / n


def clip_by_group_norm(
    u: jnp.ndarray,
    axes: Tuple[int, ...],
    eps: float,
    min_norm: Optional[float],
    max_norm: Optional[float],
    *,
    dtype_accum=jnp.float32,
) -> jnp.ndarray:
    """
    Clip group-wise norm of u into [min_norm, max_norm].
    If min_norm is set, norms smaller than min_norm are scaled up.
    If max_norm is set, norms larger than max_norm are scaled down.
    """
    if min_norm is None and max_norm is None:
        return u

    n = group_norm(u, axes, eps, dtype_accum=dtype_accum)
    scale = jnp.ones_like(n)

    if min_norm is not None:
        mn = jnp.asarray(float(min_norm), dtype=jnp.float32)
        scale = jnp.where(n < mn, mn / (n + eps), scale)

    if max_norm is not None:
        mx = jnp.asarray(float(max_norm), dtype=jnp.float32)
        scale = jnp.where(n > mx, mx / (n + eps), scale)

    return u * scale


def clamp_group_norm(
    x: jnp.ndarray,
    axes: Tuple[int, ...],
    eps: float,
    min_norm: Optional[float],
    max_norm: Optional[float],
    *,
    dtype_accum=jnp.float32,
) -> jnp.ndarray:
    """
    Clamp group norms of x by scaling x so that ||x|| is within [min_norm, max_norm].
    """
    if min_norm is None and max_norm is None:
        return x

    n = group_norm(x, axes, eps, dtype_accum=dtype_accum)
    scale = jnp.ones_like(n)

    if min_norm is not None:
        mn = jnp.asarray(float(min_norm), dtype=jnp.float32)
        scale = jnp.where(n < mn, mn / (n + eps), scale)

    if max_norm is not None:
        mx = jnp.asarray(float(max_norm), dtype=jnp.float32)
        scale = jnp.where(n > mx, mx / (n + eps), scale)

    return x * scale


# -----------------------
# Hyperball configuration
# -----------------------


@dataclass(frozen=True)
class HyperballConfig:
    """Configuration for the Hyperball wrapper."""

    radius: float = 1.0
    mode: Mode = "sphere"
    proj_tangent: bool = True

    granularity: Granularity = "row"
    axes_fn: Optional[AxesFn] = None

    # Step control: either target_angle OR target_update_norm.
    target_angle: Optional[Schedule] = None  # radians
    target_update_norm: Optional[Schedule] = None
    update_norm_clip: Tuple[Optional[float], Optional[float]] = (None, None)

    # Ball-mode extras (optional):
    radial_decay: Optional[float] = None  # effective decay coefficient
    radial_lr_scale: float = 1.0
    ball_norm_clamp: Tuple[Optional[float], Optional[float]] = (
        None,
        None,
    )  # clamp group norms of params

    # Masking:
    mask: Optional[Union[PyTree, Callable[[PyTree], PyTree]]] = None

    # Numerics:
    eps: float = 1e-8
    dtype_accum: jnp.dtype = jnp.float32

    # Logging:
    emit_metrics: bool = True
    metrics_prefix: str = "hyperball"


@jax.tree_util.register_pytree_node_class
@dataclass
class HyperballState:
    """Optax state: wraps base optimizer state + internal step counter + last metrics."""

    base_state: Any
    count: jnp.ndarray
    last_metrics: Mapping[str, jnp.ndarray]

    def tree_flatten(self):
        children = (self.base_state, self.count, self.last_metrics)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        base_state, count, last_metrics = children
        return cls(base_state=base_state, count=count, last_metrics=last_metrics)


def _init_metrics_dict(prefix: str) -> Dict[str, jnp.ndarray]:
    # Scalars (0d) for stable state structure
    z = jnp.asarray(0.0, dtype=jnp.float32)
    return {
        f"{prefix}/w_norm_mean": z,
        f"{prefix}/u_norm_mean": z,
        f"{prefix}/u_perp_norm_mean": z,
        f"{prefix}/angle_mean": z,
        f"{prefix}/radial_frac_mean": z,
        f"{prefix}/masked_frac": z,
    }


# -------------------------
# Mask handling (tree-wise)
# -------------------------


def _resolve_mask(
    mask: Optional[Union[PyTree, Callable[[PyTree], PyTree]]], params: PyTree
) -> PyTree:
    """
    Resolve a mask specification into a pytree of booleans matching params.
    If mask is None -> all True.
    If callable -> mask(params)
    Else -> assumed to be pytree already.
    """
    if mask is None:
        return jax.tree.map(lambda _: True, params)
    if callable(mask):
        m = mask(params)
        return m
    return mask


# -------------------------
# Leaf update implementation
# -------------------------


def _hyperball_leaf(
    u: jnp.ndarray,
    w: jnp.ndarray,
    apply_mask: bool,
    step: jnp.ndarray,
    cfg: HyperballConfig,
    axes_fn: AxesFn,
    angle_sched: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    norm_sched: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    radius_sched: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Transform a single-leaf update vector u into a Hyperball-consistent update.
    Returns (u_out, leaf_metrics).
    """
    if (not apply_mask) or (not _is_array(u)) or (not _is_array(w)):
        return u, {}

    axes = axes_fn((), w)

    # Keep everything in float32 for geometric computations, then cast back.
    u_f = u.astype(jnp.float32)
    w_f = w.astype(jnp.float32)

    # Tangent projection (optional)
    if cfg.proj_tangent:
        u_perp, u_par = project_tangent(u_f, w_f, axes, cfg.eps, dtype_accum=cfg.dtype_accum)
    else:
        u_perp, u_par = u_f, jnp.zeros_like(u_f)

    # Step control: choose scaling rule
    if angle_sched is not None and norm_sched is not None:
        # In practice, we allow both but strongly discourage; default is "angle wins".
        pass

    if angle_sched is not None:
        alpha = angle_sched(step)  # radians
        w_norm = group_norm(w_f, axes, cfg.eps, dtype_accum=cfg.dtype_accum)
        desired = alpha * w_norm
        u_norm = group_norm(u_perp, axes, cfg.eps, dtype_accum=cfg.dtype_accum)
        u_perp = u_perp * (desired / (u_norm + cfg.eps))
    elif norm_sched is not None:
        desired = norm_sched(step)  # absolute update-norm target
        u_norm = group_norm(u_perp, axes, cfg.eps, dtype_accum=cfg.dtype_accum)
        u_perp = u_perp * (desired / (u_norm + cfg.eps))

    # Optional update-norm clipping
    clip_min, clip_max = cfg.update_norm_clip
    u_perp = clip_by_group_norm(
        u_perp, axes, cfg.eps, clip_min, clip_max, dtype_accum=cfg.dtype_accum
    )

    # Ball-mode radial term (optional): add shrink on w to build controlled radius attractor
    # Note: This is applied in update-space and then optionally clamped.
    if cfg.mode == "ball" and cfg.radial_decay is not None and cfg.radial_decay > 0.0:
        lam = jnp.asarray(float(cfg.radial_decay), dtype=jnp.float32)
        radial = -cfg.radial_lr_scale * lam * w_f
        u_total = u_perp + radial
    else:
        u_total = u_perp

    # Retraction / clamp
    if cfg.mode == "sphere":
        # Retract to fixed radius
        rad = (
            radius_sched(step)
            if radius_sched is not None
            else jnp.asarray(cfg.radius, dtype=jnp.float32)
        )
        # Broadcast to keepdims shape
        rad_keep = jnp.asarray(rad, dtype=jnp.float32)
        w_new = retract_to_radius(
            w_f + u_total, axes, rad_keep, cfg.eps, dtype_accum=cfg.dtype_accum
        )
    else:
        w_new = w_f + u_total
        mn, mx = cfg.ball_norm_clamp
        w_new = clamp_group_norm(w_new, axes, cfg.eps, mn, mx, dtype_accum=cfg.dtype_accum)

    u_out = (w_new - w_f).astype(u.dtype)

    # Metrics (leaf-level)
    metrics: Dict[str, jnp.ndarray] = {}
    if cfg.emit_metrics:
        w_norm = group_norm(w_f, axes, cfg.eps, dtype_accum=cfg.dtype_accum)
        u_norm = group_norm(u_f, axes, cfg.eps, dtype_accum=cfg.dtype_accum)
        u_perp_norm = group_norm(u_perp, axes, cfg.eps, dtype_accum=cfg.dtype_accum)
        u_par_norm = group_norm(u_par, axes, cfg.eps, dtype_accum=cfg.dtype_accum)

        # Approx angle from tangent norm, robust for small steps
        angle = jnp.arctan2(jnp.squeeze(u_perp_norm), jnp.squeeze(w_norm) + cfg.eps)
        radial_frac = jnp.squeeze(u_par_norm) / (jnp.squeeze(u_norm) + cfg.eps)

        # Reduce per-group metrics to scalars by mean
        metrics = {
            "w_norm_mean": jnp.mean(w_norm),
            "u_norm_mean": jnp.mean(u_norm),
            "u_perp_norm_mean": jnp.mean(u_perp_norm),
            "angle_mean": jnp.mean(angle),
            "radial_frac_mean": jnp.mean(radial_frac),
        }

    return u_out, metrics


def _aggregate_metrics(
    prefix: str, leaf_metrics: Sequence[Dict[str, jnp.ndarray]], masked_frac: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Aggregate a list of leaf metric dicts into stable scalar summaries."""
    if not leaf_metrics:
        out = _init_metrics_dict(prefix)
        out[f"{prefix}/masked_frac"] = masked_frac
        return out

    # Collect keys
    keys = set()
    for m in leaf_metrics:
        keys |= set(m.keys())

    out: Dict[str, jnp.ndarray] = {}
    for k in sorted(keys):
        vals = [m[k] for m in leaf_metrics if k in m]
        if vals:
            out[f"{prefix}/{k}"] = jnp.mean(jnp.stack(vals))
        else:
            out[f"{prefix}/{k}"] = jnp.asarray(0.0, dtype=jnp.float32)

    # Ensure standard keys exist
    init = _init_metrics_dict(prefix)
    for sk, sv in init.items():
        out.setdefault(sk, sv)

    out[f"{prefix}/masked_frac"] = masked_frac
    return out


# -------------------------
# Public Optax transformation
# -------------------------


def hyperball(
    base: optax.GradientTransformation,
    *,
    radius: float = 1.0,
    mode: Mode = "sphere",
    proj_tangent: bool = True,
    granularity: Granularity = "row",
    axes_fn: Optional[AxesFn] = None,
    target_angle: Optional[Schedule] = None,
    target_update_norm: Optional[Schedule] = None,
    update_norm_clip: Tuple[Optional[float], Optional[float]] = (None, None),
    radial_decay: Optional[float] = None,
    radial_lr_scale: float = 1.0,
    ball_norm_clamp: Tuple[Optional[float], Optional[float]] = (None, None),
    mask: Optional[Union[PyTree, Callable[[PyTree], PyTree]]] = None,
    eps: float = 1e-8,
    dtype_accum: jnp.dtype = jnp.float32,
    emit_metrics: bool = True,
    metrics_prefix: str = "hyperball",
) -> optax.GradientTransformation:
    """
    Create an Optax-compatible Hyperball wrapper.

    Parameters
    ----------
    base:
        Any Optax GradientTransformation producing updates from grads.
    radius:
        Target radius for constrained groups in sphere mode.
    mode:
        "sphere" retraction to fixed radius, "ball" optional clamp + radial attractor.
    proj_tangent:
        If True, remove radial component so updates are tangent rotations.
    granularity:
        How norms are computed per leaf: leaf / row / col / channel.
    axes_fn:
        Optional path-aware axes selector. If None, uses granularity defaults.
    target_angle:
        Optional target angular step (radians). If provided, scales tangent
        updates so ||u_perp|| ≈ α||w||.
    target_update_norm:
        Optional target update norm. Mutually exclusive with target_angle in typical use.
    update_norm_clip:
        Optional (min,max) clip on group-wise update norms.
    radial_decay:
        In ball mode, adds an internal radial shrink term: u += -radial_lr_scale * radial_decay * w
    ball_norm_clamp:
        In ball mode, clamp group norms of w_new to (min,max).
    mask:
        Optional pytree bool or callable(params)->mask to select constrained leaves.
    emit_metrics:
        If True, stores last-step scalar summaries in optimizer state under state.last_metrics.
    """
    cfg = HyperballConfig(
        radius=radius,
        mode=mode,
        proj_tangent=proj_tangent,
        granularity=granularity,
        axes_fn=axes_fn,
        target_angle=target_angle,
        target_update_norm=target_update_norm,
        update_norm_clip=update_norm_clip,
        radial_decay=radial_decay,
        radial_lr_scale=radial_lr_scale,
        ball_norm_clamp=ball_norm_clamp,
        mask=mask,
        eps=eps,
        dtype_accum=dtype_accum,
        emit_metrics=emit_metrics,
        metrics_prefix=metrics_prefix,
    )

    # Resolve schedules
    angle_sched = _as_schedule(cfg.target_angle)
    norm_sched = _as_schedule(cfg.target_update_norm)

    # Allow a schedule for radius in the future; for now it's constant.
    radius_sched = _as_schedule(cfg.radius)

    axf = cfg.axes_fn or make_axes_fn(cfg.granularity)

    def init_fn(params: PyTree) -> HyperballState:
        base_state = base.init(params)
        return HyperballState(
            base_state=base_state,
            count=jnp.asarray(0, dtype=jnp.int32),
            last_metrics=_init_metrics_dict(cfg.metrics_prefix) if cfg.emit_metrics else {},
        )

    def update_fn(grads: PyTree, state: HyperballState, params: Optional[PyTree] = None):
        if params is None:
            raise ValueError(
                "Hyperball wrapper requires `params` to compute projections/retractions."
            )

        # Base update
        base_updates, new_base_state = base.update(grads, state.base_state, params)

        # Mask resolution
        mask_tree = _resolve_mask(cfg.mask, params)

        # Leaf-wise transform
        leaf_metrics = []

        def _map_fn(u, w, m):
            nonlocal leaf_metrics
            u_new, mets = _hyperball_leaf(
                u=u,
                w=w,
                apply_mask=bool(m),
                step=state.count,
                cfg=cfg,
                axes_fn=axf,
                angle_sched=angle_sched,
                norm_sched=norm_sched,
                radius_sched=radius_sched,
            )
            if mets:
                leaf_metrics.append(mets)
            return u_new

        updates = jax.tree.map(_map_fn, base_updates, params, mask_tree)

        # Aggregate metrics
        if cfg.emit_metrics:
            # compute fraction of constrained leaves (approx)
            # This is a heuristic: count leaves True vs total leaves
            leaves = jax.tree.leaves(mask_tree)
            total = jnp.asarray(len(leaves), dtype=jnp.float32)
            enabled = jnp.asarray(sum(bool(x) for x in leaves), dtype=jnp.float32)
            masked_frac = jnp.where(total > 0, enabled / total, 0.0)

            metrics = _aggregate_metrics(cfg.metrics_prefix, leaf_metrics, masked_frac)
        else:
            metrics = {}

        new_state = HyperballState(
            base_state=new_base_state,
            count=state.count + jnp.asarray(1, dtype=jnp.int32),
            last_metrics=metrics,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# -------------------------
# Convenience: metric access
# -------------------------


def get_last_metrics(state: HyperballState) -> Mapping[str, jnp.ndarray]:
    """Retrieve last-step scalar metrics from HyperballState."""
    return state.last_metrics
