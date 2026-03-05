from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import jax
import jax.numpy as jnp
import optax

from modulus.optim.groups import make_grouped_hyperball_tx, make_llm_default_labels
from modulus.optim.hyperball import hyperball
from modulus.optim.masks import default_llm_hyperball_mask
from modulus.peft.lora import apply_lora_grad_hook


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    hyperball_on: bool
    grouped: bool
    lora_hook_on: bool


def _model_forward(params: Mapping[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    attn = x @ params["attn"]["kernel"]
    mlp = x @ params["mlp"]["kernel"]
    lora = (x @ params["mlp"]["adapter"]["lora_A"]) @ params["mlp"]["adapter"]["lora_B"]
    return attn + mlp + lora


def _make_initial_params(key: jax.Array, width: int, rank: int) -> Dict[str, Any]:
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 1.0 / jnp.sqrt(float(width))
    return {
        "attn": {"kernel": jax.random.normal(k1, (width, width)) * scale},
        "mlp": {
            "kernel": jax.random.normal(k2, (width, width)) * scale,
            "adapter": {
                "lora_A": jax.random.normal(k3, (width, rank)) * 0.01,
                "lora_B": jnp.zeros((rank, width), dtype=jnp.float32),
            },
        },
        # Included for realistic group/mask behavior even though not used in loss.
        "embed": {"embedding": jax.random.normal(k1, (128, width)) * 0.01},
        "LayerNorm_0": {"scale": jnp.ones((width,), dtype=jnp.float32)},
    }


def _make_teacher(key: jax.Array, width: int) -> jnp.ndarray:
    return jax.random.normal(key, (width, width)) * (1.0 / jnp.sqrt(float(width)))


def _build_optimizer(cfg: BenchmarkConfig, params: Mapping[str, Any], lr: float, wd: float):
    base = optax.adamw(learning_rate=lr, weight_decay=wd)
    mask_fn = default_llm_hyperball_mask(
        include_embeddings=False, exclude_lora=True, exclude_1d=True
    )

    if not cfg.grouped:
        if not cfg.hyperball_on:
            return base
        return hyperball(
            base,
            radius=1.0,
            mode="sphere",
            proj_tangent=True,
            granularity="row",
            target_angle=0.04,
            mask=mask_fn,
            emit_metrics=True,
        )

    labels_fn = make_llm_default_labels()
    base_by_group = {
        "attn": base,
        "mlp": base,
        "other": base,
        "embed": base,
        "norm": base,
        "bias": base,
    }

    if not cfg.hyperball_on:
        hb_kwargs_by_group: Mapping[str, Mapping[str, Any]] = {}
    else:
        hb_common = dict(
            radius=1.0,
            mode="sphere",
            proj_tangent=True,
            granularity="row",
            mask=mask_fn,
            emit_metrics=True,
        )
        hb_kwargs_by_group = {
            "attn": dict(**hb_common, target_angle=0.03),
            "mlp": dict(**hb_common, target_angle=0.05),
            "other": {},
            "embed": {},
            "norm": {},
            "bias": {},
        }

    return make_grouped_hyperball_tx(
        base_by_group=base_by_group,
        hyperball_kwargs_by_group=hb_kwargs_by_group,
        labels_fn=labels_fn,
        default_group="other",
    )(params)


def _loss(params: Mapping[str, Any], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = _model_forward(params, x)
    return jnp.mean((pred - y) ** 2)


def _find_hyperball_metric_maps(root: Any) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    stack = [root]
    seen = set()

    while stack:
        cur = stack.pop()
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)

        last_metrics = getattr(cur, "last_metrics", None)
        if isinstance(last_metrics, Mapping):
            out.append(last_metrics)

        if dataclasses.is_dataclass(cur):
            for f in dataclasses.fields(cur):
                stack.append(getattr(cur, f.name))
            continue

        if isinstance(cur, Mapping):
            stack.extend(cur.values())
            continue

        if hasattr(cur, "_asdict"):
            stack.extend(cur._asdict().values())
            continue

        if isinstance(cur, (list, tuple)):
            stack.extend(cur)

    return out


def _aggregate_hyperball_metrics(opt_state: Any) -> Dict[str, float]:
    metric_maps = _find_hyperball_metric_maps(opt_state)
    if not metric_maps:
        return {}

    merged: Dict[str, List[float]] = {}
    for mm in metric_maps:
        for k, v in mm.items():
            merged.setdefault(k, []).append(float(jnp.asarray(v)))
    return {k: sum(vals) / len(vals) for k, vals in merged.items()}


def _make_step_fn(tx, lora_hook_on: bool):
    def step_fn(params, opt_state, x, y):
        loss_val, grads = jax.value_and_grad(_loss)(params, x, y)
        if lora_hook_on:
            grads = apply_lora_grad_hook(params, grads, a_name="lora_A", b_name="lora_B", eps=1e-6)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    return jax.jit(step_fn)


def _timestamp_dir(base_dir: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    out_root = root / "artifacts" / "benchmarks"
    out_dir = _timestamp_dir(out_root) if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        BenchmarkConfig("baseline", hyperball_on=False, grouped=False, lora_hook_on=False),
        BenchmarkConfig("lora_hook_only", hyperball_on=False, grouped=False, lora_hook_on=True),
        BenchmarkConfig(
            "hyperball_ungrouped", hyperball_on=True, grouped=False, lora_hook_on=False
        ),
        BenchmarkConfig("hyperball_grouped", hyperball_on=True, grouped=True, lora_hook_on=False),
        BenchmarkConfig(
            "hyperball_grouped_lora", hyperball_on=True, grouped=True, lora_hook_on=True
        ),
    ]

    rng = jax.random.PRNGKey(args.seed)
    k_params, k_teacher, k_batch = jax.random.split(rng, 3)
    init_params = _make_initial_params(k_params, width=args.width, rank=args.lora_rank)
    teacher = _make_teacher(k_teacher, width=args.width)

    # Fixed batches keep benchmark comparisons fair.
    x_all = jax.random.normal(
        k_batch, (args.steps + args.warmup_steps, args.batch_size, args.width)
    )
    y_all = jnp.einsum("tbd,df->tbf", x_all, teacher)

    step_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for cfg in configs:
        params = jax.tree.map(lambda z: jnp.array(z, copy=True), init_params)
        tx = _build_optimizer(cfg, params=params, lr=args.lr, wd=args.weight_decay)
        opt_state = tx.init(params)
        step_fn = _make_step_fn(tx=tx, lora_hook_on=cfg.lora_hook_on)

        # JIT warmup (excluded from timed rows).
        for i in range(args.warmup_steps):
            params, opt_state, loss_val = step_fn(params, opt_state, x_all[i], y_all[i])
            jax.block_until_ready(loss_val)

        loss_last = None
        step_ms_total = 0.0
        hb_last = {}

        for step in range(args.steps):
            x = x_all[args.warmup_steps + step]
            y = y_all[args.warmup_steps + step]

            t0 = time.perf_counter()
            params, opt_state, loss_val = step_fn(params, opt_state, x, y)
            jax.block_until_ready(loss_val)
            step_ms = (time.perf_counter() - t0) * 1000.0

            hb_metrics = _aggregate_hyperball_metrics(opt_state)
            loss_scalar = float(loss_val)

            row = {
                "config": cfg.name,
                "hyperball_on": int(cfg.hyperball_on),
                "grouped": int(cfg.grouped),
                "lora_hook_on": int(cfg.lora_hook_on),
                "step": step,
                "loss": f"{loss_scalar:.8f}",
                "step_ms": f"{step_ms:.4f}",
                "hyperball_angle_mean": (
                    f"{hb_metrics.get('hyperball/angle_mean', float('nan')):.8f}"
                ),
                "hyperball_radial_frac_mean": (
                    f"{hb_metrics.get('hyperball/radial_frac_mean', float('nan')):.8f}"
                ),
            }
            step_rows.append(row)
            loss_last = loss_scalar
            step_ms_total += step_ms
            hb_last = hb_metrics

        summary_rows.append(
            {
                "config": cfg.name,
                "hyperball_on": int(cfg.hyperball_on),
                "grouped": int(cfg.grouped),
                "lora_hook_on": int(cfg.lora_hook_on),
                "steps": args.steps,
                "final_loss": f"{(loss_last if loss_last is not None else float('nan')):.8f}",
                "avg_step_ms": f"{(step_ms_total / max(args.steps, 1)):.4f}",
                "final_hyperball_angle_mean": (
                    f"{hb_last.get('hyperball/angle_mean', float('nan')):.8f}"
                ),
                "final_hyperball_radial_frac_mean": (
                    f"{hb_last.get('hyperball/radial_frac_mean', float('nan')):.8f}"
                ),
            }
        )

    step_csv = out_dir / "benchmark_steps.csv"
    summary_csv = out_dir / "benchmark_summary.csv"
    meta_json = out_dir / "run_meta.json"

    _write_csv(
        step_csv,
        step_rows,
        fieldnames=[
            "config",
            "hyperball_on",
            "grouped",
            "lora_hook_on",
            "step",
            "loss",
            "step_ms",
            "hyperball_angle_mean",
            "hyperball_radial_frac_mean",
        ],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "config",
            "hyperball_on",
            "grouped",
            "lora_hook_on",
            "steps",
            "final_loss",
            "avg_step_ms",
            "final_hyperball_angle_mean",
            "final_hyperball_radial_frac_mean",
        ],
    )

    meta = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "width": args.width,
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "output_dir": str(out_dir),
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {step_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {meta_json}")
    print("")
    print("Summary:")
    for r in summary_rows:
        angle = r["final_hyperball_angle_mean"]
        radial = r["final_hyperball_radial_frac_mean"]
        print(
            f"- {r['config']}: final_loss={r['final_loss']}, avg_step_ms={r['avg_step_ms']}, "
            f"angle={angle}, radial={radial}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MODULUS ablation benchmarks and emit CSV artifacts."
    )
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
