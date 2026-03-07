from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

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


def _available_benchmark_configs() -> Dict[str, BenchmarkConfig]:
    return {
        "baseline": BenchmarkConfig(
            "baseline", hyperball_on=False, grouped=False, lora_hook_on=False
        ),
        "lora_hook_only": BenchmarkConfig(
            "lora_hook_only", hyperball_on=False, grouped=False, lora_hook_on=True
        ),
        "hyperball_ungrouped": BenchmarkConfig(
            "hyperball_ungrouped", hyperball_on=True, grouped=False, lora_hook_on=False
        ),
        "hyperball_grouped": BenchmarkConfig(
            "hyperball_grouped", hyperball_on=True, grouped=True, lora_hook_on=False
        ),
        "hyperball_grouped_lora": BenchmarkConfig(
            "hyperball_grouped_lora", hyperball_on=True, grouped=True, lora_hook_on=True
        ),
    }


@dataclass(frozen=True)
class ModelConfig:
    width: int
    num_layers: int
    num_heads: int
    seq_len: int
    vocab_size: int
    mlp_mult: int
    lora_rank: int

    @property
    def head_dim(self) -> int:
        return self.width // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        return self.width * self.mlp_mult


@dataclass(frozen=True)
class ObjectiveConfig:
    distill_temperature: float
    distill_weight: float
    label_smoothing: float


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    name: str
    config: Optional[str]
    train_split: str
    eval_split: str
    text_keys: Tuple[str, ...]
    shuffle_buffer: int
    max_doc_tokens: int
    train_max_docs: Optional[int]
    eval_max_docs: Optional[int]
    trust_remote_code: bool
    rows_endpoint: str
    rows_page_size: int
    http_max_retries: int
    http_min_interval_sec: float
    http_token_env: str
    http_cache_dir: Optional[str]
    http_cache_read: bool
    http_cache_write: bool


def _layer_norm(x: jnp.ndarray, scale: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.mean((x32 - mean) ** 2, axis=-1, keepdims=True)
    y = (x32 - mean) / jnp.sqrt(var + eps)
    return y * scale.astype(jnp.float32)


def _split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    bsz, seqlen, width = x.shape
    head_dim = width // num_heads
    return x.reshape(bsz, seqlen, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    bsz, num_heads, seqlen, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(bsz, seqlen, num_heads * head_dim)


def _causal_mask(seq_len: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]


def _model_forward(
    params: Mapping[str, Any],
    tokens: jnp.ndarray,
    *,
    model_cfg: ModelConfig,
    causal_mask: jnp.ndarray,
    use_lora: bool,
) -> jnp.ndarray:
    bsz, seqlen = tokens.shape
    del bsz

    x = params["embed"]["token_embedding"][tokens]
    x = x + params["embed"]["pos_embedding"][None, :seqlen, :]
    x = x.astype(jnp.float32)

    for layer_idx in range(model_cfg.num_layers):
        blk = params[f"block_{layer_idx}"]

        h = _layer_norm(x, blk["norm1"]["scale"])
        qkv = h @ blk["attn"]["qkv_kernel"]
        q, k, v = jnp.split(qkv, 3, axis=-1)

        qh = _split_heads(q, model_cfg.num_heads)
        kh = _split_heads(k, model_cfg.num_heads)
        vh = _split_heads(v, model_cfg.num_heads)

        logits = jnp.einsum("bhqd,bhkd->bhqk", qh, kh) / math.sqrt(float(model_cfg.head_dim))
        logits = jnp.where(causal_mask[:, :, :seqlen, :seqlen], logits, -1e30)
        attn = jax.nn.softmax(logits, axis=-1)
        attn_ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, vh)
        attn_out = _merge_heads(attn_ctx) @ blk["attn"]["out_kernel"]
        x = x + attn_out

        h2 = _layer_norm(x, blk["norm2"]["scale"])
        mlp = jax.nn.gelu(h2 @ blk["mlp"]["up_kernel"], approximate=False)
        mlp = mlp @ blk["mlp"]["down_kernel"]

        if use_lora:
            lora = (h2 @ blk["mlp"]["adapter"]["lora_A"]) @ blk["mlp"]["adapter"]["lora_B"]
            mlp = mlp + lora

        x = x + mlp

    x = _layer_norm(x, params["final_norm"]["scale"])
    return x @ params["lm_head"]["kernel"]


def _make_initial_params(key: jax.Array, model_cfg: ModelConfig) -> Dict[str, Any]:
    keys = iter(jax.random.split(key, 6 + model_cfg.num_layers * 8))

    def next_key() -> jax.Array:
        return next(keys)

    w_scale = 1.0 / math.sqrt(float(model_cfg.width))
    mlp_up_scale = 1.0 / math.sqrt(float(model_cfg.width))
    mlp_down_scale = 1.0 / math.sqrt(float(model_cfg.mlp_hidden))

    params: Dict[str, Any] = {
        "embed": {
            "token_embedding": jax.random.normal(
                next_key(), (model_cfg.vocab_size, model_cfg.width)
            ).astype(jnp.float32)
            * 0.02,
            "pos_embedding": jax.random.normal(
                next_key(), (model_cfg.seq_len, model_cfg.width)
            ).astype(jnp.float32)
            * 0.01,
        },
        "final_norm": {"scale": jnp.ones((model_cfg.width,), dtype=jnp.float32)},
        "lm_head": {
            "kernel": jax.random.normal(next_key(), (model_cfg.width, model_cfg.vocab_size)).astype(
                jnp.float32
            )
            * w_scale
        },
    }

    for layer_idx in range(model_cfg.num_layers):
        params[f"block_{layer_idx}"] = {
            "attn": {
                "qkv_kernel": jax.random.normal(
                    next_key(), (model_cfg.width, 3 * model_cfg.width)
                ).astype(jnp.float32)
                * w_scale,
                "out_kernel": jax.random.normal(
                    next_key(), (model_cfg.width, model_cfg.width)
                ).astype(jnp.float32)
                * w_scale,
            },
            "mlp": {
                "up_kernel": jax.random.normal(
                    next_key(), (model_cfg.width, model_cfg.mlp_hidden)
                ).astype(jnp.float32)
                * mlp_up_scale,
                "down_kernel": jax.random.normal(
                    next_key(), (model_cfg.mlp_hidden, model_cfg.width)
                ).astype(jnp.float32)
                * mlp_down_scale,
                "adapter": {
                    "lora_A": jax.random.normal(
                        next_key(), (model_cfg.width, model_cfg.lora_rank)
                    ).astype(jnp.float32)
                    * 0.01,
                    "lora_B": jnp.zeros((model_cfg.lora_rank, model_cfg.width), dtype=jnp.float32),
                },
            },
            "norm1": {"scale": jnp.ones((model_cfg.width,), dtype=jnp.float32)},
            "norm2": {"scale": jnp.ones((model_cfg.width,), dtype=jnp.float32)},
        }

    return params


def _build_optimizer(
    cfg: BenchmarkConfig,
    params: Mapping[str, Any],
    *,
    learning_rate: Any,
    wd: float,
    grad_clip_norm: float,
):
    base = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=wd),
    )
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


def _build_lr_schedule(
    *,
    base_lr: float,
    schedule_name: str,
    warmup_steps: int,
    min_ratio: float,
    total_steps: int,
):
    if schedule_name == "constant":
        return optax.constant_schedule(base_lr)

    if schedule_name == "warmup_cosine":
        warmup = min(max(warmup_steps, 0), total_steps)
        if warmup > 0:
            warmup_sched = optax.linear_schedule(
                init_value=0.0,
                end_value=base_lr,
                transition_steps=max(warmup, 1),
            )
        else:
            warmup_sched = optax.constant_schedule(base_lr)
        cosine_sched = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=max(total_steps - warmup, 1),
            alpha=min_ratio,
        )
        if warmup > 0:
            return optax.join_schedules([warmup_sched, cosine_sched], [warmup])
        return cosine_sched

    raise ValueError(
        f"Unknown --lr-schedule '{schedule_name}'. Valid: constant, warmup_cosine."
    )


def _next_token_ce(
    logits: jnp.ndarray,
    tokens: jnp.ndarray,
    *,
    label_smoothing: float,
) -> jnp.ndarray:
    logits_next = logits[:, :-1, :]
    labels_next = tokens[:, 1:]
    if label_smoothing <= 0.0:
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits_next, labels_next))

    vocab = logits_next.shape[-1]
    one_hot = jax.nn.one_hot(labels_next, num_classes=vocab, dtype=jnp.float32)
    smooth = label_smoothing / float(vocab)
    target = one_hot * (1.0 - label_smoothing) + smooth
    log_probs = jax.nn.log_softmax(logits_next, axis=-1)
    return -jnp.mean(jnp.sum(target * log_probs, axis=-1))


def _objective(
    params: Mapping[str, Any],
    teacher_params: Mapping[str, Any],
    tokens: jnp.ndarray,
    *,
    model_cfg: ModelConfig,
    objective_cfg: ObjectiveConfig,
    causal_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    student_logits = _model_forward(
        params,
        tokens,
        model_cfg=model_cfg,
        causal_mask=causal_mask,
        use_lora=True,
    )
    teacher_logits = jax.lax.stop_gradient(
        _model_forward(
            teacher_params,
            tokens,
            model_cfg=model_cfg,
            causal_mask=causal_mask,
            use_lora=False,
        )
    )

    temp = jnp.asarray(objective_cfg.distill_temperature, dtype=jnp.float32)
    student_log_probs = jax.nn.log_softmax(student_logits / temp, axis=-1)
    teacher_probs = jax.nn.softmax(teacher_logits / temp, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / temp, axis=-1)
    distill_kl = jnp.mean(
        jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    ) * (temp**2)

    next_token_ce = _next_token_ce(
        student_logits,
        tokens,
        label_smoothing=objective_cfg.label_smoothing,
    )
    total = (
        objective_cfg.distill_weight * distill_kl
        + (1.0 - objective_cfg.distill_weight) * next_token_ce
    )
    return total, {"distill_kl": distill_kl, "next_token_ce": next_token_ce}


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


def _make_step_fn(
    tx: optax.GradientTransformation,
    teacher_params: Mapping[str, Any],
    *,
    lora_hook_on: bool,
    model_cfg: ModelConfig,
    objective_cfg: ObjectiveConfig,
    grad_accum_steps: int,
    causal_mask: jnp.ndarray,
):
    def loss_with_aux(p, tok):
        return _objective(
            p,
            teacher_params,
            tok,
            model_cfg=model_cfg,
            objective_cfg=objective_cfg,
            causal_mask=causal_mask,
        )

    def step_fn(params, opt_state, tokens):
        if grad_accum_steps == 1:
            (loss_val, aux), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(params, tokens)
        else:
            micro_bs = tokens.shape[0] // grad_accum_steps
            tokens_micro = tokens.reshape((grad_accum_steps, micro_bs, tokens.shape[1]))
            grads = jax.tree.map(jnp.zeros_like, params)
            loss_val = jnp.asarray(0.0, dtype=jnp.float32)
            distill_kl = jnp.asarray(0.0, dtype=jnp.float32)
            next_token_ce = jnp.asarray(0.0, dtype=jnp.float32)

            for i in range(grad_accum_steps):
                (loss_i, aux_i), grads_i = jax.value_and_grad(loss_with_aux, has_aux=True)(
                    params, tokens_micro[i]
                )
                grads = jax.tree.map(lambda a, b: a + b, grads, grads_i)
                loss_val = loss_val + loss_i
                distill_kl = distill_kl + aux_i["distill_kl"]
                next_token_ce = next_token_ce + aux_i["next_token_ce"]

            scale = 1.0 / float(grad_accum_steps)
            grads = jax.tree.map(lambda g: g * scale, grads)
            loss_val = loss_val * scale
            aux = {
                "distill_kl": distill_kl * scale,
                "next_token_ce": next_token_ce * scale,
            }

        if lora_hook_on:
            grads = apply_lora_grad_hook(params, grads, a_name="lora_A", b_name="lora_B", eps=1e-6)

        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        update_norm = optax.global_norm(updates)
        new_params = optax.apply_updates(params, updates)
        return (
            new_params,
            new_opt_state,
            loss_val,
            aux["next_token_ce"],
            aux["distill_kl"],
            grad_norm,
            update_norm,
        )

    return jax.jit(step_fn)


def _make_eval_fn(
    teacher_params: Mapping[str, Any],
    *,
    model_cfg: ModelConfig,
    objective_cfg: ObjectiveConfig,
    causal_mask: jnp.ndarray,
):
    def eval_fn(params, tokens):
        loss_val, aux = _objective(
            params,
            teacher_params,
            tokens,
            model_cfg=model_cfg,
            objective_cfg=objective_cfg,
            causal_mask=causal_mask,
        )
        return loss_val, aux["next_token_ce"], aux["distill_kl"]

    return jax.jit(eval_fn)


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


def _fmt_metric(v: float, digits: int = 4) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def _make_token_batches(
    key: jax.Array,
    *,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    shift_start_batch: int,
    rare_inject_prob: float,
) -> jnp.ndarray:
    k_base, k_shift, k_mix, k_rare = jax.random.split(key, 4)
    ranks = jnp.arange(vocab_size, dtype=jnp.float32) + 1.0
    base_logits = -1.10 * jnp.log(ranks)
    shifted_logits = base_logits + jnp.where(
        jnp.arange(vocab_size) >= (vocab_size // 2),
        1.20,
        -0.25,
    ).astype(jnp.float32)

    base_tokens = jax.random.categorical(
        k_base,
        base_logits,
        shape=(num_batches, batch_size, seq_len),
    ).astype(jnp.int32)
    shifted_tokens = jax.random.categorical(
        k_shift,
        shifted_logits,
        shape=(num_batches, batch_size, seq_len),
    ).astype(jnp.int32)

    phase_mask = jnp.arange(num_batches)[:, None, None] >= shift_start_batch
    tokens = jnp.where(phase_mask, shifted_tokens, base_tokens)

    rare_bucket = max(vocab_size // 8, 1)
    rare_start = max(vocab_size - rare_bucket, 0)
    rare_start = min(rare_start, vocab_size - 1)
    rare_tokens = jax.random.randint(
        k_rare,
        shape=tokens.shape,
        minval=rare_start,
        maxval=vocab_size,
        dtype=jnp.int32,
    )
    inject_mask = jax.random.bernoulli(k_mix, p=rare_inject_prob, shape=tokens.shape)
    return jnp.where(inject_mask, rare_tokens, tokens).astype(jnp.int32)


_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _is_numpy_umath_center_error(exc: BaseException) -> bool:
    s = str(exc)
    return "_center" in s and "numpy._core.umath" in s


def _stable_token_id(token: str, vocab_size: int) -> int:
    if vocab_size <= 3:
        return 0
    h = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
    return 3 + (h % (vocab_size - 3))


def _text_to_ids(text: str, *, vocab_size: int, max_doc_tokens: int) -> List[int]:
    pieces = _WORD_RE.findall(text.lower())
    if not pieces:
        return []

    budget = max(max_doc_tokens, 2)
    body = pieces[: max(0, budget - 2)]
    ids = [1]
    ids.extend(_stable_token_id(tok, vocab_size) for tok in body)
    ids.append(2)
    return ids


def _extract_text(example: Mapping[str, Any], text_keys: Sequence[str]) -> Optional[str]:
    for key in text_keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _make_hf_text_iterator(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_keys: Sequence[str],
    shuffle_buffer: int,
    seed: int,
    trust_remote_code: bool,
) -> Iterator[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        if _is_numpy_umath_center_error(exc):
            raise RuntimeError(
                "Detected inconsistent NumPy installation in this Colab runtime. "
                "Run: `%pip install -U --force-reinstall --no-cache-dir \"numpy==2.1.3\"` "
                "then restart runtime and rerun setup."
            ) from exc
        raise RuntimeError(
            "Hugging Face datasets is required for --data-source=hf_stream. "
            "Install with: python -m pip install datasets"
        ) from exc

    load_kwargs: Dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if dataset_config:
        load_kwargs["name"] = dataset_config
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    try:
        dataset = load_dataset(**load_kwargs)
    except TypeError:
        load_kwargs.pop("trust_remote_code", None)
        try:
            dataset = load_dataset(**load_kwargs)
        except Exception as exc:
            if _is_numpy_umath_center_error(exc):
                raise RuntimeError(
                    "Detected inconsistent NumPy installation in this Colab runtime. "
                    "Run: `%pip install -U --force-reinstall --no-cache-dir \"numpy==2.1.3\"` "
                    "then restart runtime and rerun setup."
                ) from exc
            raise RuntimeError(
                "Failed to load dataset after retry without trust_remote_code. "
                "Try --dataset-name JeanKaddour/minipile to validate streaming path first."
            ) from exc
    except Exception as exc:
        if _is_numpy_umath_center_error(exc):
            raise RuntimeError(
                "Detected inconsistent NumPy installation in this Colab runtime. "
                "Run: `%pip install -U --force-reinstall --no-cache-dir \"numpy==2.1.3\"` "
                "then restart runtime and rerun setup."
            ) from exc
        msg = str(exc)
        hint = (
            "Failed to load dataset in hf_stream mode. "
            "Try a known public fallback such as --dataset-name JeanKaddour/minipile "
            "or authenticate with `huggingface-cli login` if the dataset is gated/private."
        )
        if "DatasetNotFoundError" in msg or "doesn't exist on the Hub" in msg:
            raise RuntimeError(f"{hint} Original error: {exc}") from exc
        raise

    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for example in dataset:
        if not isinstance(example, Mapping):
            continue
        text = _extract_text(example, text_keys)
        if text is not None:
            yield text


def _collect_stream_token_ids(
    text_iter: Iterator[str],
    *,
    required_tokens: int,
    vocab_size: int,
    max_doc_tokens: int,
    max_docs: Optional[int],
    progress_label: str,
) -> Tuple[jnp.ndarray, int]:
    flat_tokens: List[int] = []
    docs_seen = 0

    for text in text_iter:
        try:
            docs_seen += 1
            flat_tokens.extend(
                _text_to_ids(
                    text,
                    vocab_size=vocab_size,
                    max_doc_tokens=max_doc_tokens,
                )
            )
        except ImportError as exc:
            if _is_numpy_umath_center_error(exc):
                raise RuntimeError(
                    "Detected inconsistent NumPy installation while streaming dataset. "
                    "Run: `%pip install -U --force-reinstall --no-cache-dir \"numpy==2.1.3\"` "
                    "then restart runtime and rerun setup."
                ) from exc
            raise

        if len(flat_tokens) >= required_tokens:
            break
        if max_docs is not None and docs_seen >= max_docs:
            break
        if docs_seen % 1000 == 0:
            print(
                f"{progress_label}: docs={docs_seen}, "
                f"tokens={len(flat_tokens)}/{required_tokens}"
            )

    if len(flat_tokens) < required_tokens:
        raise RuntimeError(
            f"{progress_label}: insufficient tokens ({len(flat_tokens)} < {required_tokens}). "
            "Increase max docs, reduce benchmark size, or use a denser text field."
        )

    token_arr = jnp.asarray(flat_tokens[:required_tokens], dtype=jnp.int32)
    return token_arr, docs_seen


def _make_hf_stream_batches(
    ds_cfg: DatasetConfig,
    *,
    split: str,
    seed: int,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    max_docs: Optional[int],
) -> jnp.ndarray:
    required_tokens = num_batches * batch_size * seq_len
    text_iter = _make_hf_text_iterator(
        dataset_name=ds_cfg.name,
        dataset_config=ds_cfg.config,
        split=split,
        text_keys=ds_cfg.text_keys,
        shuffle_buffer=ds_cfg.shuffle_buffer,
        seed=seed,
        trust_remote_code=ds_cfg.trust_remote_code,
    )
    flat, docs_seen = _collect_stream_token_ids(
        text_iter,
        required_tokens=required_tokens,
        vocab_size=vocab_size,
        max_doc_tokens=ds_cfg.max_doc_tokens,
        max_docs=max_docs,
        progress_label=f"hf_stream[{split}]",
    )
    print(
        f"hf_stream[{split}]: collected {required_tokens} tokens from {docs_seen} documents."
    )
    return flat.reshape((num_batches, batch_size, seq_len))


def _make_hf_http_text_iterator(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_keys: Sequence[str],
    rows_endpoint: str,
    rows_page_size: int,
    max_retries: int,
    min_interval_sec: float,
    token_env: str,
    cache_dir: Optional[str],
    cache_read: bool,
    cache_write: bool,
) -> Iterator[str]:
    def cache_path_for_url(root: Path, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return root / f"{digest}.json"

    offset = 0
    page_len = min(max(rows_page_size, 1), 100)
    if rows_page_size != page_len:
        print(
            f"dataset_rows_page_size={rows_page_size} adjusted to {page_len} "
            "(datasets-server /rows max length is 100)."
        )
    endpoint = rows_endpoint.rstrip("/")
    token = os.environ.get(token_env) if token_env else None
    headers = {"accept": "application/json", "user-agent": "modulus-benchmark/1.0"}
    if token:
        headers["authorization"] = f"Bearer {token}"

    cache_root: Optional[Path] = None
    if cache_dir and (cache_read or cache_write):
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

    cache_hits = 0
    cache_misses = 0

    while True:
        query = {
            "dataset": dataset_name,
            "split": split,
            "offset": offset,
            "length": page_len,
        }
        if dataset_config:
            query["config"] = dataset_config
        url = f"{endpoint}?{urllib.parse.urlencode(query)}"

        payload: Dict[str, Any] | None = None
        loaded_from_cache = False
        last_error: Optional[BaseException] = None
        cache_path: Optional[Path] = None
        if cache_root is not None:
            cache_path = cache_path_for_url(cache_root, url)

        if cache_path is not None and cache_read and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                loaded_from_cache = True
                cache_hits += 1
            except Exception:
                payload = None

        if payload is None:
            cache_misses += 1
            for attempt in range(max_retries + 1):
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code == 429 and attempt < max_retries:
                        retry_after = exc.headers.get("Retry-After") if exc.headers else None
                        if retry_after is not None:
                            try:
                                sleep_s = max(float(retry_after), min_interval_sec)
                            except ValueError:
                                sleep_s = min_interval_sec
                        else:
                            sleep_s = max(min_interval_sec, 2.0 * (attempt + 1))
                        sleep_s += random.uniform(0.0, 0.5)
                        print(
                            f"HTTP 429 from datasets-server; backing off {sleep_s:.2f}s "
                            f"(attempt {attempt + 1}/{max_retries})."
                        )
                        time.sleep(sleep_s)
                        continue
                    if attempt >= max_retries:
                        break
                    time.sleep(max(min_interval_sec, 1.0 + attempt))
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_retries:
                        break
                    time.sleep(max(min_interval_sec, 1.5 * (attempt + 1)))

        if payload is None:
            raise RuntimeError(
                f"Failed to fetch dataset rows from {url}. Last error: {last_error}"
            )

        if (
            cache_path is not None
            and cache_write
            and not loaded_from_cache
            and isinstance(payload, Mapping)
        ):
            try:
                tmp_path = cache_path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(payload), encoding="utf-8")
                tmp_path.replace(cache_path)
            except Exception:
                pass

        rows = payload.get("rows", [])
        if not isinstance(rows, list) or len(rows) == 0:
            break

        for row in rows:
            if isinstance(row, Mapping):
                example = row.get("row", row)
                if isinstance(example, Mapping):
                    text = _extract_text(example, text_keys)
                    if text is not None:
                        yield text

        offset += len(rows)
        if min_interval_sec > 0 and not loaded_from_cache:
            time.sleep(min_interval_sec)

        total_pages = cache_hits + cache_misses
        if total_pages > 0 and total_pages % 100 == 0:
            print(
                f"hf_http cache stats: pages={total_pages}, "
                f"hits={cache_hits}, misses={cache_misses}"
            )


def _make_hf_http_batches(
    ds_cfg: DatasetConfig,
    *,
    split: str,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    max_docs: Optional[int],
) -> jnp.ndarray:
    required_tokens = num_batches * batch_size * seq_len
    text_iter = _make_hf_http_text_iterator(
        dataset_name=ds_cfg.name,
        dataset_config=ds_cfg.config,
        split=split,
        text_keys=ds_cfg.text_keys,
        rows_endpoint=ds_cfg.rows_endpoint,
        rows_page_size=ds_cfg.rows_page_size,
        max_retries=ds_cfg.http_max_retries,
        min_interval_sec=ds_cfg.http_min_interval_sec,
        token_env=ds_cfg.http_token_env,
        cache_dir=ds_cfg.http_cache_dir,
        cache_read=ds_cfg.http_cache_read,
        cache_write=ds_cfg.http_cache_write,
    )
    flat, docs_seen = _collect_stream_token_ids(
        text_iter,
        required_tokens=required_tokens,
        vocab_size=vocab_size,
        max_doc_tokens=ds_cfg.max_doc_tokens,
        max_docs=max_docs,
        progress_label=f"hf_http[{split}]",
    )
    print(
        f"hf_http[{split}]: collected {required_tokens} tokens from {docs_seen} documents."
    )
    return flat.reshape((num_batches, batch_size, seq_len))


def run(args: argparse.Namespace) -> None:
    if args.width % args.num_heads != 0:
        raise ValueError("--width must be divisible by --num-heads")
    if args.batch_size % args.grad_accum_steps != 0:
        raise ValueError("--batch-size must be divisible by --grad-accum-steps")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2 for next-token objective")
    if args.vocab_size < 16:
        raise ValueError("--vocab-size must be >= 16")
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be >= 0")
    if args.step_record_interval < 1:
        raise ValueError("--step-record-interval must be >= 1")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.token_pool_batches < 1:
        raise ValueError("--token-pool-batches must be >= 1")
    if args.target_runtime_minutes < 0:
        raise ValueError("--target-runtime-minutes must be >= 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be >= 0")
    if not (0.0 <= args.lr_min_ratio <= 1.0):
        raise ValueError("--lr-min-ratio must be in [0, 1]")
    if args.lr_total_steps is not None and args.lr_total_steps < 1:
        raise ValueError("--lr-total-steps must be >= 1 when provided")
    if args.dataset_http_min_interval_sec < 0:
        raise ValueError("--dataset-http-min-interval-sec must be >= 0")
    if (
        (args.dataset_http_cache_read or args.dataset_http_cache_write)
        and not args.dataset_http_cache_dir
    ):
        raise ValueError(
            "--dataset-http-cache-dir is required when cache read/write is enabled"
        )
    if args.max_steps is not None and args.max_steps < args.steps:
        raise ValueError("--max-steps must be >= --steps")
    if args.target_train_tokens is not None and args.target_train_tokens < 1:
        raise ValueError("--target-train-tokens must be >= 1 when provided")
    if not (0.0 <= args.distill_weight <= 1.0):
        raise ValueError("--distill-weight must be in [0, 1]")
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not (0.0 <= args.shift_start_frac <= 1.0):
        raise ValueError("--shift-start-frac must be in [0, 1]")
    if args.data_source not in {"synthetic", "hf_stream", "hf_http"}:
        raise ValueError("--data-source must be one of: synthetic, hf_stream, hf_http")
    if args.data_source in {"hf_stream", "hf_http"} and not args.dataset_name:
        raise ValueError("--dataset-name is required when --data-source is hf_stream or hf_http")

    root = Path(__file__).resolve().parents[1]
    out_root = root / "artifacts" / "benchmarks"
    out_dir = _timestamp_dir(out_root) if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        mlp_mult=args.mlp_mult,
        lora_rank=args.lora_rank,
    )
    objective_cfg = ObjectiveConfig(
        distill_temperature=args.distill_temperature,
        distill_weight=args.distill_weight,
        label_smoothing=args.label_smoothing,
    )
    ds_cfg = DatasetConfig(
        source=args.data_source,
        name=args.dataset_name,
        config=args.dataset_config,
        train_split=args.dataset_train_split,
        eval_split=args.dataset_eval_split,
        text_keys=tuple(k.strip() for k in args.dataset_text_keys.split(",") if k.strip()),
        shuffle_buffer=args.dataset_shuffle_buffer,
        max_doc_tokens=args.dataset_max_doc_tokens,
        train_max_docs=args.dataset_train_max_docs,
        eval_max_docs=args.dataset_eval_max_docs,
        trust_remote_code=args.dataset_trust_remote_code,
        rows_endpoint=args.dataset_rows_endpoint,
        rows_page_size=args.dataset_rows_page_size,
        http_max_retries=args.dataset_http_max_retries,
        http_min_interval_sec=args.dataset_http_min_interval_sec,
        http_token_env=args.dataset_http_token_env,
        http_cache_dir=args.dataset_http_cache_dir,
        http_cache_read=args.dataset_http_cache_read,
        http_cache_write=args.dataset_http_cache_write,
    )
    mask = _causal_mask(args.seq_len)

    all_configs = _available_benchmark_configs()
    requested_cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not requested_cfgs or requested_cfgs == ["all"]:
        configs = list(all_configs.values())
    else:
        unknown = [c for c in requested_cfgs if c not in all_configs]
        if unknown:
            raise ValueError(
                f"Unknown --configs value(s): {unknown}. "
                f"Valid: {sorted(all_configs.keys())} or 'all'."
            )
        configs = [all_configs[c] for c in requested_cfgs]

    train_pool_batches = max(args.token_pool_batches, args.warmup_steps + 1)
    shift_start_batch = int(train_pool_batches * args.shift_start_frac)
    tokens_per_step = args.batch_size * args.seq_len
    target_token_steps = 0
    if args.target_train_tokens is not None:
        target_token_steps = math.ceil(args.target_train_tokens / float(tokens_per_step))
        if args.max_steps is not None and args.max_steps < target_token_steps:
            raise ValueError(
                "--max-steps is smaller than steps required by --target-train-tokens. "
                "Increase --max-steps or lower --target-train-tokens."
            )
    min_steps = max(args.steps, target_token_steps)
    default_lr_total_steps = args.max_steps if args.max_steps is not None else min_steps
    lr_total_steps = (
        int(args.lr_total_steps) if args.lr_total_steps is not None else int(default_lr_total_steps)
    )
    lr_fn = _build_lr_schedule(
        base_lr=args.lr,
        schedule_name=args.lr_schedule,
        warmup_steps=args.lr_warmup_steps,
        min_ratio=args.lr_min_ratio,
        total_steps=lr_total_steps,
    )

    rng = jax.random.PRNGKey(args.seed)
    k_student, k_teacher, k_train, k_eval = jax.random.split(rng, 4)
    init_params = _make_initial_params(k_student, model_cfg=model_cfg)
    teacher_params = _make_initial_params(k_teacher, model_cfg=model_cfg)
    if args.data_source == "synthetic":
        train_tokens = _make_token_batches(
            k_train,
            num_batches=train_pool_batches,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            shift_start_batch=shift_start_batch,
            rare_inject_prob=args.train_rare_token_prob,
        )
        eval_tokens = _make_token_batches(
            k_eval,
            num_batches=args.eval_batches,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            shift_start_batch=0,
            rare_inject_prob=args.eval_rare_token_prob,
        )
    else:
        print(
            "Using real-world text stream dataset: "
            f"{ds_cfg.name} [{ds_cfg.config or 'default'}]"
        )
        if args.data_source == "hf_http":
            token_present = bool(ds_cfg.http_token_env and os.environ.get(ds_cfg.http_token_env))
            print(
                f"hf_http auth env {ds_cfg.http_token_env!r}: "
                f"{'set' if token_present else 'not set'}"
            )
            print(
                f"hf_http cache dir={ds_cfg.http_cache_dir!r}, "
                f"read={ds_cfg.http_cache_read}, write={ds_cfg.http_cache_write}"
            )
        if args.data_source == "hf_stream":
            train_tokens = _make_hf_stream_batches(
                ds_cfg,
                split=ds_cfg.train_split,
                seed=args.seed + 11,
                num_batches=train_pool_batches,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                max_docs=ds_cfg.train_max_docs,
            )

            def build_eval(split_name: str, seed_val: int):
                return _make_hf_stream_batches(
                    ds_cfg,
                    split=split_name,
                    seed=seed_val,
                    num_batches=args.eval_batches,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    vocab_size=args.vocab_size,
                    max_docs=ds_cfg.eval_max_docs,
                )

        else:
            train_tokens = _make_hf_http_batches(
                ds_cfg,
                split=ds_cfg.train_split,
                num_batches=train_pool_batches,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                max_docs=ds_cfg.train_max_docs,
            )

            def build_eval(split_name: str, seed_val: int):
                del seed_val
                return _make_hf_http_batches(
                    ds_cfg,
                    split=split_name,
                    num_batches=args.eval_batches,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    vocab_size=args.vocab_size,
                    max_docs=ds_cfg.eval_max_docs,
                )

        try:
            eval_tokens = build_eval(ds_cfg.eval_split, args.seed + 29)
        except Exception as exc:
            if args.dataset_eval_fallback_to_train and ds_cfg.eval_split != ds_cfg.train_split:
                print(
                    f"Eval split '{ds_cfg.eval_split}' failed ({exc}). "
                    f"Falling back to train split '{ds_cfg.train_split}'."
                )
                eval_tokens = build_eval(ds_cfg.train_split, args.seed + 47)
            else:
                raise

    step_csv = out_dir / "benchmark_steps.csv"
    summary_csv = out_dir / "benchmark_summary.csv"
    meta_json = out_dir / "run_meta.json"

    step_fieldnames = [
        "config",
        "hyperball_on",
        "grouped",
        "lora_hook_on",
        "step",
        "loss",
        "next_token_ce",
        "distill_kl",
        "step_ms",
        "tokens_per_s",
        "grad_norm",
        "update_norm",
        "learning_rate",
        "eval_loss",
        "eval_next_token_ce",
        "eval_distill_kl",
        "hyperball_angle_mean",
        "hyperball_radial_frac_mean",
    ]
    step_file = step_csv.open("w", newline="", encoding="utf-8")
    step_writer = csv.DictWriter(step_file, fieldnames=step_fieldnames)
    step_writer.writeheader()
    summary_rows: List[Dict[str, Any]] = []
    try:
        for cfg in configs:
            print(
                f"Starting config={cfg.name} "
                f"(eval_interval={args.eval_interval}, log_interval={args.log_interval}, "
                f"step_record_interval={args.step_record_interval}, "
                f"lr={args.lr:.6g}, lr_schedule={args.lr_schedule}, "
                f"lr_warmup_steps={args.lr_warmup_steps}, lr_min_ratio={args.lr_min_ratio:.4f}, "
                f"lr_total_steps={lr_total_steps})"
            )
            params = jax.tree.map(lambda z: jnp.array(z, copy=True), init_params)
            tx = _build_optimizer(
                cfg,
                params=params,
                learning_rate=lr_fn,
                wd=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
            )
            opt_state = tx.init(params)
            step_fn = _make_step_fn(
                tx=tx,
                teacher_params=teacher_params,
                lora_hook_on=cfg.lora_hook_on,
                model_cfg=model_cfg,
                objective_cfg=objective_cfg,
                grad_accum_steps=args.grad_accum_steps,
                causal_mask=mask,
            )
            eval_fn = _make_eval_fn(
                teacher_params=teacher_params,
                model_cfg=model_cfg,
                objective_cfg=objective_cfg,
                causal_mask=mask,
            )

            # JIT warmup (excluded from timing rows).
            for i in range(args.warmup_steps):
                params, opt_state, loss_val, ce_val, kl_val, grad_norm, update_norm = step_fn(
                    params, opt_state, train_tokens[i]
                )
                jax.block_until_ready(loss_val)
                jax.block_until_ready(ce_val)
                jax.block_until_ready(kl_val)
                jax.block_until_ready(grad_norm)
                jax.block_until_ready(update_norm)

            target_seconds = args.target_runtime_minutes * 60.0
            max_steps = args.max_steps
            run_t0 = time.perf_counter()

            loss_last = float("nan")
            ce_last = float("nan")
            kl_last = float("nan")
            best_eval_loss = float("inf")
            final_eval_loss = float("nan")
            step_ms_total = 0.0
            tokens_per_s_total = 0.0
            grad_norm_total = 0.0
            update_norm_total = 0.0
            learning_rate_total = 0.0
            learning_rate_last = float("nan")
            hb_last: Dict[str, float] = {}
            step_count = 0
            tokens_processed = 0

            while True:
                step = step_count
                token_idx = (args.warmup_steps + step) % train_pool_batches
                tokens = train_tokens[token_idx]
                t0 = time.perf_counter()
                params, opt_state, loss_val, ce_val, kl_val, grad_norm, update_norm = step_fn(
                    params, opt_state, tokens
                )
                jax.block_until_ready(loss_val)
                jax.block_until_ready(ce_val)
                jax.block_until_ready(kl_val)
                jax.block_until_ready(grad_norm)
                jax.block_until_ready(update_norm)
                step_ms = (time.perf_counter() - t0) * 1000.0
                tokens_per_s = (args.batch_size * args.seq_len) / max(step_ms / 1000.0, 1e-9)

                loss_scalar = float(loss_val)
                ce_scalar = float(ce_val)
                kl_scalar = float(kl_val)
                grad_norm_scalar = float(grad_norm)
                update_norm_scalar = float(update_norm)
                learning_rate_scalar = float(jnp.asarray(lr_fn(step)))
                hb_metrics = _aggregate_hyperball_metrics(opt_state)

                eval_loss = float("nan")
                eval_ce = float("nan")
                eval_kl = float("nan")
                should_eval = args.eval_interval > 0 and (
                    (step + 1) % args.eval_interval == 0 or step == (args.steps - 1)
                )
                if should_eval:
                    eval_loss_acc = 0.0
                    eval_ce_acc = 0.0
                    eval_kl_acc = 0.0
                    for eval_idx in range(args.eval_batches):
                        loss_eval, ce_eval, kl_eval = eval_fn(params, eval_tokens[eval_idx])
                        loss_eval = float(jax.block_until_ready(loss_eval))
                        ce_eval = float(jax.block_until_ready(ce_eval))
                        kl_eval = float(jax.block_until_ready(kl_eval))
                        eval_loss_acc += loss_eval
                        eval_ce_acc += ce_eval
                        eval_kl_acc += kl_eval
                    eval_loss = eval_loss_acc / float(args.eval_batches)
                    eval_ce = eval_ce_acc / float(args.eval_batches)
                    eval_kl = eval_kl_acc / float(args.eval_batches)
                    final_eval_loss = eval_loss
                    best_eval_loss = min(best_eval_loss, eval_loss)

                row = {
                    "config": cfg.name,
                    "hyperball_on": int(cfg.hyperball_on),
                    "grouped": int(cfg.grouped),
                    "lora_hook_on": int(cfg.lora_hook_on),
                    "step": step,
                    "loss": f"{loss_scalar:.8f}",
                    "next_token_ce": f"{ce_scalar:.8f}",
                    "distill_kl": f"{kl_scalar:.8f}",
                    "step_ms": f"{step_ms:.4f}",
                    "tokens_per_s": f"{tokens_per_s:.2f}",
                    "grad_norm": f"{grad_norm_scalar:.8f}",
                    "update_norm": f"{update_norm_scalar:.8f}",
                    "learning_rate": f"{learning_rate_scalar:.10f}",
                    "eval_loss": f"{eval_loss:.8f}",
                    "eval_next_token_ce": f"{eval_ce:.8f}",
                    "eval_distill_kl": f"{eval_kl:.8f}",
                    "hyperball_angle_mean": (
                        f"{hb_metrics.get('hyperball/angle_mean', float('nan')):.8f}"
                    ),
                    "hyperball_radial_frac_mean": (
                        f"{hb_metrics.get('hyperball/radial_frac_mean', float('nan')):.8f}"
                    ),
                }
                should_record_step = (
                    args.step_record_interval <= 1
                    or (step_count % args.step_record_interval == 0)
                    or should_eval
                    or step_count == 0
                )
                if should_record_step:
                    step_writer.writerow(row)
                    if step_count % 200 == 0:
                        step_file.flush()

                loss_last = loss_scalar
                ce_last = ce_scalar
                kl_last = kl_scalar
                step_ms_total += step_ms
                tokens_per_s_total += tokens_per_s
                grad_norm_total += grad_norm_scalar
                update_norm_total += update_norm_scalar
                learning_rate_total += learning_rate_scalar
                learning_rate_last = learning_rate_scalar
                hb_last = hb_metrics
                step_count += 1
                tokens_processed += tokens_per_step

                elapsed = time.perf_counter() - run_t0
                reached_min_steps = step_count >= min_steps
                reached_target_runtime = elapsed >= target_seconds if target_seconds > 0 else True
                reached_max_steps = (max_steps is not None) and (step_count >= max_steps)
                reached_token_target = (
                    (args.target_train_tokens is None)
                    or (tokens_processed >= args.target_train_tokens)
                )

                should_log = False
                if args.log_interval > 0 and (step_count % args.log_interval == 0):
                    should_log = True
                if should_eval:
                    should_log = True
                if step_count == 1:
                    should_log = True
                if should_log:
                    avg_step_ms = step_ms_total / max(step_count, 1)
                    avg_toks_per_s = tokens_per_s_total / max(step_count, 1)
                    runtime_min = elapsed / 60.0
                    eta_by_steps_min = (
                        max(min_steps - step_count, 0) * avg_step_ms / 60000.0
                    )
                    eta_by_runtime_min = (
                        max(target_seconds - elapsed, 0.0) / 60.0 if target_seconds > 0 else 0.0
                    )
                    eta_by_tokens_min = (
                        (
                            max(args.target_train_tokens - tokens_processed, 0)
                            / max(avg_toks_per_s, 1e-9)
                        )
                        / 60.0
                        if args.target_train_tokens is not None
                        else 0.0
                    )
                    eta_min = max(eta_by_steps_min, eta_by_runtime_min, eta_by_tokens_min)
                    progress_parts = [
                        f"[{cfg.name}]",
                        f"step={step_count}",
                        f"elapsed_min={runtime_min:.2f}",
                        f"eta_min={eta_min:.2f}",
                        f"train_loss={_fmt_metric(loss_scalar, 6)}",
                        f"train_ce={_fmt_metric(ce_scalar, 6)}",
                        f"train_kl={_fmt_metric(kl_scalar, 6)}",
                        f"avg_step_ms={_fmt_metric(avg_step_ms, 2)}",
                        f"avg_tok_s={_fmt_metric(avg_toks_per_s, 2)}",
                        f"grad_norm={_fmt_metric(grad_norm_scalar, 6)}",
                        f"update_norm={_fmt_metric(update_norm_scalar, 6)}",
                        f"lr={_fmt_metric(learning_rate_scalar, 8)}",
                    ]
                    if args.target_train_tokens is not None:
                        progress_parts.append(
                            f"tokens={tokens_processed}/{args.target_train_tokens}"
                        )
                    if should_eval:
                        progress_parts.append(f"val_loss={_fmt_metric(eval_loss, 6)}")
                        progress_parts.append(f"val_ce={_fmt_metric(eval_ce, 6)}")
                        progress_parts.append(f"val_kl={_fmt_metric(eval_kl, 6)}")
                        progress_parts.append(f"best_val={_fmt_metric(best_eval_loss, 6)}")
                    angle = hb_metrics.get("hyperball/angle_mean", float("nan"))
                    radial = hb_metrics.get("hyperball/radial_frac_mean", float("nan"))
                    if not math.isnan(angle):
                        progress_parts.append(f"hb_angle={_fmt_metric(angle, 6)}")
                    if not math.isnan(radial):
                        progress_parts.append(f"hb_radial={_fmt_metric(radial, 6)}")
                    print(" | ".join(progress_parts))

                if reached_max_steps and (
                    (not reached_target_runtime)
                    or (not reached_token_target)
                    or (not reached_min_steps)
                ):
                    warn_parts = [f"max_steps={max_steps} reached early:"]
                    if not reached_min_steps:
                        warn_parts.append(f"min_steps={min_steps} not met")
                    if not reached_target_runtime:
                        warn_parts.append(
                            f"target_runtime_minutes={args.target_runtime_minutes:.2f} not met"
                        )
                    if not reached_token_target and args.target_train_tokens is not None:
                        warn_parts.append(
                            f"target_train_tokens={args.target_train_tokens} not met"
                        )
                    print("WARNING: " + "; ".join(warn_parts))
                    break
                if reached_min_steps and reached_target_runtime and reached_token_target:
                    break
                if reached_max_steps:
                    break

        # Always produce a terminal eval snapshot for summary-level comparison.
        eval_loss_acc = 0.0
        eval_ce_acc = 0.0
        eval_kl_acc = 0.0
        for eval_idx in range(args.eval_batches):
            loss_eval, ce_eval, kl_eval = eval_fn(params, eval_tokens[eval_idx])
            loss_eval = float(jax.block_until_ready(loss_eval))
            ce_eval = float(jax.block_until_ready(ce_eval))
            kl_eval = float(jax.block_until_ready(kl_eval))
            eval_loss_acc += loss_eval
            eval_ce_acc += ce_eval
            eval_kl_acc += kl_eval
        final_eval_loss = eval_loss_acc / float(args.eval_batches)
        best_eval_loss = min(best_eval_loss, final_eval_loss)

        summary_rows.append(
            {
                "config": cfg.name,
                "hyperball_on": int(cfg.hyperball_on),
                "grouped": int(cfg.grouped),
                "lora_hook_on": int(cfg.lora_hook_on),
                "steps": step_count,
                "requested_min_steps": args.steps,
                "target_runtime_minutes": f"{args.target_runtime_minutes:.2f}",
                "tokens_processed": tokens_processed,
                "target_train_tokens": (
                    int(args.target_train_tokens) if args.target_train_tokens is not None else ""
                ),
                "final_loss": f"{loss_last:.8f}",
                "final_next_token_ce": f"{ce_last:.8f}",
                "final_distill_kl": f"{kl_last:.8f}",
                "final_eval_loss": f"{final_eval_loss:.8f}",
                "best_eval_loss": f"{best_eval_loss:.8f}",
                "avg_step_ms": f"{(step_ms_total / max(step_count, 1)):.4f}",
                "avg_tokens_per_s": f"{(tokens_per_s_total / max(step_count, 1)):.2f}",
                "avg_grad_norm": f"{(grad_norm_total / max(step_count, 1)):.8f}",
                "avg_update_norm": f"{(update_norm_total / max(step_count, 1)):.8f}",
                "avg_learning_rate": f"{(learning_rate_total / max(step_count, 1)):.10f}",
                "final_learning_rate": f"{learning_rate_last:.10f}",
                "final_hyperball_angle_mean": (
                    f"{hb_last.get('hyperball/angle_mean', float('nan')):.8f}"
                ),
                "final_hyperball_radial_frac_mean": (
                    f"{hb_last.get('hyperball/radial_frac_mean', float('nan')):.8f}"
                ),
            }
        )
    finally:
        step_file.flush()
        step_file.close()

    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "config",
            "hyperball_on",
            "grouped",
            "lora_hook_on",
            "steps",
            "requested_min_steps",
            "target_runtime_minutes",
            "tokens_processed",
            "target_train_tokens",
            "final_loss",
            "final_next_token_ce",
            "final_distill_kl",
            "final_eval_loss",
            "best_eval_loss",
            "avg_step_ms",
            "avg_tokens_per_s",
            "avg_grad_norm",
            "avg_update_norm",
            "avg_learning_rate",
            "final_learning_rate",
            "final_hyperball_angle_mean",
            "final_hyperball_radial_frac_mean",
        ],
    )

    meta = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "max_steps": args.max_steps,
        "target_runtime_minutes": args.target_runtime_minutes,
        "token_pool_batches": train_pool_batches,
        "tokens_per_step": tokens_per_step,
        "target_train_tokens": args.target_train_tokens,
        "target_token_steps": target_token_steps,
        "configs": [c.name for c in configs],
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "lr_total_steps": lr_total_steps,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "grad_accum_steps": args.grad_accum_steps,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "log_interval": args.log_interval,
        "step_record_interval": args.step_record_interval,
        "shift_start_frac": args.shift_start_frac,
        "train_rare_token_prob": args.train_rare_token_prob,
        "eval_rare_token_prob": args.eval_rare_token_prob,
        "distill_temperature": args.distill_temperature,
        "distill_weight": args.distill_weight,
        "label_smoothing": args.label_smoothing,
        "data_source": args.data_source,
        "dataset": (dataclasses.asdict(ds_cfg) if args.data_source != "synthetic" else None),
        "model": dataclasses.asdict(model_cfg),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
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
            f"- {r['config']}: final_loss={r['final_loss']}, "
            f"final_eval_loss={r['final_eval_loss']}, "
            f"avg_step_ms={r['avg_step_ms']}, avg_tokens_per_s={r['avg_tokens_per_s']}, "
            f"angle={angle}, radial={radial}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run realistic MODULUS ablation benchmarks with sequence-model distillation, "
            "distribution shift, and evaluation passes."
        )
    )
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--target-runtime-minutes", type=float, default=0.0)
    p.add_argument("--target-train-tokens", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=6)
    p.add_argument("--token-pool-batches", type=int, default=256)
    p.add_argument(
        "--configs",
        type=str,
        default="all",
        help="Comma-separated config names or 'all'.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--step-record-interval", type=int, default=1)
    p.add_argument("--shift-start-frac", type=float, default=0.5)
    p.add_argument("--train-rare-token-prob", type=float, default=0.03)
    p.add_argument("--eval-rare-token-prob", type=float, default=0.08)
    p.add_argument("--data-source", type=str, default="synthetic")
    p.add_argument("--dataset-name", type=str, default="JeanKaddour/minipile")
    p.add_argument("--dataset-config", type=str, default="default")
    p.add_argument("--dataset-train-split", type=str, default="train")
    p.add_argument("--dataset-eval-split", type=str, default="validation")
    p.add_argument("--dataset-text-keys", type=str, default="text,content,document")
    p.add_argument("--dataset-shuffle-buffer", type=int, default=10000)
    p.add_argument("--dataset-max-doc-tokens", type=int, default=512)
    p.add_argument("--dataset-train-max-docs", type=int, default=None)
    p.add_argument("--dataset-eval-max-docs", type=int, default=None)
    p.add_argument("--dataset-trust-remote-code", action="store_true")
    p.add_argument(
        "--dataset-eval-fallback-to-train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--dataset-rows-endpoint",
        type=str,
        default="https://datasets-server.huggingface.co/rows",
    )
    p.add_argument("--dataset-rows-page-size", type=int, default=100)
    p.add_argument("--dataset-http-max-retries", type=int, default=10)
    p.add_argument("--dataset-http-min-interval-sec", type=float, default=0.35)
    p.add_argument("--dataset-http-token-env", type=str, default="HF_TOKEN")
    p.add_argument(
        "--dataset-http-cache-dir",
        type=str,
        default="artifacts/datasets/hf_http_cache",
    )
    p.add_argument(
        "--dataset-http-cache-read",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--dataset-http-cache-write",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=("constant", "warmup_cosine"),
    )
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--lr-min-ratio", type=float, default=0.10)
    p.add_argument("--lr-total-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--distill-temperature", type=float, default=1.5)
    p.add_argument("--distill-weight", type=float, default=0.6)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    if argv is None:
        # Colab/Jupyter kernels populate sys.argv with launcher args; ignore unknowns.
        args, unknown = p.parse_known_args()
        if unknown:
            print(f"Ignoring unknown launcher args: {unknown}")
        return args
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
