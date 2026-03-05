from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_float(value: str) -> float:
    if value is None:
        return float("nan")
    s = value.strip().lower()
    if s in {"", "nan", "none"}:
        return float("nan")
    return float(s)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latest_benchmark_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No benchmark run directories found under: {root}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _baseline_row(rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("config") == "baseline":
            return row
    return None


def _fmt(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, numbers.Real):
        try:
            if math.isnan(float(value)):
                return None
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _build_rankings(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    baseline = _baseline_row(summary_rows)
    base_loss = _parse_float(baseline["final_loss"]) if baseline else float("nan")
    base_ms = _parse_float(baseline["avg_step_ms"]) if baseline else float("nan")

    parsed: List[Dict[str, Any]] = []
    for row in summary_rows:
        final_loss = _parse_float(row["final_loss"])
        avg_step_ms = _parse_float(row["avg_step_ms"])
        loss_delta = final_loss - base_loss if not math.isnan(base_loss) else float("nan")
        speed_ratio = (
            (avg_step_ms / base_ms) if (not math.isnan(base_ms) and base_ms > 0.0) else float("nan")
        )
        parsed.append(
            {
                "config": row["config"],
                "hyperball_on": int(row["hyperball_on"]),
                "grouped": int(row["grouped"]),
                "lora_hook_on": int(row["lora_hook_on"]),
                "final_loss": final_loss,
                "avg_step_ms": avg_step_ms,
                "loss_delta_vs_baseline": loss_delta,
                "speed_ratio_vs_baseline": speed_ratio,
                "final_hyperball_angle_mean": _parse_float(
                    row.get("final_hyperball_angle_mean", "nan")
                ),
                "final_hyperball_radial_frac_mean": _parse_float(
                    row.get("final_hyperball_radial_frac_mean", "nan")
                ),
            }
        )

    parsed.sort(key=lambda x: (x["final_loss"], x["avg_step_ms"]))
    return parsed


def _best_tradeoff(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Minimize loss + normalized time with equal weight.
    loss_vals = [r["final_loss"] for r in rows if not math.isnan(r["final_loss"])]
    ms_vals = [r["avg_step_ms"] for r in rows if not math.isnan(r["avg_step_ms"])]
    loss_min, loss_max = min(loss_vals), max(loss_vals)
    ms_min, ms_max = min(ms_vals), max(ms_vals)

    def norm(v: float, lo: float, hi: float) -> float:
        if math.isnan(v):
            return 1.0
        if hi <= lo:
            return 0.0
        return (v - lo) / (hi - lo)

    best_row = rows[0]
    best_score = float("inf")
    for r in rows:
        score = norm(r["final_loss"], loss_min, loss_max) + norm(r["avg_step_ms"], ms_min, ms_max)
        if score < best_score:
            best_score = score
            best_row = r
    return {"config": best_row["config"], "score": best_score}


def _emit_plots(
    out_dir: Path, step_rows: Sequence[Dict[str, str]], ranking_rows: Sequence[Dict[str, Any]]
) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {"status": "skipped", "reason": "matplotlib_not_available"}

    # Loss curves
    by_config: Dict[str, List[Tuple[int, float]]] = {}
    for row in step_rows:
        cfg = row["config"]
        step = int(row["step"])
        loss = _parse_float(row["loss"])
        by_config.setdefault(cfg, []).append((step, loss))

    for cfg in by_config:
        by_config[cfg].sort(key=lambda x: x[0])

    fig1 = plt.figure(figsize=(8, 5))
    ax1 = fig1.add_subplot(111)
    for cfg, seq in by_config.items():
        xs = [x for x, _ in seq]
        ys = [y for _, y in seq]
        ax1.plot(xs, ys, label=cfg)
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    loss_plot = out_dir / "loss_curves.png"
    fig1.tight_layout()
    fig1.savefig(loss_plot, dpi=150)
    plt.close(fig1)

    # Speed vs quality
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111)
    xs = [r["avg_step_ms"] for r in ranking_rows]
    ys = [r["final_loss"] for r in ranking_rows]
    labels = [r["config"] for r in ranking_rows]
    ax2.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        ax2.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax2.set_title("Speed vs Final Loss")
    ax2.set_xlabel("Average Step Time (ms)")
    ax2.set_ylabel("Final Loss")
    ax2.grid(alpha=0.3)
    tradeoff_plot = out_dir / "speed_vs_loss.png"
    fig2.tight_layout()
    fig2.savefig(tradeoff_plot, dpi=150)
    plt.close(fig2)

    return {
        "status": "ok",
        "loss_curves": str(loss_plot),
        "speed_vs_loss": str(tradeoff_plot),
    }


def _write_markdown(
    out_path: Path,
    run_dir: Path,
    ranking_rows: Sequence[Dict[str, Any]],
    best_tradeoff: Dict[str, Any],
    plot_info: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Configs: `{len(ranking_rows)}`")
    lines.append("")
    lines.append("## Ranking (lower loss first)")
    lines.append("")
    lines.append(
        "| config | final_loss | avg_step_ms | loss_delta_vs_baseline | speed_ratio_vs_baseline |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for row in ranking_rows:
        lines.append(
            "| {config} | {loss} | {ms} | {dloss} | {sratio} |".format(
                config=row["config"],
                loss=_fmt(row["final_loss"], 8),
                ms=_fmt(row["avg_step_ms"], 4),
                dloss=_fmt(row["loss_delta_vs_baseline"], 8),
                sratio=_fmt(row["speed_ratio_vs_baseline"], 4),
            )
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"- Best balanced config (loss/speed heuristic): `{best_tradeoff['config']}` "
        f"(score={best_tradeoff['score']:.4f})"
    )
    lines.append("")
    lines.append("## Plot Outputs")
    lines.append("")
    if plot_info.get("status") == "ok":
        lines.append(f"- Loss curves: `{plot_info['loss_curves']}`")
        lines.append(f"- Speed vs loss: `{plot_info['speed_vs_loss']}`")
    else:
        lines.append(f"- Plots skipped: `{plot_info.get('reason', 'unknown')}`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    runs_root = root / "artifacts" / "benchmarks"
    run_dir = Path(args.run_dir) if args.run_dir else _latest_benchmark_dir(runs_root)

    summary_csv = run_dir / "benchmark_summary.csv"
    steps_csv = run_dir / "benchmark_steps.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing file: {summary_csv}")
    if not steps_csv.exists():
        raise FileNotFoundError(f"Missing file: {steps_csv}")

    summary_rows = _load_csv(summary_csv)
    step_rows = _load_csv(steps_csv)
    ranking_rows = _build_rankings(summary_rows)
    best = _best_tradeoff(ranking_rows)
    plot_info = (
        _emit_plots(run_dir, step_rows, ranking_rows)
        if args.with_plots
        else {
            "status": "skipped",
            "reason": "plots_disabled",
        }
    )

    md_path = run_dir / "benchmark_report.md"
    json_path = run_dir / "benchmark_report.json"
    _write_markdown(md_path, run_dir, ranking_rows, best, plot_info)

    payload = {
        "run_dir": str(run_dir),
        "ranked_configs": ranking_rows,
        "best_tradeoff": best,
        "plots": plot_info,
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    if plot_info.get("status") == "ok":
        print(f"Wrote: {plot_info['loss_curves']}")
        print(f"Wrote: {plot_info['speed_vs_loss']}")
    else:
        print(f"Plots: skipped ({plot_info.get('reason', 'unknown')})")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build benchmark report from CSV artifacts.")
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Benchmark run directory. Defaults to latest artifacts/benchmarks/* directory.",
    )
    p.add_argument(
        "--with-plots",
        action="store_true",
        help="Generate PNG plots if matplotlib is available.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
