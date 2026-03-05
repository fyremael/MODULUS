# Artifacts Layout

Benchmark outputs are written to:

- `artifacts/benchmarks/<timestamp>/benchmark_steps.csv`
- `artifacts/benchmarks/<timestamp>/benchmark_summary.csv`
- `artifacts/benchmarks/<timestamp>/run_meta.json`

## Generate benchmark artifacts

```bash
python scripts/run_benchmarks.py
```

## Build benchmark report

```bash
python scripts/build_benchmark_report.py
```

With plots (if `matplotlib` is installed):

```bash
python scripts/build_benchmark_report.py --with-plots
```

## Quick run (faster)

```bash
python scripts/run_benchmarks.py --steps 20 --warmup-steps 3 --batch-size 32 --width 64
```
