from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _style_cmd(targets: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "E,F,I",
        "--ignore",
        "E501",
        *targets,
    ]


def run_staged() -> int:
    targets = ["modulus/optim/presets.py", "scripts"]
    proc = _run(_style_cmd(targets))
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def run_advisory(out_path: Path) -> int:
    targets = ["modulus", "scripts"]
    proc = _run(_style_cmd(targets))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    payload.append("# Ruff Style Advisory (E,F,I; E501 ignored)\n")
    payload.append(f"Command: {' '.join(_style_cmd(targets))}\n")
    payload.append(f"Exit code: {proc.returncode}\n\n")
    if proc.stdout:
        payload.append("## Output\n\n")
        payload.append(proc.stdout)
    if proc.stderr:
        payload.append("\n## STDERR\n\n")
        payload.append(proc.stderr)
    out_path.write_text("".join(payload), encoding="utf-8")

    if proc.returncode == 0:
        print("Advisory style check: clean")
    else:
        print(f"Advisory style check: findings captured at {out_path}")
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run staged style lint phases.")
    p.add_argument(
        "--phase",
        choices=["staged", "advisory"],
        default="staged",
        help="staged: blocking strict-style on curated scope; advisory: full repo findings report.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/lint/style_advisory.txt"),
        help="Output path for advisory report.",
    )
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.phase == "staged":
        return run_staged()
    return run_advisory(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
