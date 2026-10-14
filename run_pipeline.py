"""
Full AeroSurrogate pipeline runner.

Steps:
  1  CFD simulations   (simulations/pipeline.py)
  2  Build dataset     (training/build_dataset.py)
  3  Train ForceNet    (training/train_force_net.py)
  4  Train FieldNet    (training/train_field_net.py)

Usage:
  python run_pipeline.py              # run all steps
  python run_pipeline.py --from 2     # start at step 2 (skip CFD)
  python run_pipeline.py --only 3 4   # run only steps 3 and 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    (1, "CFD Simulations",   ROOT / "simulations" / "pipeline.py"),
    (2, "Build Dataset",     ROOT / "training"     / "build_dataset.py"),
    (3, "Train ForceNet",    ROOT / "training"     / "train_force_net.py"),
    (4, "Train FieldNet",    ROOT / "training"     / "train_field_net.py"),
]

WIDTH = 70


def banner(title: str, char: str = "=") -> None:
    print(f"\n{char * WIDTH}")
    print(f"  {title}")
    print(f"{char * WIDTH}\n", flush=True)


def run_step(num: int, name: str, script: Path) -> float:
    banner(f"STEP {num} — {name}")
    print(f"Script : {script.relative_to(ROOT)}")
    print(f"Started: {time.strftime('%H:%M:%S')}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    print(flush=True)
    if result.returncode != 0:
        print(f"{'!' * WIDTH}")
        print(f"  STEP {num} FAILED (exit code {result.returncode})")
        print(f"{'!' * WIDTH}", flush=True)
        sys.exit(result.returncode)

    print(f"  Step {num} finished in {elapsed:.1f} s ({elapsed/60:.1f} min)")
    return elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full AeroSurrogate pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {n}  {name}" for n, name, _ in STEPS),
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_step", type=int, default=1, metavar="N",
        help="Start from step N (default: 1)",
    )
    group.add_argument(
        "--only", dest="only_steps", type=int, nargs="+", metavar="N",
        help="Run only the listed steps",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.only_steps:
        selected = [s for s in STEPS if s[0] in args.only_steps]
    else:
        selected = [s for s in STEPS if s[0] >= args.from_step]

    if not selected:
        print("No steps selected. Exiting.")
        sys.exit(0)

    banner("AeroSurrogate Pipeline", char="#")
    print(f"Python : {sys.executable}")
    print(f"Root   : {ROOT}")
    print(f"Steps  : {', '.join(str(s[0]) for s in selected)}")

    wall_start = time.time()
    timings: list[tuple[str, float]] = []

    for num, name, script in selected:
        elapsed = run_step(num, name, script)
        timings.append((name, elapsed))

    total = time.time() - wall_start
    banner("Pipeline complete", char="#")
    for name, t in timings:
        print(f"  {name:<30} {t:7.1f} s  ({t/60:.1f} min)")
    print(f"  {'TOTAL':<30} {total:7.1f} s  ({total/60:.1f} min)")
    print()


if __name__ == "__main__":
    main()
