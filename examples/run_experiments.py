#!/usr/bin/env python3
"""Run a small suite of standalone RAYA simulations using only the standard library."""

import argparse
import csv
from pathlib import Path
import re
import subprocess


def parse_summary(text):
    return dict(re.findall(r"\b([a-z][a-z0-9_]*)\s*[:=]\s*([^\s]+)", text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--build-dir", type=Path, default=root / "build")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("quadrotor_figure8", "quadrotor", ["--trajectory", "figure8", "--wind", "8"]),
        ("quadrotor_circle", "quadrotor", ["--trajectory", "circle", "--wind", "10", "--seed", "7"]),
        ("quadrotor_turbulence", "quadrotor", ["--wind-profile", "shift_turb", "--wind", "9", "--seed", "5"]),
        ("f1tenth_strips", "f1tenth", ["--friction-profile", "strips", "--friction", "0.30"]),
        ("f1tenth_single_patch", "f1tenth", ["--friction-profile", "single_patch", "--friction", "0.25", "--patch-time", "8"]),
        ("f1tenth_random", "f1tenth", ["--friction-profile", "random", "--friction", "0.35", "--patch-count", "4", "--patch-phase", "1.1"]),
    ]
    for _, platform, _ in cases:
        executable = build / ("raya_" + platform)
        if not executable.is_file():
            parser.error(f"Missing executable: {executable}. Build the simulations first.")
    destinations = [output / "summary.csv"]
    for name, _, _ in cases:
        destinations.extend([output / (name + ".csv"), output / (name + ".txt")])
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        parser.error("Output files already exist; select a new --output directory.")
    records = []
    for name, platform, options in cases:
        command = [str(build / ("raya_" + platform)), *options, "--csv", str(output / (name + ".csv"))]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        (output / (name + ".txt")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise SystemExit(f"{name} failed with exit code {result.returncode}; see {name}.txt")
        metrics = parse_summary(result.stdout)
        records.append({"experiment": name, "platform": platform, **metrics})
        print(f"{name}: success={metrics.get('success', '?')}, steps={metrics.get('steps', '?')}")
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved trajectories and summary to {output}")


if __name__ == "__main__":
    main()
