#!/usr/bin/env python3
"""Run the frozen RAYA simulation grids from the paper (standard library only)."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


PAPER = {
    "quadrotor": {"episodes": 4200, "successes": 1715, "survival_percent": 40.83,
                  "mean_authority": 0.698, "survivor_tracking_m": 0.390},
    "f1tenth": {"episodes": 3000, "successes": 1462, "survival_percent": 48.73,
                "mean_authority": 0.271, "survivor_tracking_m": 0.161},
}


def scenarios(selected="all"):
    result = []

    def add(robot, family, cell, seed, options):
        result.append({"episode": f"{robot}_{len(result):04d}", "platform": robot,
                       "family": family, "cell": cell, "seed": seed,
                       "args": [*options, "--seed", str(seed)]})

    if selected in ("all", "quadrotor"):
        families = [
            ("figure8", "figure8", "shift_flip", "1", "0"),
            ("circle", "circle", "shift_flip", "1", "0"),
            ("line_y", "line_y", "shift_flip", "1", "0"),
            ("star", "star", "shift_flip", "1", "0"),
            ("turbulent_wind", "figure8", "shift_turb", "1", "0"),
            ("heavy_plant", "figure8", "default", "1.2", "0.2"),
        ]
        for family, trajectory, wind_profile, mass, drag in families:
            for wind in range(6, 13):
                for seed in range(100):
                    add("quadrotor", family, f"{family}_wind{wind}", seed,
                        ["--trajectory", trajectory, "--wind", str(wind),
                         "--wind-profile", wind_profile, "--mass-scale", mass,
                         "--drag", drag, "--steps", "281"])
    if selected in ("all", "f1tenth"):
        for mu in (0.20, 0.25, 0.30, 0.35, 0.40):
            base = ["--friction", f"{mu:g}", "--steps", "400"]
            for period in (3.0, 4.0, 5.0):
                for index in range(100):
                    phase = (index + 0.5) * period / 100
                    add("f1tenth", "strips", f"strips_p{period:g}_mu{mu:g}", 91000 + index,
                        [*base, "--friction-profile", "strips", "--patch-period", f"{period:g}",
                         "--patch-phase", f"{phase:.12g}", "--patch-time", "4"])
            for count in (4, 6):
                for index in range(100):
                    phase = (index + 0.5) * 2.0 * math.pi / 100
                    add("f1tenth", "random", f"random_n{count}_mu{mu:g}", 91000 + index,
                        [*base, "--friction-profile", "random", "--patch-count", str(count),
                         "--patch-phase", f"{phase:.12g}", "--patch-time", "4"])
            for index in range(100):
                patch_time = 4.0 + (index + 0.5) * (20.5 - 4.0) / 100
                add("f1tenth", "single_patch", f"single_patch_mu{mu:g}", 91000 + index,
                    [*base, "--friction-profile", "single_patch", "--patch-phase", "0",
                     "--patch-time", f"{patch_time:.12g}"])
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows):
    valid = [r for r in rows if not r["error"]]
    survivors = [r for r in valid if r["success"] == 1]
    return {
        "episodes": len(rows), "valid_episodes": len(valid), "errors": len(rows) - len(valid),
        "successes": len(survivors),
        "survival_percent": 100.0 * len(survivors) / len(rows),
        "mean_authority": statistics.fmean(r["mean_authority"] for r in valid) if valid else None,
        "survivor_tracking_m": statistics.fmean(r["tracking_m"] for r in survivors) if survivors else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--build-dir", type=Path, default=root / "build")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--platform", choices=("all", "quadrotor", "f1tenth"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--constraint-stages", type=int, default=1)
    parser.add_argument("--check-paper", action="store_true", help="Fail if rounded RAYA aggregate results differ")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.constraint_stages <= 19:
        parser.error("Use 1..32 workers and 1..19 constraint stages")
    if args.check_paper and args.constraint_stages != 1:
        parser.error("Paper-selected results use --constraint-stages 1")
    cases = scenarios(args.platform)
    build = args.build_dir.resolve()
    binaries = {name: build / ("raya_" + name) for name in {c["platform"] for c in cases}}
    for executable in binaries.values():
        if not executable.is_file():
            parser.error(f"Missing executable: {executable}")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output path already exists; choose a new directory")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    source_hashes = {}
    for parent in ("examples", "third_party/tinympc", "third_party/eigen"):
        for path in sorted((root / parent).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source_hashes[str(path.relative_to(root))] = sha256(path)
    source_hashes["CMakeLists.txt"] = sha256(root / "CMakeLists.txt")
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "controller": "RAYA",
        "constraint_stages": args.constraint_stages, "workers": args.workers,
        "environment": {"system": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version()},
        "binary_sha256": {name: sha256(path) for name, path in binaries.items()},
        "source_sha256": source_hashes, "paper_selected_targets": PAPER,
        "cases": cases,
    }
    cache = build / "CMakeCache.txt"
    manifest["cmake_configuration"] = [line for line in cache.read_text().splitlines()
        if re.match(r"CMAKE_(BUILD_TYPE|CXX_COMPILER|CXX_FLAGS|OSX_ARCHITECTURES)[^=]*=", line)] if cache.exists() else []
    # Freeze the complete protocol and artifact hashes before any episode runs.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(case):
        command = [str(binaries[case["platform"]]), *case["args"],
                   "--constraint-stages", str(args.constraint_stages)]
        record = {key: case[key] for key in ("episode", "platform", "family", "cell", "seed")}
        record.update(success=-1, steps=0, tracking_m="", mean_authority="", mean_authority_switch="", error="")
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=output)
            (output / "logs" / (case["episode"] + ".txt")).write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"Executable exited {completed.returncode}")
            metrics = dict(re.findall(r"\b([a-z][a-z0-9_]*)\s*[:=]\s*([^\s]+)", completed.stdout))
            mapping = {"success": "success", "steps": "steps", "mean_authority": "mean_authority_w",
                       "mean_authority_switch": "mean_authority_switch",
                       "tracking_m": "mean_xy_err" if case["platform"] == "quadrotor" else "mean_tube_err"}
            values = {key: float(metrics[value]) for key, value in mapping.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError("Non-finite summary metric")
            if values["success"] not in (0, 1) or not 0 <= values["mean_authority"] <= 1:
                raise RuntimeError("Invalid success or authority metric")
            maximum_steps = 281 if case["platform"] == "quadrotor" else 400
            if not 1 <= values["steps"] <= maximum_steps or (values["success"] and values["steps"] != maximum_steps):
                raise RuntimeError("Incomplete episode")
            record.update(values)
            record["success"], record["steps"] = int(values["success"]), int(values["steps"])
        except (OSError, subprocess.SubprocessError, KeyError, ValueError, RuntimeError) as error:
            record["error"] = str(error)
        return record

    started = time.monotonic()
    rows = []
    fields = ["episode", "platform", "family", "cell", "seed", "success", "steps", "tracking_m",
              "mean_authority", "mean_authority_switch", "error"]
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, case) for case in cases]
            for future in as_completed(futures):
                record = future.result()
                rows.append(record)
                writer.writerow(record)
                if len(rows) % 100 == 0 or len(rows) == len(cases):
                    handle.flush()
                    print(f"{len(rows)}/{len(cases)} episodes, {sum(bool(r['error']) for r in rows)} errors, "
                          f"{time.monotonic() - started:.1f}s", flush=True)
    rows.sort(key=lambda row: row["episode"])
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"controller": "RAYA", "constraint_stages": args.constraint_stages,
               "elapsed_seconds": time.monotonic() - started, "platforms": {}, "families": {}, "cells": {}}
    for grouping, key in (("platforms", "platform"), ("families", "family"), ("cells", "cell")):
        groups = defaultdict(list)
        for row in rows:
            group = row[key] if grouping == "platforms" else row["platform"] + "/" + row[key]
            groups[group].append(row)
        summary[grouping] = {name: aggregate(subset) for name, subset in sorted(groups.items())}
    if args.constraint_stages == 1:
        summary["paper_comparison"] = {}
        for name, measured in summary["platforms"].items():
            summary["paper_comparison"][name] = {
                key: {"expected": expected, "measured": measured[key],
                      "matches": measured[key] is not None and round(measured[key],
                          2 if key == "survival_percent" else 3) == expected}
                for key, expected in PAPER[name].items()
            }
    summary["episode_csv_sha256"] = sha256(output / "episodes.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, metrics in summary["platforms"].items():
        print(f"{name}: {metrics['successes']}/{metrics['episodes']} = {metrics['survival_percent']:.5f}%; "
              f"authority={metrics['mean_authority']}; survivor tracking={metrics['survivor_tracking_m']}")
    print(f"Saved protocol, episode metrics, logs, and summary to {output}")
    failed = any(row["error"] for row in rows)
    if args.check_paper:
        failed |= any(not metric["matches"] for checks in summary["paper_comparison"].values() for metric in checks.values())
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
