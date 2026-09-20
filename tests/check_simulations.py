#!/usr/bin/env python3
"""Check standalone execution, closed-loop telemetry, replay, and invalid inputs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrotor", type=Path, required=True)
    parser.add_argument("--f1tenth", type=Path, required=True)
    args = parser.parse_args()
    binaries = {name: getattr(args, name).resolve() for name in ("quadrotor", "f1tenth")}
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "tests/model_hashes.json").read_text())
    for relative, expected in manifest.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        require(actual == expected, f"Frozen model changed: {relative}")
    print(f"PASS: {len(manifest)} frozen model assets")

    with tempfile.TemporaryDirectory(prefix="raya-check-") as temporary:
        cwd = Path(temporary)

        def invoke(platform, options):
            return subprocess.run([str(binaries[platform]), *options], cwd=cwd, text=True,
                                  capture_output=True, timeout=90)

        def episode(platform, name, options, maximum_steps):
            path = cwd / (name + ".csv")
            result = invoke(platform, [*options, "--csv", str(path)])
            require(result.returncode == 0, f"{name}: {result.stderr}")
            require("RAYA" in result.stdout, f"{name}: missing controller identity")
            require(re.search(r"\bsuccess\s*[:=]\s*[01]\b", result.stdout), f"{name}: missing success metric")
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(0 < len(rows) <= maximum_steps, f"{name}: invalid episode length")
            dt = 0.05 if platform == "quadrotor" else 0.1
            for step, row in enumerate(rows):
                require(None not in row and all(value is not None for value in row.values()), f"{name}: malformed CSV")
                values = {key: float(value) for key, value in row.items()}
                require(all(math.isfinite(value) for value in values.values()), f"{name}: non-finite telemetry")
                require(values["step"] == step, f"{name}: skipped or duplicate step")
                require(abs(values["time"] - (step + 1) * dt) < 1e-9, f"{name}: incorrect timestamp")
                require(0 <= values["authority"] <= 1, f"{name}: authority outside bounds")
                if platform == "quadrotor":
                    require(all(abs(values[f"u{i}"]) <= 0.500001 for i in range(4)), f"{name}: actuator bounds")
                else:
                    require(abs(values["acceleration"]) <= 3.000001, f"{name}: acceleration bounds")
                    require(abs(values["steering_rate"]) <= 2.500001, f"{name}: steering bounds")
            print(f"PASS: {name}, {len(rows)} updates")
            return rows

        first = episode("quadrotor", "quadrotor_figure8", ["--wind", "8", "--seed", "0"], 281)
        replay = episode("quadrotor", "quadrotor_replay", ["--wind", "8", "--seed", "0", "--constraint-stages", "1"], 281)
        require(len(first) == len(replay), "Replay length changed")
        for a, b in zip(first, replay):
            require({k: v for k, v in a.items() if k != "solve_ms"} ==
                    {k: v for k, v in b.items() if k != "solve_ms"}, "Seeded replay changed")
        require(invoke("quadrotor", ["--seed", "2147483647", "--steps", "1"]).returncode == 0, "Maximum seed failed")
        for trajectory in ("circle", "line_y", "star", "hover"):
            episode("quadrotor", "quadrotor_" + trajectory, ["--trajectory", trajectory, "--wind", "8"], 281)
        episode("quadrotor", "quadrotor_turbulence", ["--wind-profile", "shift_turb", "--wind", "9"], 281)
        episode("quadrotor", "quadrotor_plant_shift", ["--wind-profile", "default", "--mass-scale", "1.2", "--drag", "0.2"], 281)
        for profile in ("strips", "single_patch", "random", "none"):
            episode("f1tenth", "f1tenth_" + profile,
                    ["--friction-profile", profile, "--friction", "0.30", "--patch-count", "4"], 400)

        for platform in binaries:
            require(invoke(platform, ["--help"]).returncode == 0, "Help failed")
            for options in (["--unknown", "1"], ["--steps", "0"], ["--steps", "1.5"],
                            ["--constraint-stages", "0"], ["--constraint-stages", "20"],
                            ["--constraint-stages", "1.5"], ["--steps"], ["--seed", "nan"], ["--seed", "3junk"],
                            ["--mass-scale", "0"], ["--method", "unknown"]):
                require(invoke(platform, options).returncode != 0, f"Accepted invalid input: {options}")
            require(invoke(platform, ["--csv", str(cwd)]).returncode != 0, "Accepted directory as CSV file")
        require(invoke("quadrotor", ["--wind", "inf"]).returncode != 0, "Accepted infinite wind")
        require(invoke("f1tenth", ["--friction", "1.5"]).returncode != 0, "Accepted invalid friction")
        print("PASS: CLI validation and output errors")


if __name__ == "__main__":
    main()
