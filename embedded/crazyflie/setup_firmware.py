#!/usr/bin/env python3
"""Fetch the pinned firmware source and apply the RAYA integration patch."""
import argparse
from pathlib import Path
import subprocess

REVISION = "de14ff9f08ae80cbb36ffd37859efc2bedfbf6bf"
URL = "https://github.com/moisesmata/crazyflie-firmware.git"

def run(*args):
    subprocess.run(args, check=True)

def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=here.parents[1] / "third_party/crazyflie-firmware")
    args = parser.parse_args()
    dest = args.directory.resolve()
    if not dest.exists():
        run("git", "clone", "--no-checkout", URL, str(dest))
        run("git", "-C", str(dest), "checkout", "--detach", REVISION)
    revision = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected firmware {REVISION}, found {revision}; use a new directory.")
    run("git", "-C", str(dest), "submodule", "update", "--init", "--recursive")
    patch = str(here / "patches/firmware.patch")
    applied = subprocess.run(["git", "-C", str(dest), "apply", "--reverse", "--check", patch], capture_output=True).returncode == 0
    if not applied:
        run("git", "-C", str(dest), "apply", "--check", patch)
        run("git", "-C", str(dest), "apply", patch)
    print(f"RAYA firmware dependency ready: {dest}")

if __name__ == "__main__":
    main()
