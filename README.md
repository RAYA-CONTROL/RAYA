# RAYA

**Recoverability-Aware Yielding of Authority**

[![Paper](docs/assets/paper.svg)](https://arxiv.org/abs/2609.21690) [![Project Page](docs/assets/project-page.svg)](https://raya-control.github.io/) ![YouTube](docs/assets/youtube.svg)

RAYA places a learned recoverability margin inside model predictive control and uses a bounded learned scheduler to adjust task weights as recovery becomes difficult.

This repository contains standalone **quadrotor** and **F1TENTH** simulations, plus the **Crazyflie 2.1 Brushless** controller. The simulations include frozen model weights, authority policies, model parameters, and the required TinyMPC and Eigen source. No training, ROS, or external model downloads are needed to run them.

[Simulation guide](docs/simulations.md) · [Crazyflie code](embedded/crazyflie/README.md)

## Build and run

Requirements: a C++17 compiler and CMake 3.20 or newer. Python 3.8 or newer is used for the checks and experiment runner.

```sh
git clone https://github.com/RAYA-CONTROL/RAYA.git
cd RAYA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/raya_quadrotor --trajectory figure8 --wind 8 --csv quadrotor.csv
./build/raya_f1tenth --friction-profile strips --friction 0.30 --csv f1tenth.csv

ctest --test-dir build --output-on-failure
```

Each executable prints an episode summary and optionally writes a trajectory CSV. Its frozen controller is built into the executable, so it also runs from another working directory. Use `--help` for all supported experiment options.

To build without Python-based checks, configure with `-DBUILD_TESTING=OFF`.

## Contents

| Directory | Purpose |
| --- | --- |
| `examples/quadrotor/` | Nonlinear 12-state quadrotor, wind disturbances, recoverability constraints, and learned authority scheduling |
| `examples/f1tenth/` | Bicycle-model trajectory tracking with hidden friction changes and causal recoverability features |
| `embedded/crazyflie/` | Onboard RAYA controller, float32 model and cache bank, build files, and firmware integration patch |
| `third_party/` | Solver and linear algebra dependencies with their original notices |
| `tests/` | Closed-loop checks and frozen-model integrity checks |

Both simulations and the embedded controller default to the paper-selected first predicted constraint stage (`K=1`). Simulations expose `--constraint-stages`; firmware exposes `ACTIVE_HORIZON` at build time. Their solver iteration budgets and plant models are documented separately.

## Experiments

Run a small collection of RAYA episodes and save each trajectory and summary:

```sh
python3 examples/run_experiments.py --build-dir build --output results
```

See [the simulation guide](docs/simulations.md) for disturbances, units, outputs, and evaluation definitions. Hardware compilation and deployment are described in [the Crazyflie README](embedded/crazyflie/README.md).

## Reproduce the paper simulations

```sh
python3 examples/reproduce_paper.py --build-dir build --output results/paper --workers 4 --check-paper
```

This runs all 4,200 quadrotor and 3,000 F1TENTH RAYA episodes. It saves the frozen protocol, source and binary hashes, per-episode metrics, console logs, and a comparison with the paper. Use a new output directory for each run. See [paper reproduction](docs/paper-reproduction.md) for the measured results and scope.


## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. The project title is *RAYA: Learning Where and When to Intervene for Robot Recovery*.

RAYA simulation code and repository utilities use the [MIT license](LICENSE). The Crazyflie controller retains its [GPLv3 license](embedded/crazyflie/LICENSE). Third-party notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## BibTeX

```bibtex
@misc{mahajan2026raya,
  title         = {{RAYA}: Learning Where and When to Intervene for Robot Recovery},
  author        = {Ishaan Mahajan and Charles Chen and Frederike D{\"u}mbgen and Brian Plancher},
  year          = {2026},
  eprint        = {2609.21690},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2609.21690},
  url           = {https://arxiv.org/abs/2609.21690}
}
```
