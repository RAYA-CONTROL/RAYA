# Standalone simulations

Both examples run the frozen RAYA controller in a closed simulation loop. A disturbance value belongs to the plant; it is not passed to the learned authority policy. The controller uses the state estimate, reference, recoverability margin, and causal history.

## Quadrotor

```sh
./build/raya_quadrotor --trajectory figure8 --wind 8 --seed 0 --csv flight.csv
./build/raya_quadrotor --trajectory circle --wind 10 --seed 7
./build/raya_quadrotor --trajectory figure8 --wind-profile shift_turb --wind 9
./build/raya_quadrotor --trajectory figure8 --wind-profile default --mass-scale 1.2 --drag 0.2
```

The plant has 12 states and four control inputs. The simulation runs at 20 Hz for 281 steps by default. MPC uses a 20-state horizon, a 30-iteration ADMM cap, fixed penalty `rho=5`, and a learned constraint at the first predicted stage (`K=1`). The frozen recoverability network uses six features. The 11-coefficient authority policy produces a bounded scalar and applies smoothing of 0.70.

| Option | Meaning | Default |
| --- | --- | --- |
| `--trajectory` | `figure8`, `circle`, `line_y`, `star`, or `hover` | `figure8` |
| `--wind` | Nonnegative scale of the generated wind acceleration disturbance | `8` |
| `--wind-profile` | `default`, `shift_flip`, or `shift_turb` | `shift_flip` |
| `--seed` | Random seed | `0` |
| `--steps` | Number of control updates, from 1 to 281 | `281` |
| `--mass-scale` | Plant mass multiplier | `1` |
| `--drag` | Plant drag coefficient | `0` |
| `--constraint-stages` | Number of active learned constraint stages, from 1 to 19 | `1` |
| `--csv` | Output trajectory path | No file |

`success=1` means the episode has no floor, attitude, or angular-rate violations. Tracking errors are reported separately. A completed executable run can therefore report `success=0`; its exit status indicates whether the program ran correctly, not whether the robot completed its task safely.

CSV state values are recorded **after** the control update, with their corresponding reference. Inputs and disturbances are those used for that update. `authority` is the applied scheduler output in `[0,1]`. The solver may reach its iteration limit and still apply its available iterate, matching the controller's runtime behavior.

## F1TENTH

```sh
./build/raya_f1tenth --friction-profile strips --friction 0.30 --patch-period 4
./build/raya_f1tenth --friction-profile single_patch --friction 0.25 --patch-time 8
./build/raya_f1tenth --friction-profile random --friction 0.35 --patch-count 4 --patch-phase 1.1
```

This experiment uses a five-state kinematic bicycle model: position, heading, speed, and steering angle. Inputs are longitudinal acceleration and steering rate. A figure-eight reference is tracked at 10 Hz for up to 400 updates. The controller uses a 20-state horizon, an 80-iteration cap, and one active constraint stage (`K=1`). It includes a frozen 20-feature recoverability network and an 11-coefficient authority policy.

The online friction estimator is disabled. The authority policy uses the fixed causal feature set without estimated or ground-truth friction. The plant applies friction changes through the yaw response and lateral-slip model.

| Option | Meaning | Default |
| --- | --- | --- |
| `--friction-profile` | `strips`, `single_patch`, `random`, or `none` | `strips` |
| `--friction` | Grip inside a patch, from 0.01 to 1 | `0.30` |
| `--patch-period` | Repeating strip period in simulation time, seconds | `4` |
| `--patch-phase` | Continuous strip-time offset or random-layout phase | `0` |
| `--patch-time` | Reference time used to locate the single patch, seconds | `4` |
| `--patch-count` | Number of patches for the random layout | `1` |
| `--seed` | Random seed | `0` |
| `--steps` | Maximum control updates | `400` |
| `--mass-scale` | Plant response multiplier | `1` |
| `--drag` | Plant drag coefficient | `0` |
| `--constraint-stages` | Number of active learned constraint stages, from 1 to 19 | `1` |
| `--csv` | Output trajectory path | No file |

Patch geometry uses continuous placement parameters. Changing `--seed` alone does not provide new strip or single-patch locations; change `--patch-phase` or `--patch-time` as appropriate.

The episode stops at the first declared tracking, spinout, slip, or track-departure failure. Tracking failure requires five consecutive updates outside the 0.8 m tube. `success=1` means no declared failure occurred within the requested episode length. CSV states are recorded after each update; `time` is in seconds, positions in meters, and angles in radians.

## Reproducibility

The shipped model constants are immutable assets checked by `tests/model_hashes.json`. RAYA outputs were compared with the source controller during extraction. The disturbance and initial-state sampling protocol is explicit in `examples/common/random.hpp`, matching the benchmark's GNU-library sequences. CMake disables floating-point contraction to preserve operation rounding. Independent sampling vectors and seeded replay are tested; other math-library and floating-point differences may still affect trajectories across toolchains.

The small experiment runner writes `summary.csv`, per-episode CSVs, and console output under the requested directory. It runs only RAYA. No Python packages beyond the standard library are required.

The full [paper reproduction](paper-reproduction.md) runner uses the original six quadrotor families and continuously placed friction geometries. Its default is the selected `K=1` controller. The RAYA-only constraint-depth ablation can also be run with `--platform quadrotor --constraint-stages 3` or `--platform f1tenth --constraint-stages 5`; omit `--check-paper` for those ablations.
