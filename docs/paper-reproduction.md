# Paper reproduction

Verified on 2026-09-20 with Apple Clang 16.0.0, native `arm64` executables, and a Release build with floating-point contraction disabled. All **7,200 RAYA episodes** completed without execution errors. The selected controller uses **one active constraint stage (`K=1`)** on both platforms.

## Results

| RAYA metric | Paper | Reproduced |
| --- | ---: | ---: |
| Quadrotor survival | 40.83% | **1715/4200 = 40.83333%** |
| Quadrotor survivor XY error | 0.390 m | 0.390218299 m |
| Quadrotor mean reallocation | 0.698 | 0.698476801 |
| F1TENTH survival | 48.73% | **1462/3000 = 48.73333%** |
| F1TENTH survivor tracking error | 0.161 m | 0.160608349 m |
| F1TENTH mean reallocation | 0.271 | 0.271133346 |

All six metrics match the paper's rounding. Tracking error is averaged over successful episodes. Reallocation is the mean of the per-episode mean scheduler output over all episodes, including failures.

Additional RAYA results also match:

| Check | Reproduced | Paper |
| --- | ---: | ---: |
| Figure-eight success at disturbance multiplier 8 | 62/100 = 62% | 62% |
| Normalized Figure-eight survival-curve area | 45.91667% | 45.92% |
| F1TENTH periodic strips | 768/1500 = 51.2% | 51.2% |
| F1TENTH random patches | 471/1000 = 47.1% | 47.1% |
| F1TENTH single patch | 223/500 = 44.6% | 44.6% |
| F1TENTH strip cells with at least 50% survival | 8/15 | 8/15 |
| F1TENTH interpolated 50%-survival exposure | 2.096154 s | 2.096 s |

The Figure-eight successes at multipliers 6 through 12 are `94, 83, 62, 40, 23, 16, 9`, each out of 100. The area uses trapezoidal integration divided by the sweep width. Periodic-strip survival pooled across friction levels is `60.0%, 52.0%, 41.6%` at exposures `1.5, 2.0, 2.5 s`; linear interpolation gives the reported exposure threshold.

## Run it

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 examples/reproduce_paper.py --build-dir build --output results/paper --workers 4 --check-paper
```

The runner uses Python's standard library. The output path must not already exist. `--platform quadrotor` or `--platform f1tenth` runs one complete platform grid. `--check-paper` returns a nonzero status if an episode cannot execute or if any aggregate differs at the paper's reported precision. An individual task failure is a valid experimental outcome and remains in the ledger.

Before launching simulations, the runner writes every episode's arguments, source hashes, binary hashes, and build settings to `manifest.json`. It then saves `episodes.csv`, per-episode console logs, and `summary.json`. The local verified run is retained in `results/paper/`, which is ignored by Git. The compact [verification artifact](reproduction.json) records all 72 cell summaries, artifact hashes, and aggregate comparisons.

## Frozen protocol

The quadrotor grid crosses six families (figure eight, circle, Y line, star, turbulent wind, and a plant with 20% extra mass and drag coefficient 0.2), seven disturbance multipliers (6 through 12), and seeds 0 through 99. Each episode runs 281 updates at 20 Hz and fails if it violates the floor, attitude, or angular-rate limits.

The F1TENTH grid crosses friction values `0.20, 0.25, 0.30, 0.35, 0.40` with three strip periods, two random patch counts, and one isolated-patch condition. Each cell has 100 distinct continuous geometry placements. Strip phases are `(i + 0.5) * period / 100`, random layout phases are `(i + 0.5) * 2π / 100`, and isolated-patch times span 4.0 to 20.5 seconds at cell midpoints. Seeds are `91000 + i`. Each episode runs up to 400 updates at 10 Hz, stopping on the declared failure condition. Ground-truth friction and patch geometry are unavailable to the controller.

## Corrections made during verification

The initial extraction used the earlier constraint depths: three stages for the quadrotor and five for F1TENTH. The defaults now match the paper-selected `K=1` configuration. Other RAYA depths remain accessible through `--constraint-stages`.

The first macOS run also exposed a sampling portability issue: standard-library random distributions produced different quadrotor disturbances from the benchmark's GNU-library sequences, yielding 41.62% survival. A GNU build using the original distributions reproduced 40.83%, the full Figure-eight curve, tracking error, and reallocation. The release now specifies the sampling algorithms explicitly in `examples/common/random.hpp` and checks them against independent GNU-generated vectors. The final native macOS run reproduced all 4,200 GNU episode success/failure decisions. No model, policy, controller weight, or failure threshold was tuned to match a target.

Floating-point trajectories are not guaranteed to be identical across architectures. In one severe star-trajectory check, both builds first failed at update index 68; their positions diverged by more than `1e-5` only at update 196, long after that failure. This did not change its outcome or the reproduced rounded paper metrics.

## Scope

This verification covers the selected RAYA simulation rows and the RAYA-only checks above. It does not independently reproduce confidence intervals, comparative gains, every ablation, or hardware mission counts. The Crazyflie controller previously compiled and linked successfully with one constraint stage, five ADMM iterations, and the 20 Hz control schedule. No flights were performed; the paper's 10/10 hardware missions require physical experiments to verify.
