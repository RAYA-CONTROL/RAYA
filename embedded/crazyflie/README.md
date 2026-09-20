# RAYA on Crazyflie

This is the RAYA out-of-tree controller for the **Crazyflie 2.1 Brushless (`cf21bl`)**. It includes the learned recoverability network, causal disturbance trigger, learned authority scheduler, a 16-by-4 float32 Riccati cache bank, and the embedded TinyMPC runtime.

## Build

Install Git, Python 3, GNU Make, a host C compiler, and the Arm GNU Toolchain (`arm-none-eabi-gcc` and `arm-none-eabi-g++` on `PATH`). On macOS, also install GNU `coreutils` and `gnu-sed` through Homebrew.

From the repository root:

```sh
python3 embedded/crazyflie/setup_firmware.py
cd embedded/crazyflie
make -j4
```

The setup script fetches firmware commit `de14ff9f08ae80cbb36ffd37859efc2bedfbf6bf`, initializes its submodules, and applies `patches/firmware.patch`. The patch supplies the motor-cap/fault interface used by this controller and the linker flag adjustment. An existing checkout at another revision is rejected; local files are not reset.

The output is `build/cf21bl.bin`. Firmware sources, build outputs, and object files are ignored by Git. A separate firmware checkout can be selected with `make CRAZYFLIE_BASE=/absolute/path/to/firmware` after applying the same patch.

## Configuration

```sh
make TRAJECTORY=4 ACTIVE_HORIZON=1
make TRAJECTORY=0 ACTIVE_HORIZON=3
make BENCH_PROFILE=1
```

| Setting | Values | Default |
| --- | --- | --- |
| `TRAJECTORY` | 0 hover, 1 X line, 2 Y line, 3 circle, 4 figure eight | `4` |
| `ACTIVE_HORIZON` | Number of constrained prediction stages | `1` |
| `BENCH_PROFILE` | 1 runs canned-state profiling with motor output disabled | `0` |
| `NUMERIC_O2` | 1 uses `-O2` for the controller and numerical kernel | `0` |

The controller runs MPC at 20 Hz with a five-iteration ADMM budget and the inner PID at 500 Hz. Figure-eight playback uses the supplied state and input references; feedforward input costs are enabled. The default is the current local deployment configuration with one active barrier stage. A three-stage configuration can be built explicitly. Changing the horizon changes the experiment and should be recorded with flight results.

The source defaults to full motor authority: the motor-fault retained percentage and global motor cap are both 100. Fault timing and takeoff/landing constants are near the top of `src/controller_raya.cpp`. Controller selection is out-of-tree controller **6**; runtime parameters are under `raya` and telemetry under `tinympc`.

Takeoff is initiated through the client after controller 6 is selected. The reference begins two seconds after the takeoff ramp starts, at the measured position, without a settling gate. The floor frame is assumed to be near estimated `z=0`. Verify the estimator, reference frame, and configured trajectory before flight. Profiling mode provides a motors-off build for timing checks.

The console's `RAYA` records contain step, time, measured/reference position, recoverability margin, applied authority, vertical-cost scale, solve time, total controller time, fault state, and deadline overruns. Position is scaled to millimeters and scalar controller values to thousandths. More detailed variables are available through the log groups.

## Source layout

- `src/controller_raya.cpp`: real-time task, constraint insertion, reference handling, takeoff, and landing.
- `src/raya_barrier.hpp` and model headers: float32 recoverability evaluation and gradients.
- `src/raya_scheduler.hpp`: bounded authority policy, causal trigger, and cache selection.
- `src/raya_cache_seed.hpp`: initial solver cost/cache entry and model setup.
- `solver/`: fixed-size embedded TinyMPC runtime.
- `patches/`: required changes to the pinned firmware dependency.

The controller carries GPLv3 notices. The solver and Eigen retain their respective notices. Compilation is an integration check; this extracted repository has not been flight-tested.
