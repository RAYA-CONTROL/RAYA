# Release verification

The extraction was checked locally on 2026-09-20 with Apple Clang 16 and a Release CMake build. The firmware was compiled with Arm GNU Toolchain 13.3.Rel1.

## Simulation verification

Ten episodes were compared against the source RAYA implementation using the same toolchain, seeds, and controller settings. The comparison was repeated after the paper configuration and portable-sampling corrections using GNU GCC 14.2.0. The comparison covered every logged state, control input, and applied authority value. The largest absolute difference was below `5e-10`, within the precision of the released trajectory CSVs.

| Platform | Conditions |
| --- | --- |
| Quadrotor | Figure eight, circle, Y line, star, turbulent wind, and a mass/drag shift |
| F1TENTH | Repeating strips, a single patch, random patches, and a severe low-friction strip case |

The subsequent full paper reproduction ran all 7,200 selected RAYA episodes after aligning constraint depth and making disturbance sampling portable. Both platforms matched the paper. See [the reproduction report](paper-reproduction.md) for the measured values, protocol, and cross-architecture limits.

## Automated checks

`ctest --test-dir build --output-on-failure` checks independent sampling vectors, the frozen-model hashes, closed-loop trajectories and edge cases, seed replay, finite telemetry, actuator and authority bounds, CSV structure and timestamps, invalid CLI arguments, and output-path failures. Executables run from a temporary directory to check that models do not depend on the launch directory.

The example experiment runner was also run end-to-end. A task failure is reported in its metrics and does not prevent the other experiments from running.

CI is configured for Linux and macOS. Those hosted runs will execute after the repository is pushed; local checks passed on macOS with native arm64 and x86_64 Apple Clang builds and x86_64 GNU GCC 14.2.0.

## Embedded integration

The RAYA-only `cf21bl` firmware compiled and linked against the pinned firmware revision with the supplied patch. The default figure-eight build used `ACTIVE_HORIZON=1` and `BENCH_PROFILE=0`.

| Memory | Used | Available |
| --- | ---: | ---: |
| Flash | 555,060 bytes | 1,032,192 bytes |
| RAM | 113,504 bytes | 131,072 bytes |
| CCM | 62,392 bytes | 65,536 bytes |

These figures describe the tested compiler and configuration. Other toolchains and build options can change them. No firmware was flashed and the extracted code has not been flight-tested.
