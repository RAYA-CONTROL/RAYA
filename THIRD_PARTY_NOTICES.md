# Third-party notices

| Component | Location | Notice |
| --- | --- | --- |
| Modified TinyMPC runtime | `third_party/tinympc/` | Original MIT notice in `LICENSE`; upstream project: https://github.com/TinyMPC/TinyMPC |
| Embedded TinyMPC runtime | `embedded/crazyflie/solver/` | Original MIT notice in `LICENSE` |
| Eigen headers | `third_party/eigen/` | Original per-file notices and `COPYING.*` files; project: https://eigen.tuxfamily.org/ |
| Eigen wrapper | `third_party/eigen/Eigen.h`, `embedded/crazyflie/include/Eigen.h` | Bolder Flight Systems MIT notice retained in each file |
| Crazyflie controller and integration | `embedded/crazyflie/` | Bitcraze copyright and GPLv3 notices retained; full license in `LICENSE` |
| Crazyflie firmware | Downloaded by `embedded/crazyflie/setup_firmware.py` | Pinned source and its recursive dependencies retain their own license files |

Repository-level licensing does not replace the notices in these components. The RAYA integration modifies the solver and controller source; their upstream notices remain included.
