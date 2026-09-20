// RAYA: Recoverability-Aware Yielding of Authority.
// Copyright (c) 2026 RAYA contributors. SPDX-License-Identifier: MIT

#include <cstdlib>

#define NSTATES 12
#define NINPUTS 4
#define NHORIZON 20
#define NTOTAL 301

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <Eigen/Cholesky>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include "../common/random.hpp"
#include <sstream>
#include <string>
#include <vector>

#include <tinympc/tiny_api.hpp>
#include "models/barrier.hpp"
#include "models/dynamics.hpp"
#include "models/reference.hpp"

extern "C" {

typedef Matrix<tinytype, NINPUTS, NHORIZON - 1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;
typedef Matrix<tinytype, NINPUTS, 1> tiny_VectorNu;
typedef Matrix<tinytype, 3, 1> tiny_Vector3;

enum class WindProfile {
    kDefault = 0,
    kShiftTurb = 1,
    kShiftFlip = 2,
};

enum class FailureMode {
    kFloor = 0,
    kAttitude = 1,
    kCombined = 2,
};

struct BarrierEval {
    tinytype h = 0.0;
    tinytype dh_dz = 0.0;
    tinytype dh_dvz = 0.0;
};

struct AdaptiveKnobRuntime {
    tinytype base_wind_assumed = 0.0;
    tinytype base_margin_scale = 0.0;
    tinytype base_recovery_gain = 0.0;
    tinytype base_q_xy_scale = 1.0;
    tinytype base_q_z_scale = 1.0;
    tinytype current_wind_assumed = 0.0;
    tinytype current_margin_scale = 0.0;
    tinytype current_recovery_gain = 0.0;
    tinytype current_q_xy_scale = 1.0;
    tinytype current_q_z_scale = 1.0;
    tinytype applied_q_xy_scale = 1.0;
    tinytype applied_q_z_scale = 1.0;
    tinytype target_wind_assumed = 0.0;
    tinytype target_margin_scale = 0.0;
    tinytype target_recovery_gain = 0.0;
    tinytype target_q_xy_scale = 1.0;
    tinytype target_q_z_scale = 1.0;
    tinytype raw_wind_delta = 0.0;
    tinytype raw_margin_delta = 0.0;
    tinytype raw_recovery_delta = 0.0;
    tinytype raw_q_xy_delta = 0.0;
    tinytype raw_q_z_delta = 0.0;
    tinytype last_deficit = 0.0;
    int last_update_step = -1;
};

struct AdaptiveObsRuntime {
    tinytype prev_barrier_deficit = 0.0;
    tinytype prev_z_err = 0.0;
    tinytype prev_sat_frac = 0.0;
    tinytype sat_ema = 0.0;
    bool initialized = false;
};

struct NoOracleQuadComponents {
    tinytype pos_residual = 0.0;
    tinytype vel_residual = 0.0;
    tinytype z_residual = 0.0;
    tinytype vz_residual = 0.0;
    tinytype attitude_residual = 0.0;
    tinytype selected_margin = 0.0;
    tinytype margin_shrink_rate = 0.0;
    tinytype barrier_deficit = 0.0;
    tinytype xy_err = 0.0;
    tinytype abs_z_err = 0.0;
    tinytype sat_frac = 0.0;
    tinytype score_raw = 0.0;
    int severe_event = 0;
};

struct NoOracleQuadRuntime {
    NoOracleQuadComponents components;
    tinytype score = 0.0;
    tinytype score_sum = 0.0;
    tinytype max_score = 0.0;
    tinytype residual_sum = 0.0;
    tinytype max_vel_residual = 0.0;
    tinytype prev_selected_margin = 0.0;
    bool prev_margin_valid = false;
    bool active = false;
    int hold_remaining = 0;
    int active_steps = 0;
    int first_trigger_step = -1;
    int severe_events = 0;
};

struct LearnedResidualDiagnostics {
    tinytype delta_h_model = 0.0;
    tinytype delta_h_raw = 0.0;
    tinytype delta_h_clipped = 0.0;
    tinytype gate_score = std::numeric_limits<tinytype>::quiet_NaN();
    tinytype h_analytic = 0.0;
    tinytype h_learned = 0.0;
    tinytype margin = 0.0;
    int constraint_active_before = 0;
    int constraint_active_after = 0;
    int active_count_before = 0;
    int active_count_after = 0;
    int first_changed_stage = -1;
    int gate_applied_count = 0;
    int gate_blocked_count = 0;
    int first_gate_head = -1;
    tinytype tv_state_min_slack_pre = std::numeric_limits<tinytype>::quiet_NaN();
    tinytype tv_state_min_slack_post = std::numeric_limits<tinytype>::quiet_NaN();
    tinytype tv_input_min_slack_pre = std::numeric_limits<tinytype>::quiet_NaN();
    tinytype tv_input_min_slack_post = std::numeric_limits<tinytype>::quiet_NaN();
};

struct Stats {
    int steps = 0;
    int floor_violations = 0;
    int attitude_violations = 0;
    int position_violations = 0;
    int combined_violations = 0;
    int solver_success = 0;
    int solver_fail = 0;
    int saturation_count = 0;

    tinytype sum_xy_err = 0.0;
    tinytype sum_abs_z_err = 0.0;
    tinytype sum_sq_xy_err = 0.0;
    tinytype sum_sq_z_err = 0.0;
    tinytype sum_control_l2 = 0.0;
    tinytype sum_control_l2_sq = 0.0;
    tinytype sum_control_abs_mean = 0.0;
    tinytype min_z = std::numeric_limits<tinytype>::infinity();
    tinytype max_abs_roll_deg = 0.0;
    tinytype max_abs_pitch_deg = 0.0;
    tinytype max_angvel_norm = 0.0;
    tinytype sum_iters = 0.0;
    tinytype sum_solve_ms = 0.0;
    tinytype sum_psf_du_norm = 0.0;
    std::vector<tinytype> solve_times_ms;
};

constexpr int kAuthorityPolicyDim = 11;

// Frozen RAYA configuration. CLI options change experiment conditions.
struct Options {

    WindProfile wind_profile = WindProfile::kShiftFlip;

    int seed = 0;
    int steps = NTOTAL - NHORIZON;
    int max_iter = 30;

    tinytype wind_actual = 8.0;
    tinytype wind_assumed = 1.6;
    tinytype wind_clip_xy = 20.0;
    tinytype wind_clip_z = 40.0;

    tinytype z_floor = 0.0;
    tinytype fail_roll_deg = 50.0;
    tinytype fail_pitch_deg = 50.0;
    tinytype fail_angvel_norm = 20.0;
    tinytype fail_track_xy_radius = 0.0;
    tinytype fail_track_z_abs = 0.0;
    FailureMode failure_mode = FailureMode::kCombined;
    bool enforce_floor_state_bound = true;
    bool enforce_attitude_state_bounds = true;

    tinytype barrier_u_context = 3.0;
    bool barrier_u_context_auto = true;

    tinytype barrier_az_coeff = 8.0;
    bool barrier_az_coeff_auto = true;
    tinytype barrier_gravity_comp = 0.0;
    tinytype barrier_base_margin = 0.01;
    tinytype robust_margin_scale = 1.5;
    tinytype near_floor_margin_gain = 0.10;
    tinytype descending_margin_gain = 0.02;
    tinytype horizon_margin_rate = 0.0008;
    tinytype learned_h_blend = 0.75;
    int barrier_active_horizon = 1;  // Paper-selected first predicted stage.
    bool barrier_skip_high_alt = true;
    tinytype barrier_skip_z = 0.75;
    tinytype barrier_skip_vz = -0.20;
    tinytype barrier_h_deadband = 0.30;
    tinytype robust_solver_activation_slack = 0.20;
    tinytype robust_structural_guard_relax = 0.70;

    tinytype learned_margin_residual_gate_activation_slack = 0.0;

    bool enable_recovery_input_constraint = true;
    int recovery_stages = 3;
    tinytype recovery_collective_base = 0.30;
    tinytype recovery_collective_gain = 0.35;
    tinytype recovery_collective_max = 1.80;

    tinytype plant_mass_scale = 1.0;
    tinytype plant_drag_coeff = 0.0;

    tinytype authority_beta_track = 1.0;
    tinytype authority_beta_attitude = 0.5;
    tinytype authority_beta_reserve = 0.0;

    tinytype authority_tau = 0.30;
    tinytype authority_deadband = 0.30;

    tinytype authority_policy_smoothing = 0.70;

    int authority_quantization_levels = 0;

    double authority_policy_weights[kAuthorityPolicyDim] = {-1.98658178691,6.58130741222,-1.04181888859,0.431873123638,-0.949319294365,1.92888487721,1.96624863652,1.07054534043,-0.0539227186924,-0.92818277774,0.83834396466};

    bool adaptive_rho = false;
    tinytype adaptive_rho_min = 8.0;
    tinytype adaptive_rho_max = 12.0;
    bool reset_duals_each_step = false;
    bool soft_tv_state_linear = true;
    tinytype soft_tv_state_linear_gamma = 20.0;
    tinytype soft_tv_state_linear_gamma_safe = 2.0;

    tinytype adaptive_knob_smoothing = 0.80;

    tinytype adaptive_wind_min = 0.80;
    tinytype adaptive_wind_max = 3.20;
    tinytype adaptive_margin_min = 0.50;
    tinytype adaptive_margin_max = 2.50;
    tinytype adaptive_recovery_min = 0.20;
    tinytype adaptive_recovery_max = 1.60;
    tinytype adaptive_q_xy_min = 0.80;
    tinytype adaptive_q_xy_max = 1.20;
    tinytype adaptive_q_z_min = 1.00;
    tinytype adaptive_q_z_max = 1.80;

    tinytype no_oracle_enter_score = 1.60;
    tinytype no_oracle_exit_score = 0.80;
    tinytype no_oracle_score_alpha = 0.35;
    int no_oracle_hold_steps = 20;
    int no_oracle_severe_hold_steps = 60;
    tinytype no_oracle_severe_vel_residual = 0.12;
    tinytype no_oracle_severe_vz_residual = 0.10;
    tinytype no_oracle_severe_score = 1.80;
    tinytype no_oracle_severe_margin = 0.25;
    tinytype no_oracle_wind_delta = 1.40;
    tinytype no_oracle_margin_delta = 0.25;
    tinytype no_oracle_recovery_delta = 0.25;
    tinytype no_oracle_q_xy_delta = 0.00;
    tinytype no_oracle_q_z_delta = 0.40;
    tinytype no_oracle_feat_score = 0.0;
    tinytype no_oracle_feat_score_raw = 0.0;
    tinytype no_oracle_feat_active = 0.0;
    tinytype no_oracle_feat_pos_residual = 0.0;
    tinytype no_oracle_feat_vel_residual = 0.0;
    tinytype no_oracle_feat_z_residual = 0.0;
    tinytype no_oracle_feat_vz_residual = 0.0;
    tinytype no_oracle_feat_attitude_residual = 0.0;
    tinytype no_oracle_feat_selected_margin = 0.0;
    tinytype no_oracle_feat_margin_shrink_rate = 0.0;
    tinytype no_oracle_feat_barrier_deficit = 0.0;
    tinytype no_oracle_feat_sat_frac = 0.0;
    tinytype no_oracle_feat_severe_event = 0.0;

    std::string step_csv;
    std::string trajectory = "figure8";
};

struct FailureEval {
    tinytype m_floor = 0.0;
    tinytype m_roll = 0.0;
    tinytype m_pitch = 0.0;
    tinytype m_rate = 0.0;
    tinytype m_safety_combined = 0.0;
    tinytype m_xy_track = std::numeric_limits<tinytype>::infinity();
    tinytype m_z_track = std::numeric_limits<tinytype>::infinity();
    tinytype m_position = std::numeric_limits<tinytype>::infinity();
    tinytype m_combined = 0.0;
    tinytype roll_deg = 0.0;
    tinytype pitch_deg = 0.0;
    tinytype angvel_norm = 0.0;
    tinytype xy_track_err = 0.0;
    tinytype abs_z_track_err = 0.0;
    bool floor_fail = false;
    bool attitude_fail = false;
    bool position_fail = false;
    bool combined_fail = false;
};

static FailureEval evaluate_failure(const tiny_VectorNx &x,
                                    const Options &opt,
                                    const tiny_VectorNx *ref);

static void apply_system_recovery_defaults(Options *opt) {
    {

        opt->recovery_collective_base = 1.00;
        opt->recovery_collective_gain = 0.80;
        opt->recovery_collective_max = 2.00;
    }
}

static bool parse_profile(const std::string &s, WindProfile *out) {
    if (s == "default") {
        *out = WindProfile::kDefault;
        return true;
    }
    if (s == "shift_turb") {
        *out = WindProfile::kShiftTurb;
        return true;
    }
    if (s == "shift_flip") {
        *out = WindProfile::kShiftFlip;
        return true;
    }
    return false;
}

static tinytype clampv(tinytype x, tinytype lo, tinytype hi) {
    return std::max<tinytype>(lo, std::min<tinytype>(hi, x));
}

static tinytype safe_div(tinytype a, tinytype b) {
    return (std::abs(b) > 1e-12) ? (a / b) : 0.0;
}

static tinytype deg2rad(tinytype deg) {
    return deg * static_cast<tinytype>(M_PI / 180.0);
}

static tinytype rad2deg(tinytype rad) {
    return rad * static_cast<tinytype>(180.0 / M_PI);
}

enum BarrierFeatureMode {
    kFeatureLegacy4 = 0,
    kFeatureAuthority4 = 1,
    kFeatureQuad8 = 2,
    kFeatureAuthority5 = 3,
    kFeatureAttitude10 = 4,
    kFeatureCombined6 = 5,
    kFeatureCombined10Traj = 6,
};

static int barrier_feature_mode() {
#ifdef LEARNED_FLOOR_BARRIER_FEATURE_MODE_ID
    return static_cast<int>(LEARNED_FLOOR_BARRIER_FEATURE_MODE_ID);
#else

    return (learned_floor_barrier::kInputDim > 4) ? kFeatureQuad8 : kFeatureLegacy4;
#endif
}

static bool is_attitude10_mode() {
    const int m = barrier_feature_mode();
    return (m == kFeatureAttitude10 && learned_floor_barrier::kInputDim == 10) ||
           (m == kFeatureCombined6 && learned_floor_barrier::kInputDim == 6) ||
           (m == kFeatureCombined10Traj && learned_floor_barrier::kInputDim == 10);
}

static void fill_trajectory_onehot(const std::string &trajectory, double *out4) {
    out4[0] = 0.0;
    out4[1] = 0.0;
    out4[2] = 0.0;
    out4[3] = 0.0;
    if (trajectory == "circle") {
        out4[1] = 1.0;
    } else if (trajectory == "star") {
        out4[2] = 1.0;
    } else if (trajectory == "line_y" || trajectory == "y" || trajectory == "line") {
        out4[3] = 1.0;
    } else {
        out4[0] = 1.0;
    }
}

static void build_random_complex_reference(const Options &opt,
                                           Matrix<tinytype, NSTATES, NTOTAL> *Xref_total) {
    std::mt19937 rng(static_cast<uint32_t>(opt.seed) + 100003u);
    raya::sampling::UniformReal unit01(0.0, 1.0);
    raya::sampling::UniformReal phase_dist(0.0, static_cast<tinytype>(2.0 * M_PI));
    raya::sampling::UniformReal rot_dist(-static_cast<tinytype>(M_PI),
                                                      static_cast<tinytype>(M_PI));
    raya::sampling::UniformReal z_center_dist(0.92, 1.08);
    raya::sampling::UniformReal envelope_depth_dist(0.10, 0.22);
    raya::sampling::UniformReal envelope_freq_dist(0.08, 0.20);

    const tinytype dt = 0.05;
    const tinytype z_center = z_center_dist(rng);
    const tinytype theta = rot_dist(rng);
    const tinytype ct = std::cos(theta);
    const tinytype st = std::sin(theta);

    const tinytype env_depth_x = envelope_depth_dist(rng);
    const tinytype env_depth_y = envelope_depth_dist(rng);
    const tinytype env_freq_x = envelope_freq_dist(rng);
    const tinytype env_freq_y = envelope_freq_dist(rng);
    const tinytype env_phase_x = phase_dist(rng);
    const tinytype env_phase_y = phase_dist(rng);

    const tinytype amp_x[4] = {
        static_cast<tinytype>(0.34 + 0.10 * unit01(rng)),
        static_cast<tinytype>(0.18 + 0.08 * unit01(rng)),
        static_cast<tinytype>(0.10 + 0.05 * unit01(rng)),
        static_cast<tinytype>(0.06 + 0.03 * unit01(rng)),
    };
    const tinytype amp_y[4] = {
        static_cast<tinytype>(0.28 + 0.10 * unit01(rng)),
        static_cast<tinytype>(0.18 + 0.08 * unit01(rng)),
        static_cast<tinytype>(0.09 + 0.05 * unit01(rng)),
        static_cast<tinytype>(0.05 + 0.03 * unit01(rng)),
    };
    const tinytype amp_z[3] = {
        static_cast<tinytype>(0.05 + 0.03 * unit01(rng)),
        static_cast<tinytype>(0.03 + 0.02 * unit01(rng)),
        static_cast<tinytype>(0.02 + 0.02 * unit01(rng)),
    };

    const tinytype w_x[4] = {
        static_cast<tinytype>(0.28 + 0.14 * unit01(rng)),
        static_cast<tinytype>(0.52 + 0.18 * unit01(rng)),
        static_cast<tinytype>(0.82 + 0.22 * unit01(rng)),
        static_cast<tinytype>(1.12 + 0.30 * unit01(rng)),
    };
    const tinytype w_y[4] = {
        static_cast<tinytype>(0.24 + 0.14 * unit01(rng)),
        static_cast<tinytype>(0.48 + 0.20 * unit01(rng)),
        static_cast<tinytype>(0.78 + 0.24 * unit01(rng)),
        static_cast<tinytype>(1.16 + 0.28 * unit01(rng)),
    };
    const tinytype w_z[3] = {
        static_cast<tinytype>(0.18 + 0.10 * unit01(rng)),
        static_cast<tinytype>(0.42 + 0.12 * unit01(rng)),
        static_cast<tinytype>(0.76 + 0.14 * unit01(rng)),
    };

    const tinytype p_x[4] = {phase_dist(rng), phase_dist(rng), phase_dist(rng), phase_dist(rng)};
    const tinytype p_y[4] = {phase_dist(rng), phase_dist(rng), phase_dist(rng), phase_dist(rng)};
    const tinytype p_z[3] = {phase_dist(rng), phase_dist(rng), phase_dist(rng)};

    for (int k = 0; k < NTOTAL; ++k) {
        const tinytype t = k * dt;
        const tinytype env_x = 1.0 + env_depth_x * std::sin(env_freq_x * t + env_phase_x);
        const tinytype env_y = 1.0 + env_depth_y * std::sin(env_freq_y * t + env_phase_y);
        const tinytype env_dot_x = env_depth_x * env_freq_x * std::cos(env_freq_x * t + env_phase_x);
        const tinytype env_dot_y = env_depth_y * env_freq_y * std::cos(env_freq_y * t + env_phase_y);

        tinytype x_base = 0.0;
        tinytype y_base = 0.0;
        tinytype dx_base = 0.0;
        tinytype dy_base = 0.0;
        tinytype z = z_center;
        tinytype vz = 0.0;
        for (int i = 0; i < 4; ++i) {
            const tinytype sx = std::sin(w_x[i] * t + p_x[i]);
            const tinytype cx = std::cos(w_x[i] * t + p_x[i]);
            const tinytype sy = std::sin(w_y[i] * t + p_y[i]);
            const tinytype cy = std::cos(w_y[i] * t + p_y[i]);
            x_base += amp_x[i] * sx;
            y_base += amp_y[i] * cy;
            dx_base += amp_x[i] * w_x[i] * cx;
            dy_base += -amp_y[i] * w_y[i] * sy;
        }
        for (int i = 0; i < 3; ++i) {
            const tinytype sz = std::sin(w_z[i] * t + p_z[i]);
            const tinytype cz = std::cos(w_z[i] * t + p_z[i]);
            z += amp_z[i] * sz;
            vz += amp_z[i] * w_z[i] * cz;
        }

        const tinytype x_env = env_x * x_base;
        const tinytype y_env = env_y * y_base;
        const tinytype dx_env = env_dot_x * x_base + env_x * dx_base;
        const tinytype dy_env = env_dot_y * y_base + env_y * dy_base;

        const tinytype x = ct * x_env - st * y_env;
        const tinytype y = st * x_env + ct * y_env;
        const tinytype vx = ct * dx_env - st * dy_env;
        const tinytype vy = st * dx_env + ct * dy_env;

        Xref_total->col(k).setZero();
        (*Xref_total)(0, k) = x;
        (*Xref_total)(1, k) = y;
        (*Xref_total)(2, k) = z;
        (*Xref_total)(6, k) = vx;
        (*Xref_total)(7, k) = vy;
        (*Xref_total)(8, k) = vz;
    }
}

static void build_reference_trajectory(const Options &opt,
                                       Matrix<tinytype, NSTATES, NTOTAL> *Xref_total) {
    *Xref_total = Eigen::Map<Matrix<tinytype, NSTATES, NTOTAL>>(Xref_data);

    if (opt.trajectory == "hover") {
        constexpr tinytype z_target = 1.0;
        for (int k = 0; k < NTOTAL; ++k) {
            Xref_total->col(k).setZero();
            (*Xref_total)(2, k) = z_target;
        }
        return;
    }
    if (opt.trajectory == "figure8") {
        const tinytype dt = 0.05;
        const tinytype omega = 0.55;
        const tinytype amp_x = 0.9;
        const tinytype amp_y = 0.45;
        const tinytype z_target = 1.0;
        for (int k = 0; k < NTOTAL; ++k) {
            const tinytype t = k * dt;
            Xref_total->col(k).setZero();
            (*Xref_total)(0, k) = amp_x * std::sin(omega * t);
            (*Xref_total)(1, k) = amp_y * std::sin(2.0 * omega * t);
            (*Xref_total)(2, k) = z_target;
            (*Xref_total)(6, k) = amp_x * omega * std::cos(omega * t);
            (*Xref_total)(7, k) = 2.0 * amp_y * omega * std::cos(2.0 * omega * t);
        }
        return;
    }
    if (opt.trajectory == "circle") {
        const tinytype dt = 0.05;
        const tinytype omega = 0.45;
        const tinytype radius = 0.8;
        const tinytype z_target = 1.0;
        for (int k = 0; k < NTOTAL; ++k) {
            const tinytype t = k * dt;
            Xref_total->col(k).setZero();
            (*Xref_total)(0, k) = radius * std::cos(omega * t);
            (*Xref_total)(1, k) = radius * std::sin(omega * t);
            (*Xref_total)(2, k) = z_target;
            (*Xref_total)(6, k) = -radius * omega * std::sin(omega * t);
            (*Xref_total)(7, k) = radius * omega * std::cos(omega * t);
        }
        return;
    }
    if (opt.trajectory == "star") {
        const tinytype dt = 0.05;
        const tinytype omega = 0.35;
        const tinytype r_outer = 0.9;
        const tinytype r_inner = 0.35;
        const int n_points = 5;
        const tinytype z_target = 1.0;
        for (int k = 0; k < NTOTAL; ++k) {
            const tinytype t = k * dt;
            const tinytype angle = omega * t;
            const tinytype r = 0.5 * (r_outer + r_inner)
                             + 0.5 * (r_outer - r_inner) * std::cos(static_cast<tinytype>(n_points) * angle);
            const tinytype dr_dangle = -0.5 * (r_outer - r_inner)
                                     * static_cast<tinytype>(n_points) * std::sin(static_cast<tinytype>(n_points) * angle);
            Xref_total->col(k).setZero();
            (*Xref_total)(0, k) = r * std::cos(angle);
            (*Xref_total)(1, k) = r * std::sin(angle);
            (*Xref_total)(2, k) = z_target;
            (*Xref_total)(6, k) = omega * (dr_dangle * std::cos(angle) - r * std::sin(angle));
            (*Xref_total)(7, k) = omega * (dr_dangle * std::sin(angle) + r * std::cos(angle));
        }
        return;
    }
    if (opt.trajectory == "helix" || opt.trajectory == "helix3d") {
        const tinytype dt = 0.05;
        const tinytype omega = 0.45;
        const tinytype radius = 0.7;
        const tinytype z_base = 0.85;
        const tinytype z_amp = 0.25;
        const tinytype z_freq = 0.18;
        for (int k = 0; k < NTOTAL; ++k) {
            const tinytype t = k * dt;
            Xref_total->col(k).setZero();
            (*Xref_total)(0, k) = radius * std::cos(omega * t);
            (*Xref_total)(1, k) = radius * std::sin(omega * t);
            (*Xref_total)(2, k) = z_base + z_amp * std::sin(z_freq * t);
            (*Xref_total)(6, k) = -radius * omega * std::sin(omega * t);
            (*Xref_total)(7, k) = radius * omega * std::cos(omega * t);
            (*Xref_total)(8, k) = z_amp * z_freq * std::cos(z_freq * t);
        }
        return;
    }
    if (opt.trajectory == "lemniscate" || opt.trajectory == "fig8_fast") {
        const tinytype dt = 0.05;
        const tinytype omega = 0.75;
        const tinytype amp_x = 1.1;
        const tinytype amp_y = 0.55;
        const tinytype z_target = 1.0;
        for (int k = 0; k < NTOTAL; ++k) {
            const tinytype t = k * dt;
            Xref_total->col(k).setZero();
            (*Xref_total)(0, k) = amp_x * std::sin(omega * t);
            (*Xref_total)(1, k) = amp_y * std::sin(2.0 * omega * t);
            (*Xref_total)(2, k) = z_target;
            (*Xref_total)(6, k) = amp_x * omega * std::cos(omega * t);
            (*Xref_total)(7, k) = 2.0 * amp_y * omega * std::cos(2.0 * omega * t);
        }
        return;
    }
    if (opt.trajectory == "random_complex" || opt.trajectory == "complex" ||
        opt.trajectory == "chaos" || opt.trajectory == "haywire") {
        build_random_complex_reference(opt, Xref_total);
    }
}

static tiny_VectorNx build_initial_state(const Options &opt,
                                         const Matrix<tinytype, NSTATES, NTOTAL> &Xref_total) {
    tiny_VectorNx x = Xref_total.col(0);
    std::mt19937 rng(static_cast<uint32_t>(opt.seed) + 17u);
    raya::sampling::Normal init_xy_jit(0.0, 0.01);
    x(0) += init_xy_jit(rng);
    x(1) += init_xy_jit(rng);
    return x;
}

static int apply_state_bounds(TinySolver *solver, const Options &opt) {
    tiny_MatrixNxNh x_min = tiny_MatrixNxNh::Constant(-5.0);
    tiny_MatrixNxNh x_max = tiny_MatrixNxNh::Constant(5.0);
    tiny_MatrixNuNhm1 u_min = tiny_MatrixNuNhm1::Constant(-0.50);
    tiny_MatrixNuNhm1 u_max = tiny_MatrixNuNhm1::Constant(0.50);
    if (opt.enforce_floor_state_bound) {
        x_min.row(2).setConstant(opt.z_floor);
    }
    if (opt.enforce_attitude_state_bounds) {
        const tinytype roll_lim = deg2rad(opt.fail_roll_deg);
        const tinytype pitch_lim = deg2rad(opt.fail_pitch_deg);
        x_min.row(3).setConstant(-roll_lim);
        x_max.row(3).setConstant(roll_lim);
        x_min.row(4).setConstant(-pitch_lim);
        x_max.row(4).setConstant(pitch_lim);
        x_min.row(9).setConstant(-opt.fail_angvel_norm);
        x_max.row(9).setConstant(opt.fail_angvel_norm);
        x_min.row(10).setConstant(-opt.fail_angvel_norm);
        x_max.row(10).setConstant(opt.fail_angvel_norm);
        x_min.row(11).setConstant(-opt.fail_angvel_norm);
        x_max.row(11).setConstant(opt.fail_angvel_norm);
    }
    x_max.row(2).setConstant(2.5);
    return tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);
}

static int update_barrier_constraints(TinySolver *solver,
                                      const Options &opt,
                                      const tiny_VectorNx &x_now,
                                      tinytype *h_now,
                                      tinytype *margin_now,
                                      LearnedResidualDiagnostics *residual_diag = nullptr,
                                      int rollout_step = 0);
static int set_relaxed_tv_constraints(TinySolver *solver);

static FailureEval evaluate_failure(const tiny_VectorNx &x,
                                    const Options &opt,
                                    const tiny_VectorNx *ref = nullptr) {
    FailureEval out;
    const tinytype roll = x(3);
    const tinytype pitch = x(4);
    const tinytype p = x(9);
    const tinytype q = x(10);
    const tinytype r = x(11);
    const tinytype angvel_norm = std::sqrt(p * p + q * q + r * r);

    const tinytype roll_lim = deg2rad(opt.fail_roll_deg);
    const tinytype pitch_lim = deg2rad(opt.fail_pitch_deg);

    out.roll_deg = rad2deg(roll);
    out.pitch_deg = rad2deg(pitch);
    out.angvel_norm = angvel_norm;
    out.m_floor = x(2) - opt.z_floor;
    out.m_roll = roll_lim - std::abs(roll);
    out.m_pitch = pitch_lim - std::abs(pitch);
    out.m_rate = opt.fail_angvel_norm - angvel_norm;
    out.m_safety_combined = std::min(
        std::min(out.m_floor, out.m_roll),
        std::min(out.m_pitch, out.m_rate));
    if (ref != nullptr) {
        const tinytype dx = x(0) - (*ref)(0);
        const tinytype dy = x(1) - (*ref)(1);
        out.xy_track_err = std::sqrt(dx * dx + dy * dy);
        out.abs_z_track_err = std::abs(x(2) - (*ref)(2));
    }
    if (opt.fail_track_xy_radius > 0.0) {
        out.m_xy_track = opt.fail_track_xy_radius - out.xy_track_err;
    }
    if (opt.fail_track_z_abs > 0.0) {
        out.m_z_track = opt.fail_track_z_abs - out.abs_z_track_err;
    }
    out.m_position = std::min(out.m_xy_track, out.m_z_track);
    out.m_combined = std::min(out.m_safety_combined, out.m_position);
    out.floor_fail = out.m_floor < 0.0;
    out.attitude_fail = (out.m_roll < 0.0) || (out.m_pitch < 0.0) || (out.m_rate < 0.0);
    out.position_fail =
        (opt.fail_track_xy_radius > 0.0 && out.m_xy_track < 0.0) ||
        (opt.fail_track_z_abs > 0.0 && out.m_z_track < 0.0);
    out.combined_fail = out.floor_fail || out.attitude_fail || out.position_fail;
    return out;
}

static tinytype compute_cos_tilt_from_state(const tiny_VectorNx &x) {

    const tinytype roll = x(3);
    const tinytype pitch = x(4);
    return clampv(std::cos(roll) * std::cos(pitch), -1.0, 1.0);
}

static tinytype compute_az_max_from_state(const tiny_VectorNx &x, tinytype az_coeff, tinytype gravity_comp) {
    return az_coeff * compute_cos_tilt_from_state(x) - gravity_comp;
}

static tinytype compute_vertical_accel_coeff(const TinySolver *solver, tinytype dt) {
    if (solver == nullptr || solver->work == nullptr || dt <= 1e-9 || solver->work->nu <= 0) {
        return 0.0;
    }
    const int nu = solver->work->nu;
    tinytype max_dvz_step = 0.0;
    for (int j = 0; j < nu; ++j) {
        const tinytype b = solver->work->Bdyn(8, j);
        const tinytype uj = (b >= 0.0) ? solver->work->u_max(j, 0) : solver->work->u_min(j, 0);
        max_dvz_step += b * uj;
    }
    return std::max<tinytype>(0.0, max_dvz_step / dt);
}

static void finalize_barrier_runtime_params(Options *opt, const TinySolver *solver) {
    if (opt == nullptr || solver == nullptr || solver->work == nullptr) {
        return;
    }

    if (opt->barrier_u_context_auto) {
        opt->barrier_u_context = std::max<tinytype>(0.0, solver->work->u_max.col(0).sum());
    }
    if (opt->barrier_az_coeff_auto) {
        opt->barrier_az_coeff = compute_vertical_accel_coeff(solver, 0.05);
    }
}

static void fill_barrier_input(
    const tiny_VectorNx &x,
    tinytype wind_scale,
    tinytype u_context,
    tinytype az_coeff,
    tinytype gravity_comp,
    const std::string &trajectory,
    double *in,
    tinytype fail_roll_rad = 0.0,
    tinytype fail_pitch_rad = 0.0,
    tinytype fail_angvel_norm = 0.0
) {
    using namespace learned_floor_barrier;
    for (int i = 0; i < kInputDim; ++i) {
        in[i] = 0.0;
    }

    const int mode = barrier_feature_mode();

    if ((mode == kFeatureCombined6 && kInputDim == 6) ||
        (mode == kFeatureCombined10Traj && kInputDim == 10)) {
        const tinytype cos_tilt = compute_cos_tilt_from_state(x);
        const tinytype az_max = compute_az_max_from_state(x, az_coeff, gravity_comp);
        const tinytype roll = x(3);
        const tinytype pitch = x(4);
        const tinytype p = x(9);
        const tinytype q = x(10);
        const tinytype r = x(11);
        const tinytype tilt_margin = std::min(fail_roll_rad - std::abs(roll),
                                              fail_pitch_rad - std::abs(pitch));
        const tinytype angvel_norm = std::sqrt(p * p + q * q + r * r);
        in[0] = static_cast<double>(x(2));
        in[1] = static_cast<double>(x(8));
        in[2] = static_cast<double>(cos_tilt);
        in[3] = static_cast<double>(az_max);
        in[4] = static_cast<double>(tilt_margin);
        in[5] = static_cast<double>(angvel_norm);
        if (mode == kFeatureCombined10Traj && kInputDim == 10) {
            fill_trajectory_onehot(trajectory, in + 6);
        }
        return;
    }

    if (mode == kFeatureAuthority4 && kInputDim == 4) {
        const tinytype cos_tilt = compute_cos_tilt_from_state(x);
        const tinytype az_max = compute_az_max_from_state(x, az_coeff, gravity_comp);
        in[0] = static_cast<double>(x(2));
        in[1] = static_cast<double>(x(8));
        in[2] = static_cast<double>(cos_tilt);
        in[3] = static_cast<double>(az_max);
        return;
    }

    if (mode == kFeatureAuthority5 && kInputDim == 5) {
        const tinytype cos_tilt = compute_cos_tilt_from_state(x);
        const tinytype az_max = compute_az_max_from_state(x, az_coeff, gravity_comp);
        in[0] = static_cast<double>(x(2));
        in[1] = static_cast<double>(x(8));
        in[2] = static_cast<double>(cos_tilt);
        in[3] = static_cast<double>(az_max);
        in[4] = static_cast<double>(wind_scale);
        return;
    }

    if (mode == kFeatureAttitude10 && kInputDim == 10) {
        const tinytype az_max = compute_az_max_from_state(x, az_coeff, gravity_comp);
        in[0] = static_cast<double>(x(2));
        in[1] = static_cast<double>(x(8));
        in[2] = static_cast<double>(x(3));
        in[3] = static_cast<double>(x(4));
        in[4] = static_cast<double>(x(9));
        in[5] = static_cast<double>(x(10));
        in[6] = static_cast<double>(x(11));
        in[7] = static_cast<double>(az_max);
        in[8] = static_cast<double>(wind_scale);
        in[9] = static_cast<double>(u_context);
        return;
    }

    if (mode == kFeatureLegacy4 || (mode != kFeatureQuad8 && kInputDim <= 4)) {
        if (kInputDim >= 1) {
            in[0] = static_cast<double>(x(2));
        }
        if (kInputDim >= 2) {
            in[1] = static_cast<double>(x(8));
        }
        if (kInputDim >= 3) {
            in[2] = static_cast<double>(wind_scale);
        }
        if (kInputDim >= 4) {
            in[3] = static_cast<double>(u_context);
        }
        return;
    }

    if (kInputDim >= 1) {
        in[0] = static_cast<double>(x(0));
    }
    if (kInputDim >= 2) {
        in[1] = static_cast<double>(x(1));
    }
    if (kInputDim >= 3) {
        in[2] = static_cast<double>(x(2));
    }
    if (kInputDim >= 4) {
        in[3] = static_cast<double>(x(6));
    }
    if (kInputDim >= 5) {
        in[4] = static_cast<double>(x(7));
    }
    if (kInputDim >= 6) {
        in[5] = static_cast<double>(x(8));
    }
    if (kInputDim >= 7) {
        in[6] = static_cast<double>(wind_scale);
    }
    if (kInputDim >= 8) {
        in[7] = static_cast<double>(u_context);
    }
}

// Recoverability MLP and analytic backpropagation to the current state.
static BarrierEval eval_learned_barrier(
    const tiny_VectorNx &x,
    tinytype wind_scale,
    tinytype u_context,
    tinytype az_coeff,
    tinytype gravity_comp,
    const std::string &trajectory,
    tiny_VectorNx *grad_x_out = nullptr,
    tinytype fail_roll_rad = 0.0,
    tinytype fail_pitch_rad = 0.0,
    tinytype fail_angvel_norm = 0.0
) {
    using namespace learned_floor_barrier;

#ifdef LEARNED_FLOOR_BARRIER_SAFE_THRESHOLD
    constexpr double kRuntimeSafeThreshold = LEARNED_FLOOR_BARRIER_SAFE_THRESHOLD;
#else
    constexpr double kRuntimeSafeThreshold = 0.0;
#endif

    double in[kInputDim];
    fill_barrier_input(x, wind_scale, u_context, az_coeff, gravity_comp, trajectory, in,
                       fail_roll_rad, fail_pitch_rad, fail_angvel_norm);
    for (int i = 0; i < kInputDim; ++i) {
        {
            in[i] = (in[i] - kInputMean[i]) / kInputStd[i];
        }
    }

    double a1[kHidden1];
    for (int j = 0; j < kHidden1; ++j) {
        {
            double s = kB1[j];
            for (int i = 0; i < kInputDim; ++i) {
                s += in[i] * kW1[i * kHidden1 + j];
            }
            a1[j] = std::tanh(s);
        }
    }

    double a2[kHidden2];
    for (int j = 0; j < kHidden2; ++j) {
        {
            double s = kB2[j];
            for (int i = 0; i < kHidden1; ++i) {
                s += a1[i] * kW2[i * kHidden2 + j];
            }
            a2[j] = std::tanh(s);
        }
    }

    double y = 0.0;
    {
        y = kB3;
        for (int i = 0; i < kHidden2; ++i) {
            y += a2[i] * kW3[i];
        }
    }

    double dy_dpre2[kHidden2];
    for (int j = 0; j < kHidden2; ++j) {
        {
            dy_dpre2[j] = kW3[j] * (1.0 - a2[j] * a2[j]);
        }
    }

    double dy_dpre1[kHidden1];
    for (int j = 0; j < kHidden1; ++j) {
        {
            double dy_dh1 = 0.0;
            for (int k = 0; k < kHidden2; ++k) {
                dy_dh1 += dy_dpre2[k] * kW2[j * kHidden2 + k];
            }
            dy_dpre1[j] = dy_dh1 * (1.0 - a1[j] * a1[j]);
        }
    }

    double dy_din[kInputDim];
    for (int i = 0; i < kInputDim; ++i) {
        {
            double s = 0.0;
            for (int j = 0; j < kHidden1; ++j) {
                s += dy_dpre1[j] * kW1[i * kHidden1 + j];
            }
            dy_din[i] = s;
        }
    }

    BarrierEval out;
    out.h = (static_cast<tinytype>(y - kRuntimeSafeThreshold));
    const auto normalized_gradient = [&](int index) -> tinytype {

        return static_cast<tinytype>(dy_din[index] / kInputStd[index]);
    };
    tiny_VectorNx grad_x = tiny_VectorNx::Zero();
    const int mode = barrier_feature_mode();
    if ((mode == kFeatureCombined6 && kInputDim == 6) ||
        (mode == kFeatureCombined10Traj && kInputDim == 10)) {
        const tinytype roll = x(3);
        const tinytype pitch = x(4);
        const tinytype p = x(9);
        const tinytype q = x(10);
        const tinytype r = x(11);
        const tinytype dcos_droll = -std::sin(roll) * std::cos(pitch);
        const tinytype dcos_dpitch = -std::cos(roll) * std::sin(pitch);

        const tinytype dh_dz = normalized_gradient(0);
        const tinytype dh_dvz = normalized_gradient(1);
        const tinytype dh_dcos = normalized_gradient(2);
        const tinytype dh_daz = normalized_gradient(3);
        const tinytype dh_dtm = normalized_gradient(4);
        const tinytype dh_dan = normalized_gradient(5);

        const tinytype cos_az_coeff = dh_dcos + dh_daz * az_coeff;
        grad_x(2) = dh_dz;
        grad_x(8) = dh_dvz;
        grad_x(3) = cos_az_coeff * dcos_droll;
        grad_x(4) = cos_az_coeff * dcos_dpitch;

        const tinytype m_roll = fail_roll_rad - std::abs(roll);
        const tinytype m_pitch = fail_pitch_rad - std::abs(pitch);
        if (m_roll <= m_pitch) {
            grad_x(3) += dh_dtm * ((roll >= 0) ? static_cast<tinytype>(-1) : static_cast<tinytype>(1));
        } else {
            grad_x(4) += dh_dtm * ((pitch >= 0) ? static_cast<tinytype>(-1) : static_cast<tinytype>(1));
        }

        const tinytype angvel_n = std::sqrt(p * p + q * q + r * r);
        if (angvel_n > 1e-6) {
            const tinytype inv_n = static_cast<tinytype>(1.0) / angvel_n;
            grad_x(9) = dh_dan * p * inv_n;
            grad_x(10) = dh_dan * q * inv_n;
            grad_x(11) = dh_dan * r * inv_n;
        }
    } else if (mode == kFeatureAuthority4 && kInputDim == 4) {
        const tinytype roll = x(3);
        const tinytype pitch = x(4);
        const tinytype dcos_droll = -std::sin(roll) * std::cos(pitch);
        const tinytype dcos_dpitch = -std::cos(roll) * std::sin(pitch);

        const tinytype dh_dz = normalized_gradient(0);
        const tinytype dh_dvz = normalized_gradient(1);
        const tinytype dh_dcos = normalized_gradient(2);
        const tinytype dh_daz = normalized_gradient(3);
        const tinytype coeff = dh_dcos + dh_daz * az_coeff;

        grad_x(2) = dh_dz;
        grad_x(8) = dh_dvz;
        grad_x(3) = coeff * dcos_droll;
        grad_x(4) = coeff * dcos_dpitch;
    } else if (mode == kFeatureAuthority5 && kInputDim == 5) {
        const tinytype roll = x(3);
        const tinytype pitch = x(4);
        const tinytype dcos_droll = -std::sin(roll) * std::cos(pitch);
        const tinytype dcos_dpitch = -std::cos(roll) * std::sin(pitch);

        const tinytype dh_dz = normalized_gradient(0);
        const tinytype dh_dvz = normalized_gradient(1);
        const tinytype dh_dcos = normalized_gradient(2);
        const tinytype dh_daz = normalized_gradient(3);
        const tinytype coeff = dh_dcos + dh_daz * az_coeff;

        grad_x(2) = dh_dz;
        grad_x(8) = dh_dvz;
        grad_x(3) = coeff * dcos_droll;
        grad_x(4) = coeff * dcos_dpitch;
    } else if (mode == kFeatureAttitude10 && kInputDim == 10) {
        const tinytype roll = x(3);
        const tinytype pitch = x(4);
        const tinytype dcos_droll = -std::sin(roll) * std::cos(pitch);
        const tinytype dcos_dpitch = -std::cos(roll) * std::sin(pitch);
        const tinytype daz_droll = az_coeff * dcos_droll;
        const tinytype daz_dpitch = az_coeff * dcos_dpitch;

        const tinytype dh_dz = normalized_gradient(0);
        const tinytype dh_dvz = normalized_gradient(1);
        const tinytype dh_droll = normalized_gradient(2);
        const tinytype dh_dpitch = normalized_gradient(3);
        const tinytype dh_dp = normalized_gradient(4);
        const tinytype dh_dq = normalized_gradient(5);
        const tinytype dh_dr = normalized_gradient(6);
        const tinytype dh_daz = normalized_gradient(7);

        grad_x(2) = dh_dz;
        grad_x(8) = dh_dvz;
        grad_x(3) = dh_droll + dh_daz * daz_droll;
        grad_x(4) = dh_dpitch + dh_daz * daz_dpitch;
        grad_x(9) = dh_dp;
        grad_x(10) = dh_dq;
        grad_x(11) = dh_dr;
    } else if (mode == kFeatureLegacy4 || (mode != kFeatureQuad8 && kInputDim <= 4)) {
        if (kInputDim >= 1) {
            grad_x(2) = normalized_gradient(0);
        }
        if (kInputDim >= 2) {
            grad_x(8) = normalized_gradient(1);
        }
    } else {
        const int feature_to_state[6] = {0, 1, 2, 6, 7, 8};
        const int n = std::min(6, kInputDim);
        for (int i = 0; i < n; ++i) {
            const int si = feature_to_state[i];
            grad_x(si) = normalized_gradient(i);
        }
    }
    if (grad_x_out) {
        *grad_x_out = grad_x;
    }
    out.dh_dz = grad_x(2);
    out.dh_dvz = grad_x(8);
    return out;
}

// Causal state/risk features and the previous scheduler output determine bounded authority.
static tinytype eval_authority_policy_quad(const tiny_VectorNx &x,
                                           const tiny_VectorNx &x_ref,
                                           const Options &opt,
                                           tinytype h_now,
                                           tinytype margin_now,
                                           tinytype sat_prox_ema,
                                           tinytype previous_w) {
    const tinytype d_auth = std::max<tinytype>(1e-6, opt.authority_deadband);
    const tinytype fixed_gate = clampv((opt.authority_tau - h_now) / d_auth, 0.0, 1.0);
    const tinytype floor_risk = clampv((opt.z_floor + 0.50 - x(2)) / 0.50, -2.0, 2.0);
    const tinytype descent_risk = clampv(-x(8) / 2.0, -2.0, 2.0);
    const tinytype xy_err = std::hypot(x(0) - x_ref(0), x(1) - x_ref(1));
    const tinytype z_err = std::abs(x(2) - x_ref(2));
    const tinytype roll_lim = std::max<tinytype>(1e-6, deg2rad(opt.fail_roll_deg));
    const tinytype pitch_lim = std::max<tinytype>(1e-6, deg2rad(opt.fail_pitch_deg));
    const tinytype tilt_risk = clampv(
        std::max(std::abs(x(3)) / roll_lim, std::abs(x(4)) / pitch_lim), 0.0, 2.0);
    const tinytype rate_norm = std::sqrt(x(9) * x(9) + x(10) * x(10) + x(11) * x(11));
    const tinytype rate_risk = clampv(
        rate_norm / std::max<tinytype>(1e-6, opt.fail_angvel_norm), 0.0, 2.0);
    const double features[kAuthorityPolicyDim] = {
        1.0,
        static_cast<double>(fixed_gate),
        static_cast<double>(clampv((margin_now - h_now) / 0.50, -2.0, 2.0)),
        static_cast<double>(floor_risk),
        static_cast<double>(descent_risk),
        static_cast<double>(clampv(xy_err / 0.75, 0.0, 2.0)),
        static_cast<double>(clampv(z_err / 0.50, 0.0, 2.0)),
        static_cast<double>(tilt_risk),
        static_cast<double>(rate_risk),
        static_cast<double>(clampv(sat_prox_ema, 0.0, 1.0)),
        static_cast<double>(clampv(previous_w, 0.0, 1.0)),
    };
    double logit = 0.0;
    for (int i = 0; i < kAuthorityPolicyDim; ++i) {
        logit += opt.authority_policy_weights[i] * features[i];
    }
    logit = std::max(-30.0, std::min(30.0, logit));
    return static_cast<tinytype>(1.0 / (1.0 + std::exp(-logit)));
}

static tinytype quantize_authority_weight(tinytype w, int levels) {
    w = clampv(w, static_cast<tinytype>(0.0), static_cast<tinytype>(1.0));
    if (levels < 2) {
        return w;
    }
    const tinytype intervals = static_cast<tinytype>(levels - 1);
    return clampv(
        std::round(w * intervals) / intervals,
        static_cast<tinytype>(0.0),
        static_cast<tinytype>(1.0));
}

static AdaptiveKnobRuntime init_adaptive_knob_runtime(const Options &opt) {
    AdaptiveKnobRuntime rt;
    rt.base_wind_assumed = opt.wind_assumed;
    rt.base_margin_scale = opt.robust_margin_scale;
    rt.base_recovery_gain = opt.recovery_collective_gain;
    rt.base_q_xy_scale = 1.0;
    rt.base_q_z_scale = 1.0;
    rt.current_wind_assumed = opt.wind_assumed;
    rt.current_margin_scale = opt.robust_margin_scale;
    rt.current_recovery_gain = opt.recovery_collective_gain;
    rt.current_q_xy_scale = rt.base_q_xy_scale;
    rt.current_q_z_scale = rt.base_q_z_scale;
    rt.applied_q_xy_scale = rt.base_q_xy_scale;
    rt.applied_q_z_scale = rt.base_q_z_scale;
    rt.target_wind_assumed = opt.wind_assumed;
    rt.target_margin_scale = opt.robust_margin_scale;
    rt.target_recovery_gain = opt.recovery_collective_gain;
    rt.target_q_xy_scale = rt.base_q_xy_scale;
    rt.target_q_z_scale = rt.base_q_z_scale;
    return rt;
}

static tinyVector scaled_state_cost(const tinyVector &base_q,
                                    tinytype q_xy_scale,
                                    tinytype q_z_scale) {
    tinyVector q = base_q;
    q(0) *= q_xy_scale;
    q(1) *= q_xy_scale;
    q(6) *= q_xy_scale;
    q(7) *= q_xy_scale;
    q(2) *= q_z_scale;
    q(8) *= q_z_scale;
    return q;
}

static void authority_scaled_costs(const tinyVector &base_q,
                                   const tinyVector &base_r,
                                   tinytype rho,
                                   tinytype adaptive_q_xy,
                                   tinytype adaptive_q_z,
                                   tinytype q_xy_scale,
                                   tinytype q_z_scale,
                                   tinytype att_scale,
                                   tinytype r_scale,
                                   tinyVector *q_work,
                                   tinyVector *r_work) {
    tinyVector q_scaled = base_q;
    q_scaled(0) *= adaptive_q_xy * q_xy_scale;
    q_scaled(1) *= adaptive_q_xy * q_xy_scale;
    q_scaled(6) *= adaptive_q_xy * q_xy_scale;
    q_scaled(7) *= adaptive_q_xy * q_xy_scale;
    q_scaled(2) *= adaptive_q_z * q_z_scale;
    q_scaled(8) *= adaptive_q_z * q_z_scale;
    q_scaled(3) *= att_scale;
    q_scaled(4) *= att_scale;
    q_scaled(9) *= att_scale;
    q_scaled(10) *= att_scale;
    q_scaled(11) *= att_scale;
    *q_work =
        q_scaled + rho * tinyMatrix::Identity(NSTATES, NSTATES).diagonal();
    *r_work =
        base_r * r_scale + rho * tinyMatrix::Identity(NINPUTS, NINPUTS).diagonal();
}

static int g_riccati_recompute_count = 0;

static void refresh_solver_state_cost(TinySolver *solver,
                                      const tinyVector &base_q,
                                      tinytype q_xy_scale,
                                      tinytype q_z_scale) {
    const tinytype rho = solver->cache->rho;
    const tinyVector q_scaled = scaled_state_cost(base_q, q_xy_scale, q_z_scale);
    solver->work->Q = (q_scaled + rho * tinyMatrix::Identity(NSTATES, NSTATES).diagonal());
    g_riccati_recompute_count += 1;
    tiny_precompute_and_set_cache(
        solver->cache,
        solver->work->Adyn,
        solver->work->Bdyn,
        solver->work->fdyn,
        solver->work->Q.asDiagonal(),
        solver->work->R.asDiagonal(),
        solver->work->nx,
        solver->work->nu,
        rho,
        0);
    if (solver->settings->adaptive_rho) {
        tiny_initialize_sensitivity_matrices(solver);
    }
}

static void refresh_solver_authority_cost(TinySolver *solver,
                                          const tinyVector &base_q,
                                          const tinyVector &base_r,
                                          tinytype adaptive_q_xy,
                                          tinytype adaptive_q_z,
                                          tinytype q_xy_scale,
                                          tinytype q_z_scale,
                                          tinytype att_scale,
                                          tinytype r_scale) {
    const tinytype rho = solver->cache->rho;
    authority_scaled_costs(
        base_q,
        base_r,
        rho,
        adaptive_q_xy,
        adaptive_q_z,
        q_xy_scale,
        q_z_scale,
        att_scale,
        r_scale,
        &solver->work->Q,
        &solver->work->R);
    g_riccati_recompute_count += 1;
    tiny_precompute_and_set_cache(
        solver->cache,
        solver->work->Adyn,
        solver->work->Bdyn,
        solver->work->fdyn,
        solver->work->Q.asDiagonal(),
        solver->work->R.asDiagonal(),
        solver->work->nx,
        solver->work->nu,
        rho,
        0);
    if (solver->settings->adaptive_rho) {
        tiny_initialize_sensitivity_matrices(solver);
    }
}

static void apply_adaptive_knobs(Options *opt,
                                 TinySolver *solver,
                                 const tinyVector &base_q,
                                 AdaptiveKnobRuntime *rt) {
    opt->wind_assumed = rt->current_wind_assumed;
    opt->robust_margin_scale = rt->current_margin_scale;
    opt->recovery_collective_gain = rt->current_recovery_gain;
    if (solver != nullptr &&
        (std::abs(rt->current_q_xy_scale - rt->applied_q_xy_scale) > 1e-6 ||
         std::abs(rt->current_q_z_scale - rt->applied_q_z_scale) > 1e-6)) {
        refresh_solver_state_cost(solver, base_q, rt->current_q_xy_scale, rt->current_q_z_scale);
        rt->applied_q_xy_scale = rt->current_q_xy_scale;
        rt->applied_q_z_scale = rt->current_q_z_scale;
    }
}

static void fill_adaptive_knob_features(const tiny_VectorNx &x,
                                        const tiny_VectorNx &x_ref,
                                        const Options &opt,
                                        tinytype h_now,
                                        tinytype margin_now,
                                        double features[7]) {
    const tinytype roll_lim = deg2rad(opt.fail_roll_deg);
    const tinytype pitch_lim = deg2rad(opt.fail_pitch_deg);
    const tinytype roll_margin = roll_lim - std::abs(x(3));
    const tinytype pitch_margin = pitch_lim - std::abs(x(4));
    const tinytype tilt_margin = std::min(roll_margin, pitch_margin);
    const tinytype p = x(9);
    const tinytype q = x(10);
    const tinytype r = x(11);
    const tinytype angvel_norm = std::sqrt(p * p + q * q + r * r);
    const tinytype xy_err = std::hypot(x(0) - x_ref(0), x(1) - x_ref(1));
    const tinytype z_err = x(2) - x_ref(2);
    const tinytype deficit = margin_now - h_now;

    features[0] = static_cast<double>(x(2));
    features[1] = static_cast<double>(x(8));
    features[2] = static_cast<double>(xy_err);
    features[3] = static_cast<double>(z_err);
    features[4] = static_cast<double>(tilt_margin);
    features[5] = static_cast<double>(angvel_norm);
    features[6] = static_cast<double>(deficit);
}

static tinytype unit_score(tinytype value, tinytype scale, tinytype cap) {
    return clampv(value / std::max<tinytype>(scale, static_cast<tinytype>(1e-9)), 0.0, cap);
}

static tinytype selected_margin_for_failure_mode(const FailureEval &failure, const Options &opt) {
    tinytype margin = failure.m_combined;
    if (opt.failure_mode == FailureMode::kFloor) {
        margin = failure.m_floor;
    } else if (opt.failure_mode == FailureMode::kAttitude) {
        margin = failure.m_safety_combined;
    }
    if (!std::isfinite(margin)) {
        margin = failure.m_safety_combined;
    }
    return margin;
}

static NoOracleQuadComponents compute_no_oracle_quad_components(
    const tiny_VectorNx &x_next,
    const tiny_VectorNx &x_nominal_next,
    const tiny_VectorNx &x_ref_next,
    const FailureEval &failure,
    const Options &opt,
    tinytype h_now,
    tinytype margin_now,
    tinytype sat_frac,
    const NoOracleQuadRuntime &rt) {
    NoOracleQuadComponents c;
    const tinytype dx = x_next(0) - x_nominal_next(0);
    const tinytype dy = x_next(1) - x_nominal_next(1);
    const tinytype dz = x_next(2) - x_nominal_next(2);
    const tinytype dvx = x_next(6) - x_nominal_next(6);
    const tinytype dvy = x_next(7) - x_nominal_next(7);
    const tinytype dvz = x_next(8) - x_nominal_next(8);
    c.pos_residual = std::sqrt(dx * dx + dy * dy);
    c.z_residual = std::abs(dz);
    c.vel_residual = std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
    c.vz_residual = std::abs(dvz);

    const tinytype droll = x_next(3) - x_nominal_next(3);
    const tinytype dpitch = x_next(4) - x_nominal_next(4);
    const tinytype dp = x_next(9) - x_nominal_next(9);
    const tinytype dq = x_next(10) - x_nominal_next(10);
    const tinytype dr = x_next(11) - x_nominal_next(11);
    c.attitude_residual = std::sqrt(droll * droll + dpitch * dpitch + 0.04 * (dp * dp + dq * dq + dr * dr));

    c.selected_margin = selected_margin_for_failure_mode(failure, opt);
    if (rt.prev_margin_valid) {
        c.margin_shrink_rate =
            std::max<tinytype>(0.0, (rt.prev_selected_margin - c.selected_margin) / 0.05);
    }
    c.barrier_deficit = std::max<tinytype>(0.0, margin_now - h_now);
    c.xy_err = std::hypot(x_next(0) - x_ref_next(0), x_next(1) - x_ref_next(1));
    c.abs_z_err = std::abs(x_next(2) - x_ref_next(2));
    c.sat_frac = sat_frac;

    const tinytype pos_score = unit_score(c.pos_residual, 0.006, 2.5);
    const tinytype z_score = unit_score(c.z_residual, 0.004, 2.0);
    const tinytype vel_score = unit_score(c.vel_residual, 0.075, 3.0);
    const tinytype vz_score = unit_score(c.vz_residual, 0.075, 3.0);
    const tinytype attitude_score = unit_score(c.attitude_residual, 0.06, 2.0);
    const tinytype low_margin_score =
        unit_score(std::max<tinytype>(0.0, 0.22 - c.selected_margin), 0.22, 2.5);
    const tinytype shrink_score = unit_score(c.margin_shrink_rate, 0.65, 2.5);
    const tinytype deficit_score = unit_score(c.barrier_deficit, 0.25, 2.0);
    const tinytype sat_score = unit_score(std::max<tinytype>(0.0, c.sat_frac - 0.50), 0.50, 1.5);
    tinytype xy_track_score = 0.0;
    if (opt.fail_track_xy_radius > 0.0) {
        xy_track_score = unit_score(
            std::max<tinytype>(0.0, c.xy_err - 0.55 * opt.fail_track_xy_radius),
            std::max<tinytype>(0.10, 0.45 * opt.fail_track_xy_radius),
            2.0);
    }
    tinytype z_track_score = 0.0;
    if (opt.fail_track_z_abs > 0.0) {
        z_track_score = unit_score(
            std::max<tinytype>(0.0, c.abs_z_err - 0.55 * opt.fail_track_z_abs),
            std::max<tinytype>(0.08, 0.45 * opt.fail_track_z_abs),
            2.0);
    }

    c.score_raw =
        0.25 * pos_score +
        0.20 * z_score +
        0.65 * vel_score +
        0.45 * vz_score +
        0.25 * attitude_score +
        0.50 * low_margin_score +
        0.45 * shrink_score +
        0.35 * deficit_score +
        0.20 * sat_score +
        0.35 * xy_track_score +
        0.30 * z_track_score;

    const bool residual_large =
        c.vel_residual >= opt.no_oracle_severe_vel_residual ||
        c.vz_residual >= opt.no_oracle_severe_vz_residual;
    c.severe_event =
        (residual_large &&
         c.score_raw >= opt.no_oracle_severe_score &&
         c.selected_margin <= opt.no_oracle_severe_margin) ? 1 : 0;
    return c;
}

// Model-residual trigger with hysteresis; measured disturbance is not an input.
static void update_no_oracle_quad_adaptive(NoOracleQuadRuntime *rt,
                                           AdaptiveKnobRuntime *knob_rt,
                                           const tiny_VectorNx &x_next,
                                           const tiny_VectorNx &x_nominal_next,
                                           const tiny_VectorNx &x_ref_next,
                                           const FailureEval &failure,
                                           const Options &opt,
                                           int step,
                                           tinytype h_now,
                                           tinytype margin_now,
                                           tinytype sat_frac,
                                           bool apply_knobs) {
    rt->components = compute_no_oracle_quad_components(
        x_next, x_nominal_next, x_ref_next, failure, opt, h_now, margin_now, sat_frac, *rt);

    const tinytype score_alpha = clampv(opt.no_oracle_score_alpha, 0.0, 1.0);
    rt->score = (step == 0)
        ? rt->components.score_raw
        : (1.0 - score_alpha) * rt->score + score_alpha * rt->components.score_raw;
    rt->score_sum += rt->score;
    rt->max_score = std::max(rt->max_score, rt->score);
    rt->residual_sum += rt->components.vel_residual;
    rt->max_vel_residual = std::max(rt->max_vel_residual, rt->components.vel_residual);

    const bool severe = rt->components.severe_event != 0;
    if (severe) {
        rt->severe_events += 1;
    }

    const bool should_enter =
        rt->score >= opt.no_oracle_enter_score ||
        rt->components.score_raw >= static_cast<tinytype>(1.25) * opt.no_oracle_enter_score ||
        severe;
    const bool should_exit = rt->score < opt.no_oracle_exit_score;
    const int requested_hold = severe
        ? std::max(std::max(0, opt.no_oracle_hold_steps), std::max(0, opt.no_oracle_severe_hold_steps))
        : std::max(0, opt.no_oracle_hold_steps);

    if (rt->active) {
        if (should_enter) {
            rt->hold_remaining = std::max(rt->hold_remaining, requested_hold);
        } else if (rt->hold_remaining > 0) {
            rt->hold_remaining -= 1;
        } else if (should_exit) {
            rt->active = false;
        }
    } else if (should_enter) {
        rt->active = true;
        rt->hold_remaining = requested_hold;
        if (rt->first_trigger_step < 0) {
            rt->first_trigger_step = step;
        }
    }

    if (rt->active) {
        rt->active_steps += 1;
    }

    if (apply_knobs && knob_rt != nullptr) {
        const tinytype target_wind = rt->active
            ? clampv(knob_rt->base_wind_assumed + opt.no_oracle_wind_delta,
                     opt.adaptive_wind_min, opt.adaptive_wind_max)
            : knob_rt->base_wind_assumed;
        const tinytype target_margin = rt->active
            ? clampv(knob_rt->base_margin_scale + opt.no_oracle_margin_delta,
                     opt.adaptive_margin_min, opt.adaptive_margin_max)
            : knob_rt->base_margin_scale;
        const tinytype target_recovery = rt->active
            ? clampv(knob_rt->base_recovery_gain + opt.no_oracle_recovery_delta,
                     opt.adaptive_recovery_min, opt.adaptive_recovery_max)
            : knob_rt->base_recovery_gain;
        const tinytype target_q_xy = rt->active
            ? clampv(knob_rt->base_q_xy_scale + opt.no_oracle_q_xy_delta,
                     opt.adaptive_q_xy_min, opt.adaptive_q_xy_max)
            : knob_rt->base_q_xy_scale;
        const tinytype target_q_z = rt->active
            ? clampv(knob_rt->base_q_z_scale + opt.no_oracle_q_z_delta,
                     opt.adaptive_q_z_min, opt.adaptive_q_z_max)
            : knob_rt->base_q_z_scale;

        const tinytype alpha = clampv(opt.adaptive_knob_smoothing, 0.0, 0.999);
        knob_rt->target_wind_assumed = target_wind;
        knob_rt->target_margin_scale = target_margin;
        knob_rt->target_recovery_gain = target_recovery;
        knob_rt->target_q_xy_scale = target_q_xy;
        knob_rt->target_q_z_scale = target_q_z;
        knob_rt->current_wind_assumed =
            alpha * knob_rt->current_wind_assumed + (1.0 - alpha) * target_wind;
        knob_rt->current_margin_scale =
            alpha * knob_rt->current_margin_scale + (1.0 - alpha) * target_margin;
        knob_rt->current_recovery_gain =
            alpha * knob_rt->current_recovery_gain + (1.0 - alpha) * target_recovery;
        knob_rt->current_q_xy_scale =
            alpha * knob_rt->current_q_xy_scale + (1.0 - alpha) * target_q_xy;
        knob_rt->current_q_z_scale =
            alpha * knob_rt->current_q_z_scale + (1.0 - alpha) * target_q_z;
        knob_rt->raw_wind_delta = rt->active ? 1.0 : 0.0;
        knob_rt->raw_margin_delta = rt->active ? 1.0 : 0.0;
        knob_rt->raw_recovery_delta = rt->active ? 1.0 : 0.0;
        knob_rt->raw_q_xy_delta = rt->active
            ? ((std::abs(opt.no_oracle_q_xy_delta) > 1e-9)
                ? (opt.no_oracle_q_xy_delta > 0.0 ? 1.0 : -1.0)
                : 0.0)
            : 0.0;
        knob_rt->raw_q_z_delta = rt->active
            ? ((std::abs(opt.no_oracle_q_z_delta) > 1e-9)
                ? (opt.no_oracle_q_z_delta > 0.0 ? 1.0 : -1.0)
                : 0.0)
            : 0.0;
        knob_rt->last_deficit = rt->components.barrier_deficit;
        knob_rt->last_update_step = step;
    }

    rt->prev_selected_margin = rt->components.selected_margin;
    rt->prev_margin_valid = true;
}

static void update_no_oracle_feature_context(Options *opt, const NoOracleQuadRuntime &rt) {
    opt->no_oracle_feat_score = rt.score;
    opt->no_oracle_feat_score_raw = rt.components.score_raw;
    opt->no_oracle_feat_active = rt.active ? 1.0 : 0.0;
    opt->no_oracle_feat_pos_residual = rt.components.pos_residual;
    opt->no_oracle_feat_vel_residual = rt.components.vel_residual;
    opt->no_oracle_feat_z_residual = rt.components.z_residual;
    opt->no_oracle_feat_vz_residual = rt.components.vz_residual;
    opt->no_oracle_feat_attitude_residual = rt.components.attitude_residual;
    opt->no_oracle_feat_selected_margin = rt.components.selected_margin;
    opt->no_oracle_feat_margin_shrink_rate = rt.components.margin_shrink_rate;
    opt->no_oracle_feat_barrier_deficit = rt.components.barrier_deficit;
    opt->no_oracle_feat_sat_frac = rt.components.sat_frac;
    opt->no_oracle_feat_severe_event = rt.components.severe_event != 0 ? 1.0 : 0.0;
}

static tinytype control_saturation_fraction(const tiny_VectorNu &u, const TinySolver *solver) {
    int sat = 0;
    for (int j = 0; j < NINPUTS; ++j) {
        const tinytype umax = solver->work->u_max(j, 0);
        if (std::abs(std::abs(u(j)) - umax) < 1e-3) {
            sat += 1;
        }
    }
    return safe_div(static_cast<tinytype>(sat), static_cast<tinytype>(NINPUTS));
}

static void fill_adaptive_obs_features_v2(const double base_features[7],
                                          tinytype sat_frac,
                                          AdaptiveObsRuntime *obs_rt,
                                          double out[11]) {
    for (int i = 0; i < 7; ++i) {
        out[i] = base_features[i];
    }
    const tinytype barrier_deficit = static_cast<tinytype>(base_features[6]);
    const tinytype z_err = static_cast<tinytype>(base_features[3]);
    if (!obs_rt->initialized) {
        obs_rt->prev_barrier_deficit = barrier_deficit;
        obs_rt->prev_z_err = z_err;
        obs_rt->prev_sat_frac = sat_frac;
        obs_rt->sat_ema = sat_frac;
        obs_rt->initialized = true;
    }
    const tinytype deficit_trend = barrier_deficit - obs_rt->prev_barrier_deficit;
    const tinytype z_err_trend = z_err - obs_rt->prev_z_err;
    const tinytype sat_trend = sat_frac - obs_rt->prev_sat_frac;
    obs_rt->sat_ema = 0.8 * obs_rt->sat_ema + 0.2 * sat_frac;
    out[7] = static_cast<double>(deficit_trend);
    out[8] = static_cast<double>(z_err_trend);
    out[9] = static_cast<double>(obs_rt->sat_ema);
    out[10] = static_cast<double>(sat_trend);
    obs_rt->prev_barrier_deficit = barrier_deficit;
    obs_rt->prev_z_err = z_err;
    obs_rt->prev_sat_frac = sat_frac;
}

static BarrierEval eval_guarded_learned_barrier_at_state(const tiny_VectorNx &x, const Options &opt) {
    const tinytype z = x(2);
    const tinytype vz = x(8);
    BarrierEval be = eval_learned_barrier(
        x,
        opt.wind_assumed,
        opt.barrier_u_context,
        opt.barrier_az_coeff,
        opt.barrier_gravity_comp,
        opt.trajectory,
        nullptr,
        deg2rad(opt.fail_roll_deg),
        deg2rad(opt.fail_pitch_deg),
        opt.fail_angvel_norm);

    if (!is_attitude10_mode()) {
        const tinytype prior_h = (z - opt.z_floor) - static_cast<tinytype>(0.20) * std::max<tinytype>(0.0, -vz);
        const tinytype prior_dh_dz = 1.0;
        const tinytype prior_dh_dvz = (vz < 0.0) ? static_cast<tinytype>(0.20) : static_cast<tinytype>(0.0);
        const tinytype alt_excess = std::max<tinytype>(0.0, z - 0.35);
        const tinytype learned_w =
            clampv(opt.learned_h_blend * std::exp(-2.8 * alt_excess), 0.05, opt.learned_h_blend);
        be.h = learned_w * be.h + (1.0 - learned_w) * prior_h;
        be.dh_dz = learned_w * be.dh_dz + (1.0 - learned_w) * prior_dh_dz;
        be.dh_dvz = learned_w * be.dh_dvz + (1.0 - learned_w) * prior_dh_dvz;
        be.dh_dz = clampv(be.dh_dz, 0.05, 1.20);
        be.dh_dvz = clampv(be.dh_dvz, 0.0, 1.20);
    }

    const tinytype h_structural =
        (z - opt.z_floor) - static_cast<tinytype>(0.20) * std::max<tinytype>(0.0, -vz);
    const tinytype h_structural_guarded = h_structural - opt.robust_structural_guard_relax;
    be.h = std::max(be.h, h_structural_guarded);
    return be;
}

static tinytype barrier_margin(const Options &opt, tinytype z, tinytype vz, int stage) {
    const tinytype high_wind = std::max<tinytype>(0.0, opt.wind_assumed - 1.35);
    tinytype margin = opt.barrier_base_margin;
    margin += 0.018 * opt.wind_assumed;
    margin += high_wind * opt.near_floor_margin_gain * std::max<tinytype>(0.0, 0.24 - z);
    margin += high_wind * opt.descending_margin_gain * std::max<tinytype>(0.0, -vz - 0.2);
    margin += opt.horizon_margin_rate * static_cast<tinytype>(stage);
    margin += 0.030 * high_wind;
    margin *= opt.robust_margin_scale;
    return margin;
}

static tinytype risk_adaptive_soft_tv_gamma(const Options &opt, tinytype h_now) {
    const tinytype gamma_safe = std::max<tinytype>(1e-6, opt.soft_tv_state_linear_gamma_safe);
    const tinytype gamma_risk = std::max<tinytype>(gamma_safe, opt.soft_tv_state_linear_gamma);
    const tinytype deadband = std::max<tinytype>(1e-6, opt.barrier_h_deadband);
    const tinytype risk = clampv((deadband - h_now) / deadband, 0.0, 1.0);
    return gamma_safe + (gamma_risk - gamma_safe) * risk;
}

static tinytype barrier_activation_threshold(const Options &opt, tinytype margin) {
    return std::max(opt.barrier_h_deadband, margin + opt.robust_solver_activation_slack);
}

static tinytype wrap_angle_pi(tinytype a) {
    constexpr tinytype kPi = static_cast<tinytype>(M_PI);
    constexpr tinytype kTwoPi = static_cast<tinytype>(2.0 * M_PI);
    while (a > kPi) {
        a -= kTwoPi;
    }
    while (a < -kPi) {
        a += kTwoPi;
    }
    return a;
}

static tiny_VectorNx propagate_nonlinear_plant(
    const tiny_VectorNx &x,
    const tiny_VectorNu &u,
    const tiny_VectorNu &wind_world,
    const Options &opt,
    const tinyMatrix &Adyn,
    const tinyMatrix &Bdyn) {
    constexpr tinytype dt = static_cast<tinytype>(0.05);
    constexpr tinytype half_dt2 = static_cast<tinytype>(0.5) * dt * dt;
    constexpr tinytype kGravity = static_cast<tinytype>(9.81);

    tiny_VectorNx x_next = x;

    const tinytype roll = x(3);
    const tinytype pitch = x(4);
    const tinytype yaw = x(5);
    const tinytype p = x(9);
    const tinytype q = x(10);
    const tinytype r = x(11);

    tinytype thrust_accel = 0.0;
    for (int j = 0; j < NINPUTS; ++j) {
        thrust_accel += (Bdyn(8, j) / std::max<tinytype>(1e-9, dt)) * u(j);
    }
    const tinytype mass_scale = std::max<tinytype>(1e-6, opt.plant_mass_scale);
    thrust_accel /= mass_scale;

    const tinytype cphi = std::cos(roll);
    const tinytype sphi = std::sin(roll);
    const tinytype cth = std::cos(pitch);
    const tinytype sth = std::sin(pitch);
    const tinytype cpsi = std::cos(yaw);
    const tinytype spsi = std::sin(yaw);

    tiny_Vector3 b3_w;
    b3_w(0) = cpsi * sth * cphi + spsi * sphi;
    b3_w(1) = spsi * sth * cphi - cpsi * sphi;
    b3_w(2) = cth * cphi;

    tiny_Vector3 accel_world;
    accel_world(0) = kGravity * b3_w(0) + thrust_accel * b3_w(0) + wind_world(0);
    accel_world(1) = kGravity * b3_w(1) + thrust_accel * b3_w(1) + wind_world(1);
    accel_world(2) = kGravity * (b3_w(2) - static_cast<tinytype>(1.0)) +
                     thrust_accel * b3_w(2) + wind_world(2);
    if (opt.plant_drag_coeff > 0.0) {
        accel_world(0) -= opt.plant_drag_coeff * x(6);
        accel_world(1) -= opt.plant_drag_coeff * x(7);
        accel_world(2) -= opt.plant_drag_coeff * x(8);
    }

    x_next(0) = x(0) + dt * x(6) + half_dt2 * accel_world(0);
    x_next(1) = x(1) + dt * x(7) + half_dt2 * accel_world(1);
    x_next(2) = x(2) + dt * x(8) + half_dt2 * accel_world(2);
    x_next(6) = x(6) + dt * accel_world(0);
    x_next(7) = x(7) + dt * accel_world(1);
    x_next(8) = x(8) + dt * accel_world(2);

    tinytype p_dot = 0.0;
    tinytype q_dot = 0.0;
    tinytype r_dot = 0.0;
    for (int j = 0; j < NINPUTS; ++j) {
        p_dot += (Bdyn(9, j) / std::max<tinytype>(1e-9, dt)) * u(j);
        q_dot += (Bdyn(10, j) / std::max<tinytype>(1e-9, dt)) * u(j);
        r_dot += (Bdyn(11, j) / std::max<tinytype>(1e-9, dt)) * u(j);
    }
    const tinytype p_next = p + dt * p_dot;
    const tinytype q_next = q + dt * q_dot;
    const tinytype r_next = r + dt * r_dot;

    const tinytype cth_safe = (std::abs(cth) < 1e-4) ? ((cth >= 0.0) ? 1e-4 : -1e-4) : cth;
    const tinytype tan_th = sth / cth_safe;
    const tinytype roll_dot = p + sphi * tan_th * q + cphi * tan_th * r;
    const tinytype pitch_dot = cphi * q - sphi * r;
    const tinytype yaw_dot = (sphi / cth_safe) * q + (cphi / cth_safe) * r;

    x_next(3) = wrap_angle_pi(x(3) + dt * roll_dot);
    x_next(4) = wrap_angle_pi(x(4) + dt * pitch_dot);
    x_next(5) = wrap_angle_pi(x(5) + dt * yaw_dot);
    x_next(9) = p_next;
    x_next(10) = q_next;
    x_next(11) = r_next;

    return x_next;
}

static std::vector<tiny_VectorNu> build_base_wind_sequence(const Options &opt, int steps) {
    std::vector<tiny_VectorNu> seq(static_cast<size_t>(steps), tiny_VectorNu::Zero());

    std::mt19937 rng(static_cast<uint32_t>(opt.seed) + 7919u);
    raya::sampling::UniformReal phase_dist(0.0, 2.0 * M_PI);
    raya::sampling::UniformReal base_z_dist(-0.45, -0.18);
    raya::sampling::UniformReal gust_amp_dist(0.75, 1.55);
    raya::sampling::UniformInt gust_start_dist(8, std::max(9, steps - 30));
    raya::sampling::UniformInt gust_len_dist(10, 28);
    raya::sampling::UniformReal piecewise_xy_dist(-0.22, 0.22);
    raya::sampling::UniformInt hold_dist(8, 20);

    const tinytype phase_x = phase_dist(rng);
    const tinytype phase_y = phase_dist(rng);
    const tinytype base_z = base_z_dist(rng);

    std::vector<tinytype> gust_profile(static_cast<size_t>(steps), 0.0);
    int num_gusts = 3;
    if (opt.wind_profile == WindProfile::kShiftTurb) {
        num_gusts = 5;
    }
    for (int g = 0; g < num_gusts; ++g) {
        const int start = gust_start_dist(rng);
        const int len = gust_len_dist(rng);
        tinytype amp = gust_amp_dist(rng);
        if (opt.wind_profile == WindProfile::kShiftFlip) {
            amp *= 1.15;
            if ((g % 2) == 1) {
                amp = -0.65 * amp;
            }
        }
        const int end = std::min(steps, start + len);
        for (int k = start; k < end; ++k) {
            const tinytype phase = static_cast<tinytype>(k - start) / std::max(1, len - 1);
            const tinytype env = (phase < 0.5) ? (2.0 * phase) : (2.0 * (1.0 - phase));
            gust_profile[static_cast<size_t>(k)] -= amp * env;
        }
    }

    tinytype piecewise_x = 0.0;
    tinytype piecewise_y = 0.0;
    int hold_counter = 0;
    for (int k = 0; k < steps; ++k) {
        if (hold_counter <= 0) {
            piecewise_x = piecewise_xy_dist(rng);
            piecewise_y = piecewise_xy_dist(rng);
            hold_counter = hold_dist(rng);
        }
        hold_counter -= 1;

        tiny_VectorNu w = tiny_VectorNu::Zero();
        const tinytype tk = static_cast<tinytype>(k);
        tinytype freq_scale = 1.0;
        if (opt.wind_profile == WindProfile::kShiftTurb) {
            freq_scale = 1.9;
        }

        w(0) = opt.wind_actual * (0.18 * std::sin(freq_scale * 0.07 * tk + phase_x)
                                  + 0.06 * std::sin(freq_scale * 0.19 * tk + 0.4 * phase_x)
                                  + 0.04 * piecewise_x);
        w(1) = opt.wind_actual * (0.14 * std::cos(freq_scale * 0.05 * tk + phase_y)
                                  + 0.06 * std::cos(freq_scale * 0.17 * tk + 0.5 * phase_y)
                                  + 0.04 * piecewise_y);
        w(2) = opt.wind_actual * (base_z + gust_profile[static_cast<size_t>(k)]);

        if (opt.wind_profile == WindProfile::kShiftFlip && k > (steps / 2)) {
            w(0) *= -1.0;
            w(1) *= -1.0;
            w(2) += 0.25 * opt.wind_actual;
        }

        w(0) = clampv(w(0), -opt.wind_clip_xy, opt.wind_clip_xy);
        w(1) = clampv(w(1), -opt.wind_clip_xy, opt.wind_clip_xy);
        w(2) = clampv(w(2), -opt.wind_clip_z, opt.wind_clip_z);
        seq[static_cast<size_t>(k)] = w;
    }
    return seq;
}

static int set_relaxed_tv_constraints(TinySolver *solver) {
    const int N = solver->work->N;
    const int nx = solver->work->nx;
    const int nu = solver->work->nu;
    tinyMatrix tv_Alin_x = tinyMatrix::Zero(N, nx);
    tinyMatrix tv_blin_x = tinyMatrix::Constant(1, N, 1e6);
    tinyMatrix tv_Alin_u = tinyMatrix::Zero(N - 1, nu);
    tinyMatrix tv_blin_u = tinyMatrix::Constant(1, N - 1, 1e6);
    return tiny_set_tv_linear_constraints(solver, tv_Alin_x, tv_blin_x, tv_Alin_u, tv_blin_u);
}

static tinytype tv_state_min_slack(const TinySolver *solver) {
    if (solver == nullptr || solver->work == nullptr || solver->settings == nullptr ||
        solver->settings->en_tv_state_linear == 0 || solver->work->N <= 0 ||
        solver->work->numtvStateLinear <= 0) {
        return std::numeric_limits<tinytype>::quiet_NaN();
    }
    tinytype min_slack = std::numeric_limits<tinytype>::infinity();
    bool found = false;
    for (int i = 0; i < solver->work->N; ++i) {
        for (int k = 0; k < solver->work->numtvStateLinear; ++k) {
            const tinyVector a = solver->work->tv_Alin_x.row((solver->work->numtvStateLinear * i) + k);
            const tinytype b = solver->work->tv_blin_x(k, i);
            if (b > static_cast<tinytype>(1e5) || a.squaredNorm() <= static_cast<tinytype>(1e-12)) {
                continue;
            }
            min_slack = std::min<tinytype>(min_slack, b - a.dot(solver->work->x.col(i)));
            found = true;
        }
    }
    return found ? min_slack : std::numeric_limits<tinytype>::quiet_NaN();
}

static tinytype tv_input_min_slack(const TinySolver *solver) {
    if (solver == nullptr || solver->work == nullptr || solver->settings == nullptr ||
        solver->settings->en_tv_input_linear == 0 || solver->work->N <= 1 ||
        solver->work->numtvInputLinear <= 0) {
        return std::numeric_limits<tinytype>::quiet_NaN();
    }
    tinytype min_slack = std::numeric_limits<tinytype>::infinity();
    bool found = false;
    for (int i = 0; i < solver->work->N - 1; ++i) {
        for (int k = 0; k < solver->work->numtvInputLinear; ++k) {
            const tinyVector a = solver->work->tv_Alin_u.row((solver->work->numtvInputLinear * i) + k);
            const tinytype b = solver->work->tv_blin_u(k, i);
            if (b > static_cast<tinytype>(1e5) || a.squaredNorm() <= static_cast<tinytype>(1e-12)) {
                continue;
            }
            min_slack = std::min<tinytype>(min_slack, b - a.dot(solver->work->u.col(i)));
            found = true;
        }
    }
    return found ? min_slack : std::numeric_limits<tinytype>::quiet_NaN();
}

static int update_barrier_constraints(TinySolver *solver, const Options &opt,
                                      const tiny_VectorNx &x_now,
                                      tinytype *h_now, tinytype *margin_now,
                                      LearnedResidualDiagnostics *residual_diag,
                                      int rollout_step) {
    const int N = solver->work->N;
    const int nx = solver->work->nx;
    const int nu = solver->work->nu;

    if (N < NHORIZON || N <= 1 || nx <= 0 || nu <= 0) {
        if (h_now) {
            *h_now = 0.0;
        }
        if (margin_now) {
            *margin_now = 0.0;
        }
        return set_relaxed_tv_constraints(solver);
    }

    tinyMatrix tv_Alin_x = tinyMatrix::Zero(N, nx);
    tinyMatrix tv_blin_x = tinyMatrix::Constant(1, N, 1e6);
    tinyMatrix tv_Alin_u = tinyMatrix::Zero(N - 1, nu);
    tinyMatrix tv_blin_u = tinyMatrix::Constant(1, N - 1, 1e6);

    const int active_horizon = std::max(1, std::min(opt.barrier_active_horizon, N));
    tinytype h0 = 0.0;
    tinytype m0 = 0.0;
    if (residual_diag) {
        *residual_diag = LearnedResidualDiagnostics{};
    }

    for (int i = 0; i < active_horizon; ++i) {
        const tiny_VectorNx xbar_plan = solver->work->x.col(i);
        const tiny_VectorNx xbar = (i == 0) ? x_now : xbar_plan;

        const tinytype z = xbar(2);
        const tinytype vz = xbar(8);
        const tinytype z_gate = (i == 0) ? x_now(2) : z;
        const tinytype vz_gate = (i == 0) ? x_now(8) : vz;
        tiny_VectorNx grad = tiny_VectorNx::Zero();
        BarrierEval be;

        {
            be = eval_learned_barrier(
                xbar,
                opt.wind_assumed,
                opt.barrier_u_context,
                opt.barrier_az_coeff,
                opt.barrier_gravity_comp,
                opt.trajectory,
                &grad,
                deg2rad(opt.fail_roll_deg),
                deg2rad(opt.fail_pitch_deg),
                opt.fail_angvel_norm
            );
        }
        for (int j = 0; j < nx; ++j) {
            grad(j) = clampv(grad(j), -2.0, 2.0);
        }
        if ((!is_attitude10_mode())) {
            const tinytype prior_h = (z - opt.z_floor) - 0.20 * std::max<tinytype>(0.0, -vz);
            const tinytype prior_dh_dz = 1.0;
            const tinytype prior_dh_dvz = (vz < 0.0) ? 0.20 : 0.0;

            const tinytype alt_excess = std::max<tinytype>(0.0, z - 0.35);
            const tinytype learned_w =
                clampv(opt.learned_h_blend * std::exp(-2.8 * alt_excess), 0.05, opt.learned_h_blend);
            be.h = learned_w * be.h + (1.0 - learned_w) * prior_h;
            be.dh_dz = learned_w * be.dh_dz + (1.0 - learned_w) * prior_dh_dz;
            be.dh_dvz = learned_w * be.dh_dvz + (1.0 - learned_w) * prior_dh_dvz;
            be.dh_dz = clampv(be.dh_dz, 0.05, 1.20);
            be.dh_dvz = clampv(be.dh_dvz, 0.0, 1.20);
            grad(2) = be.dh_dz;
            grad(8) = be.dh_dvz;
        } else {
            be.dh_dz = grad(2);
            be.dh_dvz = grad(8);
        }

        const tinytype h_analytic_pre_residual = be.h;
        const tinytype margin = barrier_margin(opt, z_gate, vz_gate, i);
        const tinytype h_structural =
            (z_gate - opt.z_floor) - 0.20 * std::max<tinytype>(0.0, -vz_gate);
        const tinytype h_structural_guarded = h_structural - opt.robust_structural_guard_relax;
        const tinytype h_analytic_gate = (std::max(h_analytic_pre_residual, h_structural_guarded));
        const bool skipped_high_alt =
            (opt.barrier_skip_high_alt) && z_gate > opt.barrier_skip_z && vz_gate > opt.barrier_skip_vz;
        const bool active_before =
            (!skipped_high_alt) && (h_analytic_gate < barrier_activation_threshold(opt, margin));

        tinytype delta_h_model = 0.0;
        tinytype delta_h_raw = 0.0;
        tinytype delta_h_clipped = 0.0;

        const tinytype h_gate = (std::max(be.h, h_structural_guarded));
        const bool active_after =
            (!skipped_high_alt) && (h_gate < barrier_activation_threshold(opt, margin));
        const bool rhs_changed_by_residual =
            (!skipped_high_alt) && (active_before || active_after) &&
            (std::abs(be.h - h_analytic_pre_residual) > static_cast<tinytype>(1e-9));
        const bool changed_by_residual =
            (!skipped_high_alt) &&
            (rhs_changed_by_residual ||
             std::abs(h_gate - h_analytic_gate) > static_cast<tinytype>(1e-9) ||
             active_before != active_after);
        if (residual_diag) {
            residual_diag->active_count_before += active_before ? 1 : 0;
            residual_diag->active_count_after += active_after ? 1 : 0;
            if (changed_by_residual && residual_diag->first_changed_stage < 0) {
                residual_diag->first_changed_stage = i;
            }
            if (i == 0) {
                residual_diag->delta_h_model = delta_h_model;
                residual_diag->delta_h_raw = delta_h_raw;
                residual_diag->delta_h_clipped = delta_h_clipped;
                residual_diag->h_analytic = h_analytic_pre_residual;
                residual_diag->h_learned = be.h;
                residual_diag->margin = margin;
                residual_diag->constraint_active_before = active_before ? 1 : 0;
                residual_diag->constraint_active_after = active_after ? 1 : 0;
            }
        }

        if (skipped_high_alt) {
            tv_blin_x(0, i) = 1e6;
            if (i == 0) {
                h0 = h_gate;
                m0 = margin;
            }
            continue;
        }

        if (h_gate >= barrier_activation_threshold(opt, margin)) {
            tv_blin_x(0, i) = 1e6;
            if (i == 0) {
                h0 = h_gate;
                m0 = margin;
            }
            continue;
        }

        tv_Alin_x.row(i) = (-grad).transpose();
        tv_blin_x(0, i) = be.h - grad.dot(xbar) - margin;

        if (opt.enable_recovery_input_constraint && i < std::max(0, opt.recovery_stages)) {
            const tinytype risk_h = std::max<tinytype>(0.0, margin - be.h);
            const tinytype risk_z = std::max<tinytype>(0.0, 0.22 - z);
            const tinytype risk_vz = std::max<tinytype>(0.0, -vz - 0.45);
            const tinytype risk = risk_h + 1.2 * risk_z + 0.35 * risk_vz;
            if (risk > 1e-6) {
                const tinytype high_wind = std::max<tinytype>(0.0, opt.wind_assumed - 1.5);
                tinytype recovery_cmd =
                    opt.recovery_collective_base +
                    opt.recovery_collective_gain * std::tanh(2.6 * risk) +
                    0.10 * high_wind;
                recovery_cmd = clampv(
                    recovery_cmd,
                    opt.recovery_collective_base,
                    std::max<tinytype>(opt.recovery_collective_base, opt.recovery_collective_max));
                tv_Alin_u.row(i).setConstant(-1.0);
                tv_blin_u(0, i) = -recovery_cmd;
            }
        }

        if (i == 0) {
            h0 = h_gate;
            m0 = margin;
        }
    }

    if (h_now) {
        *h_now = h0;
    }
    if (margin_now) {
        *margin_now = m0;
    }
    return tiny_set_tv_linear_constraints(solver, tv_Alin_x, tv_blin_x, tv_Alin_u, tv_blin_u);
}

struct SafetyInjectInfo {
    tinytype h_now = 0.0;
    tinytype margin_now = 0.0;
    LearnedResidualDiagnostics learned_diag = {};
};

// Activate learned horizon constraints when the recoverability margin becomes small.
int inject_raya_constraints(TinySolver *solver,
                          Options &opt,
                          const tiny_VectorNx &x_pred,
                          int rollout_step,
                          bool force_robust,
                          int inactive_admm_mode,
                          int robust_admm_mode,
                          tinytype inactive_abs_tol,
                          tinytype robust_abs_tol,
                          int robust_en_tv_input_linear,
                          int robust_soft_tv_state_linear,
                          SafetyInjectInfo *info) {
        if (info) {
            *info = SafetyInjectInfo{};
        }

        tinytype h_now = 0.0;
        tinytype margin_now = 0.0;
        LearnedResidualDiagnostics learned_diag;
        int status = update_barrier_constraints(
            solver, opt, x_pred, &h_now, &margin_now, &learned_diag, rollout_step);
        if (status != 0) {
            return status;
        }

        const bool barrier_active = (h_now < barrier_activation_threshold(opt, margin_now));
        if ((!barrier_active) && !force_robust) {

            solver->settings->admm_mode = inactive_admm_mode;
            solver->settings->abs_pri_tol = inactive_abs_tol;
            solver->settings->abs_dua_tol = inactive_abs_tol;
            solver->settings->en_tv_state_linear = 0;
            solver->settings->en_tv_input_linear = 0;
            solver->settings->soft_tv_state_linear = 0;
            solver->settings->soft_tv_state_linear_gamma = opt.soft_tv_state_linear_gamma_safe;
            solver->work->gl_tv.setZero();
            solver->work->yl_tv.setZero();
        } else {
            solver->settings->admm_mode = robust_admm_mode;
            solver->settings->abs_pri_tol = robust_abs_tol;
            solver->settings->abs_dua_tol = robust_abs_tol;
            solver->settings->en_tv_state_linear = 1;
            solver->settings->en_tv_input_linear = robust_en_tv_input_linear;
            solver->settings->soft_tv_state_linear = robust_soft_tv_state_linear;
            if (solver->settings->soft_tv_state_linear) {
                solver->settings->soft_tv_state_linear_gamma =
                    risk_adaptive_soft_tv_gamma(opt, h_now);
            } else {
                solver->settings->soft_tv_state_linear_gamma = opt.soft_tv_state_linear_gamma_safe;
            }
            constexpr tinytype tv_dual_decay = 0.3;
            solver->work->gl_tv *= tv_dual_decay;
            solver->work->yl_tv *= tv_dual_decay;
        }

        if (info) {
            info->h_now = h_now;
            info->margin_now = margin_now;
            info->learned_diag = learned_diag;
        }
        return 0;
    }

static bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::cout << "RAYA quadrotor simulation\nUsage: raya_quadrotor [options]\n  --seed VALUE\n  --steps VALUE\n  --constraint-stages VALUE (default 1)\n  --wind VALUE\n  --mass-scale VALUE\n  --drag VALUE\n  --trajectory figure8|circle|line_y|star|hover\n  --csv PATH\n  --wind-profile default|shift_flip|shift_turb\n";
            std::exit(0);
        }
        if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; return false; }
        const std::string value(argv[++i]);
        if (arg == "--seed") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0 || number > 2147483647 || std::floor(number) != number) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->seed = static_cast<int>(number);
            continue;
        }
        if (arg == "--constraint-stages") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 1 || number > 19 || std::floor(number) != number) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->barrier_active_horizon = static_cast<int>(number);
            continue;
        }
        if (arg == "--steps") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 1 || number > 281 || std::floor(number) != number) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->steps = static_cast<int>(number);
            continue;
        }
        if (arg == "--wind") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0 || number > 100) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->wind_actual = number;
            continue;
        }
        if (arg == "--mass-scale") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0.1 || number > 10) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->plant_mass_scale = number;
            continue;
        }
        if (arg == "--drag") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0 || number > 10) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->plant_drag_coeff = number;
            continue;
        }
        if (arg == "--trajectory") {
            if (value != "figure8" && value != "circle" && value != "line_y" && value != "star" && value != "hover") { std::cerr << "Invalid value for " << arg << "\n"; return false; }
            opt->trajectory = value; continue;
        }
        if (arg == "--csv") {
            if (value.empty()) { std::cerr << "Empty output path\n"; return false; }
            opt->step_csv = value; continue;
        }
        if (arg == "--wind-profile") {
            if (!parse_profile(value, &opt->wind_profile)) { std::cerr << "Invalid wind profile\n"; return false; }
            continue;
        }
        std::cerr << "Unknown option: " << arg << "\n"; return false;
    }
    return true;
}

// Run one closed-loop episode with the frozen controller.
int main(int argc, char **argv) {
    Options opt;
    apply_system_recovery_defaults(&opt);
    tinytype rho = rho_value;
    if (!parse_args(argc, argv, &opt)) {
        return 1;
    }
    TinySolver *solver = nullptr;
    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector fdyn = tiny_VectorNx::Zero();
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho, NSTATES, NINPUTS, NHORIZON, 0);
    if (status != 0 || solver == nullptr) {
        std::cerr << "tiny_setup failed: " << status << "\n";
        return 2;
    }

    status = apply_state_bounds(solver, opt);
    if (status != 0) {
        std::cerr << "tiny_set_bound_constraints failed: " << status << "\n";
        return 3;
    }
    finalize_barrier_runtime_params(&opt, solver);

    status = set_relaxed_tv_constraints(solver);
    if (status != 0) {
        std::cerr << "tiny_set_tv_linear_constraints init failed: " << status << "\n";
        return 4;
    }

    const tinytype inactive_abs_tol = 3e-3;
    const tinytype robust_abs_tol = 1e-2;
    const int inactive_admm_mode = TINY_ADMM_MODE_SIMPLE;
    const int robust_admm_mode = TINY_ADMM_MODE_ROBUST_LEARNED;
    const int robust_en_tv_input_linear = ((opt.enable_recovery_input_constraint)) ? 1 : 0;
    const int robust_soft_tv_state_linear = ((opt.soft_tv_state_linear)) ? 1 : 0;

    solver->settings->max_iter = opt.max_iter;
    solver->settings->check_termination = 1;
    solver->settings->abs_pri_tol = (robust_abs_tol);
    solver->settings->abs_dua_tol = (robust_abs_tol);
    solver->settings->en_tv_state_linear = (1);
    solver->settings->en_tv_input_linear = robust_en_tv_input_linear;
    solver->settings->soft_tv_state_linear = robust_soft_tv_state_linear;
    solver->settings->soft_tv_state_linear_gamma = opt.soft_tv_state_linear_gamma;
    solver->settings->adaptive_rho = (0);
    solver->settings->adaptive_rho_enable_clipping = 1;
    solver->settings->adaptive_rho_min = opt.adaptive_rho_min;
    solver->settings->adaptive_rho_max = opt.adaptive_rho_max;
    solver->settings->admm_mode = robust_admm_mode;
    if (solver->settings->adaptive_rho) {
        tiny_initialize_sensitivity_matrices(solver);
    }

    Matrix<tinytype, NSTATES, NTOTAL> Xref_total;
    build_reference_trajectory(opt, &Xref_total);

    tiny_VectorNx x = build_initial_state(opt, Xref_total);

    const std::vector<tiny_VectorNu> wind_seq = build_base_wind_sequence(opt, opt.steps + 2);

    std::ofstream step_log;
    if (!opt.step_csv.empty()) {
        step_log.open(opt.step_csv, std::ios::out);
        if (!step_log.good()) {
            std::cerr << "Failed to open step log: " << opt.step_csv << "\n";
            return 5;
        }
        step_log << "step,time,x,y,z,vx,vy,vz,roll,pitch,wx,wy,wz,ref_x,ref_y,ref_z,u0,u1,u2,u3,wind_x,wind_y,wind_z,barrier,margin,authority,trigger,solver_solved,iterations,solve_ms\n";
    }

    AdaptiveKnobRuntime adaptive_rt = init_adaptive_knob_runtime(opt);
    NoOracleQuadRuntime no_oracle_rt;

    const bool no_oracle_adaptive = true;

    {
        apply_adaptive_knobs(&opt, solver, Q, &adaptive_rt);
    }

    tinytype sat_prox_ema = 0.0;
    tinytype xy_err_ema = 0.0;
    tinytype authority_w_now = 0.0;
    tinytype authority_w_requested_now = 0.0;
    tinytype authority_policy_w_prev = 0.0;
    tinytype authority_metric_w_prev = 0.0;
    tinytype authority_w_sum = 0.0;
    tinytype authority_w_requested_sum = 0.0;
    tinytype authority_switch_sum = 0.0;
    tinytype authority_quantization_abs_error_sum = 0.0;
    tinytype authority_quantization_abs_error_max = 0.0;
    tinytype authority_q_z_applied_now = adaptive_rt.current_q_z_scale;
    tinytype authority_q_z_requested_sum = 0.0;
    tinytype authority_q_z_applied_sum = 0.0;
    tinytype authority_q_z_quantization_abs_error_sum = 0.0;
    tinytype authority_q_z_quantization_abs_error_max = 0.0;
    int authority_w_cache_index_now = -1;
    int authority_q_z_cache_index_now = -1;
    tinytype authority_cost_update_ms_sum = 0.0;
    int authority_cost_update_count = 0;

    int authority_riccati_rebuild_count = 0;
    tinytype auth_reserve_min_now = 0.0;

    Eigen::Matrix<tinytype, 4, 4> auth_mix_inv = Eigen::Matrix<tinytype, 4, 4>::Zero();
    bool auth_mix_ok = false;
    {
        Eigen::Matrix<tinytype, 4, 4> auth_mix;
        const tinytype inv_dt = static_cast<tinytype>(1.0 / 0.05);
        for (int j = 0; j < NINPUTS; ++j) {
            auth_mix(0, j) = Bdyn(8, j) * inv_dt;
            auth_mix(1, j) = Bdyn(9, j) * inv_dt;
            auth_mix(2, j) = Bdyn(10, j) * inv_dt;
            auth_mix(3, j) = Bdyn(11, j) * inv_dt;
        }
        if (std::abs(auth_mix.determinant()) > 1e-12) {
            auth_mix_inv = auth_mix.inverse();
            auth_mix_ok = true;
        }
    }

    Stats stats;
    AdaptiveObsRuntime obs_rt;

    for (int k = 0; k < opt.steps; ++k) {
        const tiny_VectorNx x_ref_now = Xref_total.col(k);
        const tiny_VectorNx x_ref_next = Xref_total.col(std::min(k + 1, NTOTAL - 1));
        solver->work->Xref = Xref_total.block<NSTATES, NHORIZON>(0, k);

        {

            apply_adaptive_knobs(
                &opt,
                (solver),
                Q,
                &adaptive_rt);
        }
        {
            update_no_oracle_feature_context(&opt, no_oracle_rt);
        }

        tiny_set_x0(solver, x);
        if (opt.reset_duals_each_step) {
            solver->work->y.setZero();
            solver->work->g.setZero();
            solver->work->yl_tv.setZero();
            solver->work->gl_tv.setZero();
        }

        tinytype h_now = 0.0;
        tinytype margin_now = 0.0;
        LearnedResidualDiagnostics learned_diag;
        SafetyInjectInfo safety_inject;
        const bool no_oracle_force_robust = (no_oracle_rt.active);
        status = inject_raya_constraints(
            solver,
            opt,
            x,
            k,
            no_oracle_force_robust,
            inactive_admm_mode,
            robust_admm_mode,
            inactive_abs_tol,
            robust_abs_tol,
            robust_en_tv_input_linear,
            robust_soft_tv_state_linear,
            &safety_inject);
        if (status != 0) {
            std::cerr << "SafetyLayer::inject_into_solve failed at step " << k << "\n";
            return 6;
        }
        h_now = safety_inject.h_now;
        margin_now = safety_inject.margin_now;
        learned_diag = safety_inject.learned_diag;

        authority_w_now = 0.0;
        authority_w_requested_now = 0.0;
        authority_q_z_applied_now = adaptive_rt.current_q_z_scale;
        authority_w_cache_index_now = -1;
        authority_q_z_cache_index_now = -1;
        if (opt.authority_beta_track > 0.0 || opt.authority_beta_reserve > 0.0 ||
            opt.authority_beta_attitude > 0.0) {
            tinytype w_auth_requested = 0.0;
            {
                const tiny_VectorNx x_ref_authority = Xref_total.col(k);
                const tinytype raw_w = eval_authority_policy_quad(
                    x, x_ref_authority, opt, h_now, margin_now,
                    sat_prox_ema, authority_policy_w_prev);
                const tinytype alpha = opt.authority_policy_smoothing;
                w_auth_requested = (k == 0)
                    ? raw_w
                    : alpha * authority_policy_w_prev + (1.0 - alpha) * raw_w;
                authority_policy_w_prev = w_auth_requested;
            }
            tinytype w_auth =
                quantize_authority_weight(w_auth_requested, opt.authority_quantization_levels);
            const tinytype adaptive_q_xy = (adaptive_rt.current_q_xy_scale);
            const tinytype adaptive_q_z = (adaptive_rt.current_q_z_scale);
            authority_q_z_applied_now = adaptive_q_z;
            const auto authority_cost_t0 = std::chrono::steady_clock::now();
            {
                const tinytype q_xy_scale = std::max<tinytype>(
                    0.1, 1.0 - opt.authority_beta_track * w_auth);
                const tinytype q_z_scale = 1.0;
                const tinytype att_scale =
                    1.0 + opt.authority_beta_attitude * w_auth;
                const tinytype r_scale =
                    1.0 + opt.authority_beta_reserve * w_auth;
                refresh_solver_authority_cost(
                    solver,
                    Q,
                    R,
                    adaptive_q_xy,
                    adaptive_q_z,
                    q_xy_scale,
                    q_z_scale,
                    att_scale,
                    r_scale);
                authority_riccati_rebuild_count += 1;
            }
            const auto authority_cost_t1 = std::chrono::steady_clock::now();
            authority_cost_update_ms_sum +=
                std::chrono::duration<double, std::milli>(
                    authority_cost_t1 - authority_cost_t0).count();
            authority_cost_update_count += 1;

            const tinytype quantization_abs_error =
                std::abs(w_auth - w_auth_requested);
            const tinytype q_z_quantization_abs_error =
                std::abs(authority_q_z_applied_now - adaptive_q_z);
            authority_w_requested_now = w_auth_requested;
            authority_w_now = w_auth;
            authority_w_requested_sum += w_auth_requested;
            authority_w_sum += w_auth;
            authority_switch_sum += std::abs(w_auth - authority_metric_w_prev);
            authority_quantization_abs_error_sum += quantization_abs_error;
            authority_quantization_abs_error_max =
                std::max(authority_quantization_abs_error_max, quantization_abs_error);
            authority_q_z_requested_sum += adaptive_q_z;
            authority_q_z_applied_sum += authority_q_z_applied_now;
            authority_q_z_quantization_abs_error_sum +=
                q_z_quantization_abs_error;
            authority_q_z_quantization_abs_error_max = std::max(
                authority_q_z_quantization_abs_error_max,
                q_z_quantization_abs_error);
            authority_metric_w_prev = w_auth;
        }
        learned_diag.tv_state_min_slack_pre = tv_state_min_slack(solver);
        learned_diag.tv_input_min_slack_pre = tv_input_min_slack(solver);

        const auto t0 = std::chrono::steady_clock::now();
        tiny_solve(solver);
        const auto t1 = std::chrono::steady_clock::now();
        const tinytype solve_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        learned_diag.tv_state_min_slack_post = tv_state_min_slack(solver);
        learned_diag.tv_input_min_slack_post = tv_input_min_slack(solver);

        stats.steps += 1;
        stats.sum_solve_ms += solve_ms;
        stats.solve_times_ms.push_back(solve_ms);
        stats.sum_iters += static_cast<tinytype>(solver->solution->iter);
        if (solver->solution->solved) {
            stats.solver_success += 1;
        } else {
            stats.solver_fail += 1;
        }

        tiny_VectorNu w = wind_seq[static_cast<size_t>(k)];
        const tinytype mass_scale = std::max<tinytype>(1e-6, opt.plant_mass_scale);
        w /= mass_scale;

        tiny_VectorNu u = solver->work->z.col(0);

        const tinytype control_l2 = u.norm();
        stats.sum_control_l2 += control_l2;
        stats.sum_control_l2_sq += u.squaredNorm();
        stats.sum_control_abs_mean += u.cwiseAbs().sum() / static_cast<tinytype>(NINPUTS);

        const tinytype sat_frac = control_saturation_fraction(u, solver);
        const tinytype sat_tol = 0.01;
        for (int j = 0; j < NINPUTS; ++j) {
            const tinytype umin = solver->work->u_min(j, 0);
            const tinytype umax = solver->work->u_max(j, 0);
            if (u(j) <= (umin + sat_tol) || u(j) >= (umax - sat_tol)) {
                stats.saturation_count += 1;
                break;
            }
        }

        const tinytype dt = 0.05;

        const tiny_VectorNx x_nominal_next = Adyn * x + Bdyn * u + fdyn;

        tiny_VectorNx x_next;
        {
            x_next = propagate_nonlinear_plant(x, u, w, opt, Adyn, Bdyn);
        }

        x = x_next;

        const tinytype xy_err = std::hypot(x(0) - x_ref_next(0), x(1) - x_ref_next(1));
        const tinytype z_err = x(2) - x_ref_next(2);
        stats.sum_xy_err += xy_err;
        stats.sum_abs_z_err += std::abs(z_err);
        stats.sum_sq_xy_err += xy_err * xy_err;
        stats.sum_sq_z_err += z_err * z_err;

        {
            tinytype sat_prox = 0.0;
            for (int j = 0; j < NINPUTS; ++j) {
                const tinytype umax = solver->work->u_max(j, 0);
                const tinytype umin = solver->work->u_min(j, 0);
                tinytype prox = 0.0;
                if (umax > 1e-9) {
                    prox = std::max(prox, u(j) / umax);
                }
                if (umin < -1e-9) {
                    prox = std::max(prox, u(j) / umin);
                }
                sat_prox = std::max(sat_prox, clampv(prox, static_cast<tinytype>(0.0),
                                                     static_cast<tinytype>(1.0)));
            }
            sat_prox_ema = 0.8 * sat_prox_ema + 0.2 * sat_prox;
            xy_err_ema = 0.8 * xy_err_ema + 0.2 * xy_err;
        }

        auth_reserve_min_now = 0.0;
        if (auth_mix_ok) {
            tinytype rmin = std::numeric_limits<tinytype>::infinity();
            for (int axis = 1; axis <= 2; ++axis) {
                for (int sgn = -1; sgn <= 1; sgn += 2) {
                    tinytype rho = std::numeric_limits<tinytype>::infinity();
                    for (int j = 0; j < NINPUTS; ++j) {
                        const tinytype dj = static_cast<tinytype>(sgn) * auth_mix_inv(j, axis);
                        if (dj > 1e-12) {
                            rho = std::min(rho, (solver->work->u_max(j, 0) - u(j)) / dj);
                        } else if (dj < -1e-12) {
                            rho = std::min(rho, (solver->work->u_min(j, 0) - u(j)) / dj);
                        }
                    }
                    rmin = std::min(rmin, std::max<tinytype>(0.0, rho));
                }
            }
            auth_reserve_min_now = std::isfinite(rmin) ? rmin : 0.0;
        }
        const FailureEval fail_eval = evaluate_failure(x, opt, &x_ref_next);
        stats.min_z = std::min(stats.min_z, x(2));
        stats.max_abs_roll_deg = std::max(stats.max_abs_roll_deg, std::abs(fail_eval.roll_deg));
        stats.max_abs_pitch_deg = std::max(stats.max_abs_pitch_deg, std::abs(fail_eval.pitch_deg));
        stats.max_angvel_norm = std::max(stats.max_angvel_norm, fail_eval.angvel_norm);
        if (fail_eval.floor_fail) {
            stats.floor_violations += 1;
        }
        if (fail_eval.attitude_fail) {
            stats.attitude_violations += 1;
        }
        if (fail_eval.position_fail) {
            stats.position_violations += 1;
        }
        if (fail_eval.combined_fail) {
            stats.combined_violations += 1;
        }

        {

            BarrierEval be;
            {
                be = eval_guarded_learned_barrier_at_state(x, opt);
            }

            h_now = be.h;
            margin_now = barrier_margin(opt, x(2), x(8), 0);
        }

        update_no_oracle_quad_adaptive(
            &no_oracle_rt,
            (&adaptive_rt),
            x,
            x_nominal_next,
            x_ref_next,
            fail_eval,
            opt,
            k,
            h_now,
            margin_now,
            sat_frac,
            no_oracle_adaptive);

        double adaptive_features[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double adaptive_features_v2[11] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        fill_adaptive_knob_features(x, x_ref_next, opt, h_now, margin_now, adaptive_features);
        fill_adaptive_obs_features_v2(adaptive_features, sat_frac, &obs_rt, adaptive_features_v2);

        if (step_log.good()) {

            step_log << std::setprecision(12)
                << k << "," << ((k + 1) * 0.05) << ","
                << x(0) << "," << x(1) << "," << x(2) << ","
                << x(6) << "," << x(7) << "," << x(8) << ","
                << x(3) << "," << x(4) << "," << x(9) << "," << x(10) << "," << x(11) << ","
                << x_ref_next(0) << "," << x_ref_next(1) << "," << x_ref_next(2) << ","
                << u(0) << "," << u(1) << "," << u(2) << "," << u(3) << ","
                << w(0) << "," << w(1) << "," << w(2) << ","
                << h_now << "," << margin_now << "," << authority_w_now << ","
                << no_oracle_rt.score << "," << solver->solution->solved << ","
                << solver->solution->iter << "," << solve_ms << "\n";
        }

        (void)x_ref_now;
    }

    const tinytype n = std::max(1, stats.steps);
    std::cout << std::setprecision(12)
              << "controller=RAYA platform=quadrotor steps=" << stats.steps << "\n"
              << "success=" << (stats.combined_violations == 0 ? 1 : 0)
              << " floor_violations=" << stats.floor_violations
              << " attitude_violations=" << stats.attitude_violations << "\n"
              << "mean_xy_err=" << stats.sum_xy_err / n
              << " mean_abs_z_err=" << stats.sum_abs_z_err / n
              << " min_z=" << stats.min_z << "\n"
              << "mean_authority_w=" << authority_w_sum / n
              << " mean_authority_switch=" << authority_switch_sum / n
              << " trigger_active_steps=" << no_oracle_rt.active_steps << "\n"
              << "mean_solve_ms=" << stats.sum_solve_ms / n
              << " mean_iters=" << stats.sum_iters / n << "\n";
    if (step_log.is_open()) {
        step_log.close();
        if (!step_log) { std::cerr << "Failed to write CSV\n"; return 5; }
    }
    return 0;
}

} // extern "C"
