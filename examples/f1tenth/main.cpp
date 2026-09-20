// RAYA: Recoverability-Aware Yielding of Authority.
// Copyright (c) 2026 RAYA contributors. SPDX-License-Identifier: MIT

#include <cstdlib>
#include "models/barrier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include "../common/random.hpp"
#include <sstream>
#include <string>
#include <vector>

#include <tinympc/tiny_api.hpp>

namespace {

constexpr int kStateDim = 5;
constexpr int kInputDim = 2;
constexpr int kHorizon = 20;
constexpr double kPi = 3.14159265358979323846;
constexpr int kAuthorityPolicyDim = 11;

enum class Scenario {
    kWall = 0,
    kGate = 1,
    kTrack = 2,
    kSlalom = 3,
    kDeadEndU = 4,
};

enum class Trajectory {
    kLineX = 0,
    kCircle = 1,
    kFigure8 = 2,
    kLineY = 3,
};

enum class FrictionProfile {
    kNone = 0,
    kSinglePatch = 1,
    kStrips = 2,
    kRandom = 3,
};

struct Obstacle {
    double x = 0.0;
    double y = 0.0;
    double radius = 0.0;
};

// Frozen RAYA configuration. CLI options change experiment conditions.
struct Options {

    std::string scenario = "track";
    std::string log_csv = "";

    double dt = 0.1;
    int steps = 400;
    int seed = 0;
    int max_iter = 80;

    double wheelbase = 0.50;
    double v_nom = 1.0;

    double v_min = 0.0;
    double v_max = 2.5;
    double delta_max = 0.6;
    double a_min = -3.0;
    double a_max = 3.0;
    double delta_dot_max = 2.5;

    double track_half_width = 1.5;
    double goal_x = 8.0;
    double goal_y = 0.0;
    double v_ref = 0.8;

    std::string trajectory = "figure8";
    double traj_circle_radius = 2.5;
    double traj_circle_omega = 0.4;
    double traj_figure8_amp_x = 4.0;
    double traj_figure8_amp_y = 1.5;
    double traj_figure8_omega = 0.15;
    double traj_line_y_offset = 0.0;
    double traj_line_y_speed = 1.0;
    double traj_center_x = 0.0;
    double traj_center_y = 0.0;

    double mu_actual = 0.3;
    std::string friction_profile = "strips";
    int n_patches = 1;
    double patch_radius = 0.8;
    double patch_first_t = 4.0;
    double patch_period = 4.0;
    double patch_seed_offset = 0.0;
    double slip_lateral_gain = 0.6;
    double plant_mass_scale = 1.0;
    double plant_drag_coeff = 0.0;
    double plant_wheelbase_scale = 1.0;

    double fail_track_radius = 0.8;
    double fail_attitude_rate = 8.0;
    double fail_severe_slip_proxy = 0.55;
    int fail_consecutive_steps = 5;

    double obs_x = 4.0;
    double obs_y = 0.0;
    double obs_radius = 0.5;
    double car_radius = 0.20;
    double obs_clearance_min = 0.05;

    double gate_obs2_x = 4.0;
    double gate_obs2_y = 0.0;
    double gate_obs2_radius = 0.0;

    double obs3_x = 0.0, obs3_y = 0.0, obs3_radius = 0.0;
    double obs4_x = 0.0, obs4_y = 0.0, obs4_radius = 0.0;
    double obs5_x = 0.0, obs5_y = 0.0, obs5_radius = 0.0;
    double obs6_x = 0.0, obs6_y = 0.0, obs6_radius = 0.0;
    double obs7_x = 0.0, obs7_y = 0.0, obs7_radius = 0.0;
    double obs8_x = 0.0, obs8_y = 0.0, obs8_radius = 0.0;

    double init_x = 0.0;
    double init_y = 0.0;
    double init_theta = 0.0;
    double init_v = 1.0;
    double init_delta = 0.0;
    double init_jitter_xy = 0.0;
    double init_jitter_theta = 0.0;

    double process_noise_std = 0.0;

    double robust_activate_h = 0.30;
    double robust_margin_base = 0.05;
    double robust_margin_gain = 0.10;

    int robust_active_horizon = 1;  // Paper-selected first predicted stage.
    double robust_solver_activation_slack = 0.01;
    double robust_structural_vdot_gain = 0.20;
    int robust_skip_safe_region = 1;
    double robust_skip_dobs = 0.40;
    double robust_skip_dobs_dot = -0.05;
    double robust_track_band = 0.40;

    double no_oracle_enter_score = 1.00;
    double no_oracle_exit_score = 0.55;
    double no_oracle_score_alpha = 0.35;
    int no_oracle_hold_steps = 20;

    double no_oracle_mu_gate = 0.58;

    double no_oracle_risk_enter = 0.55;
    double no_oracle_risk_exit = 0.35;
    double no_oracle_risk_warn = 1.10;
    double no_oracle_warning_margin_frac = 0.45;
    double no_oracle_warning_score = 3.0;
    double no_oracle_warning_mu_gate = 1.0;

    int no_oracle_severe_slip_hold_steps = -1;

    double q_x = 100.0;
    double q_y = 100.0;
    double q_theta = 50.0;
    double q_v = 10.0;
    double q_delta = 0.0;
    double r_a = 0.05;
    double r_delta = 0.15;

    double authority_beta_track = 0.8;

    double authority_tau = 0.5;
    double authority_deadband = 0.5;

    std::string authority_policy_feature_set = "no_mu";

    double authority_policy_smoothing = 0.70;
    double authority_policy_weights[kAuthorityPolicyDim] = {-6.60717933955,0.000862470812824,2.56097331488,-0.707667784364,-1.12490470595,2.4815115586,2.38019756981,2.104068676,-3.01751810442,4.18355778093,3.18824170523};

    double robust_tv_dual_decay = 0.30;
    double robust_abs_pri_tol = 5e-4;
    double robust_abs_dua_tol = 5e-4;
};

struct SafetySnapshot {
    double d_obs = 0.0;
    double d_obs_dot = 0.0;
    double d_track = 0.0;
    double d_track_dot = 0.0;
    double v_margin = 0.0;
    double delta_margin = 0.0;
    double margin_combined = 0.0;
};

double clamp_value(double x, double lo, double hi) {
    return std::min(hi, std::max(lo, x));
}

double wrap_angle(double a) {
    while (a > kPi) {
        a -= 2.0 * kPi;
    }
    while (a < -kPi) {
        a += 2.0 * kPi;
    }
    return a;
}

Scenario parse_scenario(const std::string &s) {
    if (s == "gate") return Scenario::kGate;
    if (s == "track") return Scenario::kTrack;
    if (s == "slalom") return Scenario::kSlalom;
    if (s == "dead_end_u" || s == "dead_end") return Scenario::kDeadEndU;
    return Scenario::kWall;
}

Trajectory parse_trajectory(const std::string &s) {
    if (s == "circle") return Trajectory::kCircle;
    if (s == "figure8") return Trajectory::kFigure8;
    if (s == "line_y") return Trajectory::kLineY;
    return Trajectory::kLineX;
}

FrictionProfile parse_friction_profile(const std::string &s) {
    if (s == "single_patch") return FrictionProfile::kSinglePatch;
    if (s == "strips") return FrictionProfile::kStrips;
    if (s == "random") return FrictionProfile::kRandom;
    return FrictionProfile::kNone;
}

void traj_ref(double t, const Options &opt, double *xr, double *yr, double *thr, double *vr) {
    Trajectory traj = parse_trajectory(opt.trajectory);
    if (traj == Trajectory::kCircle) {
        const double r = opt.traj_circle_radius;
        const double w = opt.traj_circle_omega;
        const double phi = w * t;
        *xr = opt.traj_center_x + r * std::sin(phi);
        *yr = opt.traj_center_y - r * (std::cos(phi) - 1.0);
        *thr = phi;
        *vr = r * w;
    } else if (traj == Trajectory::kFigure8) {
        const double ax = opt.traj_figure8_amp_x;
        const double ay = opt.traj_figure8_amp_y;
        const double w = opt.traj_figure8_omega;
        const double s = std::sin(w * t);
        const double c = std::cos(w * t);
        *xr = opt.traj_center_x + ax * s;
        *yr = opt.traj_center_y + ay * s * c;
        const double dx = ax * w * c;
        const double dy = ay * w * (c * c - s * s);
        *thr = std::atan2(dy, dx);
        *vr = std::hypot(dx, dy);
    } else if (traj == Trajectory::kLineY) {

        *xr = opt.traj_center_x + opt.traj_line_y_speed * t;
        *yr = opt.traj_center_y + opt.traj_line_y_offset;
        *thr = 0.0;
        *vr = opt.traj_line_y_speed;
    } else {

        *xr = opt.init_x + opt.v_ref * t;
        *yr = opt.init_y;
        *thr = 0.0;
        *vr = opt.v_ref;
    }
}

// Ground-truth friction belongs to the simulated plant, not to the scheduler.
double friction_mu_at(double x, double y, double t, const Options &opt) {
    FrictionProfile prof = parse_friction_profile(opt.friction_profile);
    if (prof == FrictionProfile::kNone) return 1.0;
    Trajectory traj = parse_trajectory(opt.trajectory);

    if (prof == FrictionProfile::kSinglePatch) {

        double xr, yr, thr, vr;
        traj_ref(opt.patch_first_t, opt, &xr, &yr, &thr, &vr);
        const double d = std::hypot(x - xr, y - yr);
        return d < opt.patch_radius ? opt.mu_actual : 1.0;
    }
    if (prof == FrictionProfile::kStrips) {

        double xr, yr, thr, vr;
        traj_ref(t, opt, &xr, &yr, &thr, &vr);
        const double d = std::hypot(x - xr, y - yr);
        if (d > opt.patch_radius * 1.5) return 1.0;
        const double phase = std::fmod(t + opt.patch_seed_offset,
                                       std::max(1e-3, opt.patch_period));
        const double duty = phase < 0.5 * opt.patch_period ? 1.0 : 0.0;
        return duty > 0.5 ? opt.mu_actual : 1.0;
    }
    if (prof == FrictionProfile::kRandom) {

        for (int i = 0; i < std::max(1, opt.n_patches); ++i) {
            const double tp = (i + 1) * 8.0 / (opt.n_patches + 1) +
                              0.5 * std::sin(7.0 * (i + 1) + opt.patch_seed_offset);
            double xr, yr, thr, vr;
            traj_ref(tp, opt, &xr, &yr, &thr, &vr);

            const double lateral = 0.3 * std::cos(11.0 * (i + 1) + opt.patch_seed_offset);
            const double xc = xr + lateral * std::cos(thr + 0.5 * kPi);
            const double yc = yr + lateral * std::sin(thr + 0.5 * kPi);
            const double d = std::hypot(x - xc, y - yc);
            if (d < opt.patch_radius) return opt.mu_actual;
        }
        return 1.0;
    }
    (void)traj;
    return 1.0;
}

void setup_scenario(Options *opt) {
    // Friction experiments use an open track and begin on the reference.
    opt->obs_x = 1000.0;
    opt->obs_y = 1000.0;
    opt->obs_radius = 0.0;
    opt->gate_obs2_radius = 0.0;
    opt->track_half_width = 50.0;
    double xr, yr, heading, speed;
    traj_ref(0.0, *opt, &xr, &yr, &heading, &speed);
    opt->init_x = xr;
    opt->init_y = yr;
    opt->init_theta = heading;
    opt->init_v = speed;
}

double obstacle_margin(double x, double y, const Obstacle &obs, double car_radius,
                       double clearance_min, double *grad_x, double *grad_y) {
    const double rel_x = x - obs.x;
    const double rel_y = y - obs.y;
    const double dist = std::hypot(rel_x, rel_y);
    const double margin = dist - (obs.radius + car_radius) - clearance_min;
    if (dist < 1e-9) {
        *grad_x = 0.0;
        *grad_y = 0.0;
    } else {
        *grad_x = rel_x / dist;
        *grad_y = rel_y / dist;
    }
    return margin;
}

double closest_obstacle_margin(double x, double y, const Options &opt,
                               double *grad_x, double *grad_y) {
    auto check = [&](const Obstacle &o, double *best, double *gx, double *gy) {
        if (o.radius <= 1e-9) return;
        double bx = 0.0, by = 0.0;
        const double m = obstacle_margin(x, y, o, opt.car_radius, opt.obs_clearance_min, &bx, &by);
        if (m < *best) {
            *best = m;
            *gx = bx;
            *gy = by;
        }
    };
    Obstacle a{opt.obs_x, opt.obs_y, opt.obs_radius};
    double gx = 0.0, gy = 0.0;
    double best = obstacle_margin(x, y, a, opt.car_radius, opt.obs_clearance_min, &gx, &gy);
    if (opt.obs_radius <= 1e-9) {

        best = 1.0e9;
    }
    check({opt.gate_obs2_x, opt.gate_obs2_y, opt.gate_obs2_radius}, &best, &gx, &gy);
    check({opt.obs3_x, opt.obs3_y, opt.obs3_radius}, &best, &gx, &gy);
    check({opt.obs4_x, opt.obs4_y, opt.obs4_radius}, &best, &gx, &gy);
    check({opt.obs5_x, opt.obs5_y, opt.obs5_radius}, &best, &gx, &gy);
    check({opt.obs6_x, opt.obs6_y, opt.obs6_radius}, &best, &gx, &gy);
    check({opt.obs7_x, opt.obs7_y, opt.obs7_radius}, &best, &gx, &gy);
    check({opt.obs8_x, opt.obs8_y, opt.obs8_radius}, &best, &gx, &gy);
    *grad_x = gx;
    *grad_y = gy;
    return best;
}

constexpr int kMaxLearnedFeatureDim = 32;

double ref_curvature_at(double t, const Options &opt) {
    double xr0, yr0, th0, vr0;
    double xr1, yr1, th1, vr1;
    traj_ref(t, opt, &xr0, &yr0, &th0, &vr0);
    traj_ref(t + std::max(1e-3, opt.dt), opt, &xr1, &yr1, &th1, &vr1);
    const double dtheta = wrap_angle(th1 - th0);
    return (std::abs(vr0) > 1e-3) ? (dtheta / std::max(1e-3, opt.dt) / vr0) : 0.0;
}

struct NoOracleTriggerComponents {
    double pos_residual = 0.0;
    double yaw_residual = 0.0;
    double velocity_residual = 0.0;
    double yaw_rate_residual = 0.0;
    double tube_margin = 0.0;
    double tube_margin_shrink_rate = 0.0;
    double heading_error = 0.0;
    double velocity = 0.0;
    double speed_error = 0.0;
    double yaw_rate_abs = 0.0;
    double steering_saturation = 0.0;
    double accel_saturation = 0.0;
    double brake_saturation = 0.0;
    double ref_curvature = 0.0;
    double mu_est = 1.0;
    double slip_proxy = 0.0;
    double model_residual_norm = 0.0;
    double score_raw = 0.0;
    double mu_est_min_10 = 1.0;
    double mu_est_min_30 = 1.0;
    double yaw_rate_residual_max_10 = 0.0;
    double yaw_rate_residual_max_30 = 0.0;
    double trigger_score_max_10 = 0.0;
    double tube_margin_min_10 = 0.0;
    double tube_margin_min_30 = 0.0;
    double tube_margin_shrink_rate_max_10 = 0.0;
    double time_since_severe_slip = 10.0;
    double severe_slip_count_30 = 0.0;
    double recent_severe_slip = 0.0;
};

double unit_score(double value, double scale, double cap) {
    return clamp_value(value / std::max(1e-9, scale), 0.0, cap);
}

NoOracleTriggerComponents compute_no_oracle_trigger(
    double x, double y, double theta, double v, double delta, double t, const Options &opt,
    bool prev_prediction_valid, double prev_pred_x, double prev_pred_y,
    double prev_pred_theta, double prev_pred_v, double prev_predicted_yaw_dot,
    double prev_actual_yaw_dot, bool prev_tube_margin_valid, double prev_tube_margin,
    double mu_est_ewma, double prev_a_cmd, double prev_delta_dot_cmd) {
    NoOracleTriggerComponents c;
    c.mu_est = clamp_value(mu_est_ewma, 0.05, 1.0);
    if (prev_prediction_valid) {
        c.pos_residual = std::hypot(x - prev_pred_x, y - prev_pred_y);
        c.yaw_residual = std::abs(wrap_angle(theta - prev_pred_theta));
        c.velocity_residual = std::abs(v - prev_pred_v);
        if (std::abs(prev_predicted_yaw_dot) > 0.05) {
            c.yaw_rate_residual =
                std::abs(prev_predicted_yaw_dot - prev_actual_yaw_dot) /
                std::max(0.05, std::abs(prev_predicted_yaw_dot));
        }
        c.model_residual_norm = std::max({
            c.pos_residual / 0.10,
            c.yaw_residual / 0.15,
            c.velocity_residual / 0.30,
            c.yaw_rate_residual,
        });
    }

    double xr, yr, thr, vr;
    traj_ref(t, opt, &xr, &yr, &thr, &vr);
    const double tube_err = std::hypot(x - xr, y - yr);
    c.tube_margin = opt.fail_track_radius - tube_err;
    if (prev_tube_margin_valid) {
        c.tube_margin_shrink_rate =
            std::max(0.0, (prev_tube_margin - c.tube_margin) / std::max(1e-6, opt.dt));
    }
    c.heading_error = std::abs(wrap_angle(theta - thr));
    c.velocity = v;
    c.speed_error = std::abs(v - vr);
    c.yaw_rate_abs = std::abs(prev_actual_yaw_dot);
    c.slip_proxy = std::abs(prev_predicted_yaw_dot - prev_actual_yaw_dot);
    c.steering_saturation = std::abs(delta) / std::max(1e-9, opt.delta_max);
    c.accel_saturation = std::max(0.0, prev_a_cmd) / std::max(1e-9, opt.a_max);
    c.brake_saturation = std::max(0.0, -prev_a_cmd) / std::max(1e-9, -opt.a_min);
    (void)prev_delta_dot_cmd;
    c.ref_curvature = std::abs(ref_curvature_at(t, opt));

    const double pos_score = unit_score(c.pos_residual, 0.04, 3.0);
    const double yaw_score = unit_score(c.yaw_residual, 0.06, 3.0);
    const double vel_score = unit_score(c.velocity_residual, 0.20, 2.0);
    const double yaw_rate_score = unit_score(c.yaw_rate_residual, 0.25, 3.0);
    const double slip_score = unit_score(c.slip_proxy, 0.35, 3.0);
    const double mu_score = unit_score(std::max(0.0, 0.92 - c.mu_est), 0.30, 3.0);
    const double tube_low_score =
        unit_score(std::max(0.0, 0.45 * opt.fail_track_radius - c.tube_margin),
                   0.45 * opt.fail_track_radius, 2.0);
    const double tube_shrink_score = unit_score(c.tube_margin_shrink_rate, 0.50, 2.0);
    const double heading_score = unit_score(std::max(0.0, c.heading_error - 0.15), 0.35, 2.0);
    const double speed_score = unit_score(std::max(0.0, c.speed_error - 0.25), 0.40, 2.0);
    const double steer_score =
        unit_score(std::max(0.0, c.steering_saturation - 0.55), 0.40, 1.5);
    const double curvature_score = unit_score(std::max(0.0, c.ref_curvature - 0.25), 0.40, 1.5);

    c.score_raw =
        0.45 * pos_score +
        0.35 * yaw_score +
        0.20 * vel_score +
        0.65 * yaw_rate_score +
        0.45 * slip_score +
        0.55 * mu_score +
        0.50 * tube_low_score +
        0.45 * tube_shrink_score +
        0.25 * heading_score +
        0.15 * speed_score +
        0.25 * steer_score +
        0.15 * curvature_score;
    return c;
}

double eval_no_oracle_risk_model(const NoOracleTriggerComponents &c,
                                 double trigger_score_ewma) {
    (void)c;
    (void)trigger_score_ewma;
    return -1.0;
}

// Frozen 20-feature recoverability network, including its learned normalization.
double eval_no_oracle_learned_h(const double feats[kMaxLearnedFeatureDim]) {
    using namespace car_no_oracle_barrier_params;
    static_assert(kFeatureDim <= kMaxLearnedFeatureDim,
                  "Increase kMaxLearnedFeatureDim for this no-oracle car barrier header.");
    double z[kFeatureDim];
    for (int i = 0; i < kFeatureDim; ++i) {
        const double sd = std::max(1e-9, kFeatureStd[i]);
        z[i] = (feats[i] - kFeatureMean[i]) / sd;
    }
    double h1[kHidden1];
    for (int j = 0; j < kHidden1; ++j) {
        double s = kB1[j];
        for (int i = 0; i < kFeatureDim; ++i) {
            s += z[i] * kW1[i * kHidden1 + j];
        }
        h1[j] = std::tanh(s);
    }
    double h2[kHidden2];
    for (int j = 0; j < kHidden2; ++j) {
        double s = kB2[j];
        for (int i = 0; i < kHidden1; ++i) {
            s += h1[i] * kW2[i * kHidden2 + j];
        }
        h2[j] = std::tanh(s);
    }
    double out = kB3;
    for (int i = 0; i < kHidden2; ++i) {
        out += kW3[i] * h2[i];
    }
    return out - kSafeThreshold;
}

bool no_oracle_barrier_uses_f1tenth_features() {
    return std::string(car_no_oracle_barrier_params::kFeatureSetName) == "f1tenth_no_oracle";
}

void learned_features(double x, double y, double theta, double v, double delta,
                      double t, const Options &opt,
                      double feats[kMaxLearnedFeatureDim]);
void learned_features_with_mu_feature(double x, double y, double theta, double v, double delta,
                                      double t, const Options &opt, double mu_feature,
                                      double feats[kMaxLearnedFeatureDim]);

SafetySnapshot compute_safety_snapshot(double x, double y, double theta, double v,
                                       double delta, const Options &opt) {
    SafetySnapshot s;
    double gx = 0.0, gy = 0.0;
    s.d_obs = closest_obstacle_margin(x, y, opt, &gx, &gy);

    const double vx = v * std::cos(theta);
    const double vy = v * std::sin(theta);
    s.d_obs_dot = gx * vx + gy * vy;

    const double upper_margin = opt.track_half_width - y;
    const double lower_margin = y + opt.track_half_width;
    if (lower_margin < upper_margin) {
        s.d_track = lower_margin;
        s.d_track_dot = vy;
    } else {
        s.d_track = upper_margin;
        s.d_track_dot = -vy;
    }

    s.v_margin = std::min(v - opt.v_min, opt.v_max - v);
    s.delta_margin = opt.delta_max - std::abs(delta);
    s.margin_combined = std::min({s.d_obs, s.d_track, s.v_margin, s.delta_margin});
    return s;
}

void learned_features(double x, double y, double theta, double v, double delta,
                      double t, const Options &opt,
                      double feats[kMaxLearnedFeatureDim]) {
    learned_features_with_mu_feature(
        x, y, theta, v, delta, t, opt, friction_mu_at(x, y, t, opt), feats);
}

void learned_features_with_mu_feature(double x, double y, double theta, double v, double delta,
                                      double t, const Options &opt, double mu_feature,
                                      double feats[kMaxLearnedFeatureDim]) {
    const SafetySnapshot s = compute_safety_snapshot(x, y, theta, v, delta, opt);
    for (int i = 0; i < kMaxLearnedFeatureDim; ++i) feats[i] = 0.0;
    feats[0] = s.d_obs;
    feats[1] = s.d_obs_dot;
    feats[2] = s.d_track;
    feats[3] = s.d_track_dot;
    feats[4] = s.v_margin;
    feats[5] = s.delta_margin;

    double xr, yr, thr, vr;
    traj_ref(t, opt, &xr, &yr, &thr, &vr);
    const double ex = x - xr;
    const double ey = y - yr;
    const double tube_err = std::hypot(ex, ey);
    feats[6] = opt.fail_track_radius - tube_err;

    double xrn, yrn, thrn, vrn;
    traj_ref(t + std::max(1e-3, opt.dt), opt, &xrn, &yrn, &thrn, &vrn);
    const double ref_vx = (xrn - xr) / std::max(1e-3, opt.dt);
    const double ref_vy = (yrn - yr) / std::max(1e-3, opt.dt);
    const double car_vx = v * std::cos(theta);
    const double car_vy = v * std::sin(theta);
    const double tube_err_dot =
        (tube_err > 1e-6)
            ? (ex * (car_vx - ref_vx) + ey * (car_vy - ref_vy)) / tube_err
            : 0.0;
    feats[7] = -tube_err_dot;
    feats[8] = wrap_angle(theta - thr);
    feats[9] = v - vr;
    feats[10] = ref_curvature_at(t, opt);
    feats[11] = clamp_value(mu_feature, 0.05, 1.0);
}

void f1tenth_no_oracle_features(double x, double y, double theta, double v,
                                double delta, double t, const Options &opt,
                                const NoOracleTriggerComponents *observed,
                                double feats[kMaxLearnedFeatureDim]) {
    for (int i = 0; i < kMaxLearnedFeatureDim; ++i) feats[i] = 0.0;
    double xr, yr, thr, vr;
    traj_ref(t, opt, &xr, &yr, &thr, &vr);
    const double tube_margin = opt.fail_track_radius - std::hypot(x - xr, y - yr);
    const double heading_error = std::abs(wrap_angle(theta - thr));
    const double ref_curvature_abs = std::abs(ref_curvature_at(t, opt));
    const double steering_sat = std::abs(delta) / std::max(1e-9, opt.delta_max);
    const double speed_error = std::abs(v - vr);
    feats[0] = tube_margin;
    feats[1] = heading_error;
    feats[2] = v;
    feats[3] = observed ? observed->yaw_rate_abs : 0.0;
    feats[4] = observed ? observed->yaw_rate_residual : 0.0;
    feats[5] = observed ? observed->slip_proxy : 0.0;
    feats[6] = steering_sat;
    feats[7] = observed ? observed->accel_saturation : 0.0;
    feats[8] = observed ? observed->brake_saturation : 0.0;
    feats[9] = ref_curvature_abs;
    feats[10] = observed ? observed->recent_severe_slip : 0.0;
    feats[11] = observed ? observed->severe_slip_count_30 : 0.0;
    feats[12] = observed ? observed->pos_residual : 0.0;
    feats[13] = observed ? observed->yaw_residual : 0.0;
    feats[14] = observed ? observed->velocity_residual : speed_error;
    feats[15] = observed ? observed->model_residual_norm : 0.0;
    feats[16] = observed ? observed->tube_margin_min_30 : tube_margin;
    feats[17] = observed ? observed->tube_margin_shrink_rate_max_10 : 0.0;
    feats[18] = observed ? observed->time_since_severe_slip : 10.0;
    feats[19] = observed ? observed->trigger_score_max_10 : 0.0;

    const double mu_hat = observed ? clamp_value(observed->mu_est, 0.05, 1.0) : 1.0;
    feats[20] = mu_hat;
    feats[21] = opt.fail_severe_slip_proxy -
        (1.0 - mu_hat) * (std::max(0.0, v) / std::max(1e-6, opt.wheelbase)) *
        std::abs(std::tan(delta));
}

double eval_no_oracle_learned_h_state_with_mu_feature(double x, double y, double theta,
                                                      double v, double delta, double t,
                                                      const Options &opt,
                                                      double mu_feature,
                                                      const NoOracleTriggerComponents *observed = nullptr) {
    double feats[kMaxLearnedFeatureDim];
    if (no_oracle_barrier_uses_f1tenth_features()) {
        f1tenth_no_oracle_features(x, y, theta, v, delta, t, opt, observed, feats);
    } else {
        learned_features_with_mu_feature(x, y, theta, v, delta, t, opt, mu_feature, feats);
    }
    return eval_no_oracle_learned_h(feats);
}

double eval_authority_policy_car(double x, double y, double theta, double v,
                                 double delta, double t, const Options &opt,
                                 double no_oracle_mu_feature,
                                 const NoOracleTriggerComponents &observed,
                                 double no_oracle_score,
                                 double analytic_h,
                                 double previous_w) {
    double xr, yr, thr, vr;
    traj_ref(t, opt, &xr, &yr, &thr, &vr);
    const double tube_margin = opt.fail_track_radius - std::hypot(x - xr, y - yr);
    const double fixed_gate = clamp_value(
        (opt.authority_tau - tube_margin) / std::max(1e-6, opt.authority_deadband),
        0.0, 1.0);
    const double learned_h =
        (eval_no_oracle_learned_h_state_with_mu_feature(
                  x, y, theta, v, delta, t, opt,
                  no_oracle_mu_feature, &observed));
    const double policy_h =
        (learned_h);
    const double mu_score = unit_score(
        std::max(0.0, 0.92 - observed.mu_est), 0.30, 3.0);
    const double no_mu_score = std::max(
        0.0, observed.score_raw - 0.55 * mu_score);
    const bool no_mu_features =
        opt.authority_policy_feature_set == "no_mu";
    const double features[kAuthorityPolicyDim] = {
        1.0,
        fixed_gate,
        clamp_value((opt.robust_activate_h - policy_h) / 0.50, -2.0, 2.0),
        clamp_value((opt.authority_tau - tube_margin) /
                        std::max(1e-6, opt.authority_deadband), -2.0, 2.0),
        clamp_value(observed.tube_margin_shrink_rate / 1.0, 0.0, 2.0),
        no_mu_features
            ? clamp_value(observed.model_residual_norm / 3.0, 0.0, 2.0)
            : clamp_value((1.0 - observed.mu_est) / 0.80, 0.0, 2.0),
        clamp_value(observed.yaw_rate_residual / 1.0, 0.0, 2.0),
        clamp_value(observed.slip_proxy /
                        std::max(1e-6, opt.fail_severe_slip_proxy), 0.0, 2.0),
        clamp_value(observed.steering_saturation, 0.0, 1.5),
        no_mu_features
            ? clamp_value(no_mu_score / 3.0, 0.0, 2.0)
            : clamp_value(no_oracle_score / 3.0, 0.0, 2.0),
        clamp_value(previous_w, 0.0, 1.0),
    };
    double logit = 0.0;
    for (int i = 0; i < kAuthorityPolicyDim; ++i) {
        logit += opt.authority_policy_weights[i] * features[i];
    }
    logit = std::max(-30.0, std::min(30.0, logit));
    return 1.0 / (1.0 + std::exp(-logit));
}

void relinearize_bicycle(TinySolver *solver, double x_star, double y_star,
                         double theta_star, double v_star, double delta_star,
                         const Options &opt, double mu_est = 1.0) {
    const double dt = opt.dt;
    const double L = opt.wheelbase;
    const double cth = std::cos(theta_star);
    const double sth = std::sin(theta_star);
    const double sec_d = 1.0 / std::cos(delta_star);
    const double tan_d = std::tan(delta_star);
    const double mu_clamped = clamp_value(mu_est, 0.05, 1.0);

    tinyMatrix Adyn = tinyMatrix::Identity(kStateDim, kStateDim);
    Adyn(0, 2) = -dt * v_star * sth;
    Adyn(0, 3) =  dt * cth;
    Adyn(1, 2) =  dt * v_star * cth;
    Adyn(1, 3) =  dt * sth;
    Adyn(2, 3) =  dt * mu_clamped * tan_d / L;
    Adyn(2, 4) =  dt * mu_clamped * v_star * sec_d * sec_d / L;

    tinyMatrix Bdyn = tinyMatrix::Zero(kStateDim, kInputDim);
    Bdyn(3, 0) = dt;
    Bdyn(4, 1) = dt;

    tinyVector fdyn = tinyVector::Zero(kStateDim);
    fdyn(0) = dt * v_star * sth * theta_star;
    fdyn(1) = -dt * v_star * cth * theta_star;
    fdyn(2) = -dt * v_star * sec_d * sec_d * delta_star / L;

    solver->work->Adyn = Adyn;
    solver->work->Bdyn = Bdyn;
    solver->work->fdyn = fdyn;

    tiny_precompute_and_set_cache(
        solver->cache, Adyn, Bdyn, fdyn,
        solver->work->Q.asDiagonal(), solver->work->R.asDiagonal(),
        kStateDim, kInputDim, solver->cache->rho, 0
    );
}

// Centered finite differences of the deployed recoverability margin.
void finite_diff_learned_h_grad(double x, double y, double theta, double v,
                                double delta, double t, const Options &opt,
                                double grad[5], double mu_feature_override = -1.0,
                                const NoOracleTriggerComponents *observed = nullptr) {
    const double eps_pos = 1e-3;
    const double eps_ang = 1e-3;
    const double eps_v = 5e-3;
    const double eps_d = 1e-3;
    const double s0[5] = {x, y, theta, v, delta};
    const double eps[5] = {eps_pos, eps_pos, eps_ang, eps_v, eps_d};
    for (int j = 0; j < 5; ++j) {
        double sp[5] = {s0[0], s0[1], s0[2], s0[3], s0[4]};
        double sm[5] = {s0[0], s0[1], s0[2], s0[3], s0[4]};
        sp[j] += eps[j];
        sm[j] -= eps[j];
        const bool use_override = mu_feature_override >= 0.0;
        const double hp = (((eval_no_oracle_learned_h_state_with_mu_feature(
                         sp[0], sp[1], sp[2], sp[3], sp[4], t, opt,
                         use_override ? mu_feature_override : 1.0, observed))));
        const double hm = (((eval_no_oracle_learned_h_state_with_mu_feature(
                         sm[0], sm[1], sm[2], sm[3], sm[4], t, opt,
                         use_override ? mu_feature_override : 1.0, observed))));
        grad[j] = (hp - hm) / (2.0 * eps[j]);
    }
}

double tv_state_min_slack(const TinySolver *solver) {
    if (!solver || !solver->settings || !solver->work ||
        solver->settings->en_tv_state_linear == 0 || solver->work->numtvStateLinear <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double m = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int i = 0; i < solver->work->N; ++i) {
        for (int k = 0; k < solver->work->numtvStateLinear; ++k) {
            const tinyVector a = solver->work->tv_Alin_x.row((solver->work->numtvStateLinear * i) + k);
            const double b = static_cast<double>(solver->work->tv_blin_x(k, i));
            const double slack = b - static_cast<double>(a.dot(solver->work->x.col(i)));
            m = std::min(m, slack);
            found = true;
        }
    }
    return found ? m : std::numeric_limits<double>::quiet_NaN();
}

double tv_input_min_slack(const TinySolver *solver) {
    if (!solver || !solver->settings || !solver->work ||
        solver->settings->en_tv_input_linear == 0 || solver->work->numtvInputLinear <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double m = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int i = 0; i < solver->work->N - 1; ++i) {
        for (int k = 0; k < solver->work->numtvInputLinear; ++k) {
            const tinyVector a = solver->work->tv_Alin_u.row((solver->work->numtvInputLinear * i) + k);
            const double b = static_cast<double>(solver->work->tv_blin_u(k, i));
            const double slack = b - static_cast<double>(a.dot(solver->work->u.col(i)));
            m = std::min(m, slack);
            found = true;
        }
    }
    return found ? m : std::numeric_limits<double>::quiet_NaN();
}

static bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::cout << "RAYA f1tenth simulation\nUsage: raya_f1tenth [options]\n  --seed VALUE\n  --steps VALUE\n  --constraint-stages VALUE (default 1)\n  --friction VALUE\n  --patch-period VALUE\n  --patch-phase VALUE\n  --patch-time VALUE\n  --patch-count VALUE\n  --mass-scale VALUE\n  --drag VALUE\n  --friction-profile strips|random|single_patch|none\n  --csv PATH\n";
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
            opt->robust_active_horizon = static_cast<int>(number);
            continue;
        }
        if (arg == "--steps") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 1 || number > 100000 || std::floor(number) != number) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->steps = static_cast<int>(number);
            continue;
        }
        if (arg == "--friction") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0.01 || number > 1) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->mu_actual = number;
            continue;
        }
        if (arg == "--patch-period") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0.1 || number > 100) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->patch_period = number;
            continue;
        }
        if (arg == "--patch-phase") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0 || number > 100) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->patch_seed_offset = number;
            continue;
        }
        if (arg == "--patch-time") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 0 || number > 10000) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->patch_first_t = number;
            continue;
        }
        if (arg == "--patch-count") {
            char *end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0' || !std::isfinite(number) || number < 1 || number > 100 || std::floor(number) != number) {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n"; return false;
            }
            opt->n_patches = static_cast<int>(number);
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
        if (arg == "--friction-profile") {
            if (value != "strips" && value != "random" && value != "single_patch" && value != "none") { std::cerr << "Invalid value for " << arg << "\n"; return false; }
            opt->friction_profile = value; continue;
        }
        if (arg == "--csv") {
            if (value.empty()) { std::cerr << "Empty output path\n"; return false; }
            opt->log_csv = value; continue;
        }
        std::cerr << "Unknown option: " << arg << "\n"; return false;
    }
    return true;
}

}

// Run one closed-loop episode with the frozen controller.
int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        std::cerr << "Invalid arguments.\n";
        return 1;
    }
    opt.authority_policy_smoothing = clamp_value(
        opt.authority_policy_smoothing, 0.0, 0.999);

    setup_scenario(&opt);

    tinyMatrix Adyn = tinyMatrix::Identity(kStateDim, kStateDim);
    tinyMatrix Bdyn = tinyMatrix::Zero(kStateDim, kInputDim);
    tinyVector fdyn = tinyVector::Zero(kStateDim);

    Adyn(0, 3) = opt.dt;
    Adyn(1, 2) = opt.dt * opt.v_nom;
    Adyn(2, 4) = opt.dt * opt.v_nom / opt.wheelbase;
    Bdyn(3, 0) = opt.dt;
    Bdyn(4, 1) = opt.dt;

    tinyVector q_cost = tinyVector::Zero(kStateDim);
    tinyVector r_cost = tinyVector::Zero(kInputDim);

    q_cost << opt.q_x, opt.q_y, opt.q_theta, opt.q_v, opt.q_delta;
    r_cost << opt.r_a, opt.r_delta;

    TinySolver *solver = nullptr;
    const int status = tiny_setup(
        &solver, Adyn, Bdyn, fdyn, q_cost.asDiagonal(), r_cost.asDiagonal(),
        static_cast<tinytype>(5.0), kStateDim, kInputDim, kHorizon, 0
    );
    if (status != 0 || solver == nullptr) {
        std::cerr << "tiny_setup failed.\n";
        return 1;
    }

    tinyMatrix x_min = tinyMatrix::Constant(kStateDim, kHorizon, -1.0e9);
    tinyMatrix x_max = tinyMatrix::Constant(kStateDim, kHorizon, +1.0e9);
    tinyMatrix u_min = tinyMatrix::Constant(kInputDim, kHorizon - 1, -1.0e9);
    tinyMatrix u_max = tinyMatrix::Constant(kInputDim, kHorizon - 1, +1.0e9);
    for (int i = 0; i < kHorizon; ++i) {
        x_min(3, i) = opt.v_min;
        x_max(3, i) = opt.v_max;
        x_min(4, i) = -opt.delta_max;
        x_max(4, i) = opt.delta_max;
    }
    for (int i = 0; i < kHorizon - 1; ++i) {
        u_min(0, i) = opt.a_min;
        u_max(0, i) = opt.a_max;
        u_min(1, i) = -opt.delta_dot_max;
        u_max(1, i) = opt.delta_dot_max;
    }
    tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);
    solver->settings->max_iter = opt.max_iter;
    solver->settings->check_termination = 5;
    const double inactive_abs_pri_tol = 1e-4;
    const double inactive_abs_dua_tol = 1e-4;
    solver->settings->abs_pri_tol = inactive_abs_pri_tol;
    solver->settings->abs_dua_tol = inactive_abs_dua_tol;
    solver->settings->en_tv_state_linear = 0;
    solver->settings->en_tv_input_linear = 0;

    std::mt19937 rng(static_cast<uint32_t>(opt.seed));
    raya::sampling::Normal n01(0.0, 1.0);

    tinyVector x = tinyVector::Zero(kStateDim);
    x(0) = opt.init_x + opt.init_jitter_xy * n01(rng);
    x(1) = opt.init_y + opt.init_jitter_xy * n01(rng);
    x(2) = opt.init_theta + opt.init_jitter_theta * n01(rng);
    x(3) = clamp_value(opt.init_v, opt.v_min + 1e-3, opt.v_max - 1e-3);
    x(4) = clamp_value(opt.init_delta, -opt.delta_max + 1e-3, opt.delta_max - 1e-3);

    std::ofstream csv;
    if (!opt.log_csv.empty()) {
        const std::filesystem::path path(opt.log_csv);
        if (path.has_parent_path()) {
            std::error_code error;
            std::filesystem::create_directories(path.parent_path(), error);
            if (error) {
                std::cerr << "Failed to create CSV directory: " << error.message() << "\n";
                return 5;
            }
        }
        csv.open(opt.log_csv, std::ios::out | std::ios::trunc);
        if (!csv) {
            std::cerr << "Failed to open CSV: " << opt.log_csv << "\n";
            return 5;
        }
        csv << "step,time,x,y,heading,speed,steering,ref_x,ref_y,acceleration,steering_rate,friction,tube_margin,barrier_active,authority,trigger,solver_solved,solve_ms\n";
    }

    TinyWorkspace *work = solver->work;
    double err_sum = 0.0;
    double err_max = 0.0;
    int solved_count = 0;
    double min_margin = 1e9;
    int safety_fail_steps = 0;
    int collision_steps = 0;
    int barrier_active_steps = 0;

    double solve_ms_sum = 0.0;
    double solve_ms_max = 0.0;
    std::vector<double> solve_ms_values;
    solve_ms_values.reserve(static_cast<size_t>(opt.steps));
    double authority_w_sum = 0.0;
    double authority_switch_sum = 0.0;
    double authority_w_previous = 0.0;

    int reached_goal_step = -1;
    double tube_err_sum = 0.0;
    double tube_err_max = 0.0;
    int tube_violation_steps = 0;
    int consecutive_tube_breach = 0;
    int first_tube_failure_step = -1;
    int spinout_steps = 0;
    int severe_slip_steps = 0;
    int track_departure_steps = 0;
    int first_spinout_step = -1;
    int first_severe_slip_step = -1;
    int first_track_departure_step = -1;
    double min_tube_margin = 1e9;
    double min_slip_margin = 1e9;

    double mu_est_ewma = 1.0;

    double prev_predicted_yaw_dot = 0.0;
    double prev_actual_yaw_dot = 0.0;
    bool prev_state_valid = false;
    double prev_pred_x = 0.0;
    double prev_pred_y = 0.0;
    double prev_pred_theta = 0.0;
    double prev_pred_v = 0.0;
    bool prev_tube_margin_valid = false;
    double prev_tube_margin_for_trigger = opt.fail_track_radius;
    double prev_a_cmd_for_trigger = 0.0;
    double prev_delta_dot_cmd_for_trigger = 0.0;
    double no_oracle_score = 0.0;
    double no_oracle_model_risk = -1.0;
    double no_oracle_score_sum = 0.0;
    double no_oracle_score_max = 0.0;
    bool no_oracle_active = false;
    int no_oracle_hold_remaining = 0;
    int no_oracle_active_steps = 0;
    int no_oracle_first_trigger_step = -1;

    int no_oracle_last_recur_event_step = -1000000;
    bool no_oracle_recur_event_latched = false;
    int no_oracle_last_severe_slip_step = -1000000;
    std::vector<double> no_oracle_mu_est_history;
    std::vector<double> no_oracle_yaw_residual_history;
    std::vector<double> no_oracle_score_history;
    std::vector<double> no_oracle_tube_margin_history;
    std::vector<double> no_oracle_tube_shrink_history;
    std::vector<int> no_oracle_severe_slip_history;
    no_oracle_mu_est_history.reserve(static_cast<size_t>(opt.steps));
    no_oracle_yaw_residual_history.reserve(static_cast<size_t>(opt.steps));
    no_oracle_score_history.reserve(static_cast<size_t>(opt.steps));
    no_oracle_tube_margin_history.reserve(static_cast<size_t>(opt.steps));
    no_oracle_tube_shrink_history.reserve(static_cast<size_t>(opt.steps));
    no_oracle_severe_slip_history.reserve(static_cast<size_t>(opt.steps));
    NoOracleTriggerComponents no_oracle_components;
    double adaptive_margin_scale = 1.0;
    int adaptive_active_steps = 0;

    raya::sampling::Normal process_noise(0.0, std::max(0.0, opt.process_noise_std));

    for (int k = 0; k < opt.steps; ++k) {
        const double t = k * opt.dt;

        adaptive_margin_scale =
            clamp_value(1.0 + 1.6 * std::max(0.0, 0.7 - mu_est_ewma), 1.0, 2.5);

        no_oracle_components = compute_no_oracle_trigger(
            x(0), x(1), x(2), x(3), x(4), t, opt, prev_state_valid,
            prev_pred_x, prev_pred_y, prev_pred_theta, prev_pred_v,
            prev_predicted_yaw_dot, prev_actual_yaw_dot,
            prev_tube_margin_valid, prev_tube_margin_for_trigger, mu_est_ewma,
            prev_a_cmd_for_trigger, prev_delta_dot_cmd_for_trigger);
        const double trigger_alpha = clamp_value(opt.no_oracle_score_alpha, 0.0, 1.0);
        no_oracle_score =
            (k == 0)
                ? no_oracle_components.score_raw
                : (1.0 - trigger_alpha) * no_oracle_score +
                      trigger_alpha * no_oracle_components.score_raw;
        no_oracle_score_sum += no_oracle_score;
        no_oracle_score_max = std::max(no_oracle_score_max, no_oracle_score);

        const bool severe_slip_history_event =
            (no_oracle_components.mu_est <= 0.42 ||
             no_oracle_components.slip_proxy >= opt.fail_severe_slip_proxy) &&
            no_oracle_components.yaw_rate_residual >= 0.55 &&
            no_oracle_components.score_raw >= 2.8;
        if (severe_slip_history_event) no_oracle_last_severe_slip_step = k;
        no_oracle_mu_est_history.push_back(no_oracle_components.mu_est);
        no_oracle_yaw_residual_history.push_back(no_oracle_components.yaw_rate_residual);
        no_oracle_score_history.push_back(no_oracle_score);
        no_oracle_tube_margin_history.push_back(no_oracle_components.tube_margin);
        no_oracle_tube_shrink_history.push_back(no_oracle_components.tube_margin_shrink_rate);
        no_oracle_severe_slip_history.push_back(severe_slip_history_event ? 1 : 0);
        auto history_min = [&](const std::vector<double> &values, int window) {
            const int end = static_cast<int>(values.size());
            const int begin = std::max(0, end - window);
            double best = std::numeric_limits<double>::infinity();
            for (int j = begin; j < end; ++j) best = std::min(best, values[static_cast<size_t>(j)]);
            return std::isfinite(best) ? best : 0.0;
        };
        auto history_max = [&](const std::vector<double> &values, int window) {
            const int end = static_cast<int>(values.size());
            const int begin = std::max(0, end - window);
            double best = -std::numeric_limits<double>::infinity();
            for (int j = begin; j < end; ++j) best = std::max(best, values[static_cast<size_t>(j)]);
            return std::isfinite(best) ? best : 0.0;
        };
        auto history_count = [&](const std::vector<int> &values, int window) {
            const int end = static_cast<int>(values.size());
            const int begin = std::max(0, end - window);
            int count = 0;
            for (int j = begin; j < end; ++j) count += values[static_cast<size_t>(j)];
            return count;
        };
        no_oracle_components.mu_est_min_10 = history_min(no_oracle_mu_est_history, 10);
        no_oracle_components.mu_est_min_30 = history_min(no_oracle_mu_est_history, 30);
        no_oracle_components.yaw_rate_residual_max_10 =
            history_max(no_oracle_yaw_residual_history, 10);
        no_oracle_components.yaw_rate_residual_max_30 =
            history_max(no_oracle_yaw_residual_history, 30);
        no_oracle_components.trigger_score_max_10 = history_max(no_oracle_score_history, 10);
        no_oracle_components.tube_margin_min_10 =
            history_min(no_oracle_tube_margin_history, 10);
        no_oracle_components.tube_margin_min_30 =
            history_min(no_oracle_tube_margin_history, 30);
        no_oracle_components.tube_margin_shrink_rate_max_10 =
            history_max(no_oracle_tube_shrink_history, 10);
        no_oracle_components.time_since_severe_slip =
            no_oracle_last_severe_slip_step > -100000
                ? std::min(10.0, (k - no_oracle_last_severe_slip_step) * opt.dt)
                : 10.0;
        no_oracle_components.severe_slip_count_30 =
            static_cast<double>(history_count(no_oracle_severe_slip_history, 30));
        no_oracle_components.recent_severe_slip =
            history_count(no_oracle_severe_slip_history, 10) > 0 ? 1.0 : 0.0;

        no_oracle_model_risk = eval_no_oracle_risk_model(no_oracle_components, no_oracle_score);
        prev_tube_margin_for_trigger = no_oracle_components.tube_margin;
        prev_tube_margin_valid = true;

        {

        }

        {
            const bool use_risk_model =
                false;
            const bool slip_estimate_low =
                no_oracle_components.mu_est <= opt.no_oracle_mu_gate;
            const bool recoverability_urgent =
                (no_oracle_components.tube_margin < 0.30 * opt.fail_track_radius &&
                 no_oracle_components.tube_margin_shrink_rate > 0.20);
            const bool severe_observed_slip =
                false;
            const bool recurrence_event_now =
                false;
            if (recurrence_event_now && !no_oracle_recur_event_latched) {
                no_oracle_last_recur_event_step = k;
                no_oracle_recur_event_latched = true;
            } else if (!recurrence_event_now) {
                no_oracle_recur_event_latched = false;
            }

            const bool recurrence_warning =
                false;
            const bool should_enter = use_risk_model
                ? ((no_oracle_model_risk >= opt.no_oracle_risk_enter) ||
                   severe_observed_slip ||
                   (no_oracle_model_risk >= opt.no_oracle_risk_warn &&
                    no_oracle_components.mu_est <= opt.no_oracle_warning_mu_gate &&
                    no_oracle_components.tube_margin <
                        opt.no_oracle_warning_margin_frac * opt.fail_track_radius &&
                    no_oracle_score >= opt.no_oracle_warning_score) ||
                   recurrence_warning)
                : ((slip_estimate_low || recoverability_urgent) &&
                   ((no_oracle_score >= opt.no_oracle_enter_score) ||
                    (no_oracle_components.score_raw >= 1.35 * opt.no_oracle_enter_score))) ||
                  severe_observed_slip;
            const bool should_exit = use_risk_model
                ? (no_oracle_model_risk < opt.no_oracle_risk_exit)
                : (no_oracle_score < opt.no_oracle_exit_score);
            const int normal_hold_steps = std::max(0, opt.no_oracle_hold_steps);
            const int severe_hold_steps =
                opt.no_oracle_severe_slip_hold_steps >= 0
                    ? std::max(normal_hold_steps, opt.no_oracle_severe_slip_hold_steps)
                    : normal_hold_steps;
            const int requested_hold_steps =
                severe_observed_slip ? severe_hold_steps : normal_hold_steps;
            if (no_oracle_active) {
                if (should_enter) {
                    no_oracle_hold_remaining =
                        std::max(no_oracle_hold_remaining, requested_hold_steps);
                } else if (no_oracle_hold_remaining > 0) {
                    no_oracle_hold_remaining -= 1;
                } else if (should_exit) {
                    no_oracle_active = false;
                }
            } else if (should_enter) {
                no_oracle_active = true;
                no_oracle_hold_remaining = requested_hold_steps;
                if (no_oracle_first_trigger_step < 0) no_oracle_first_trigger_step = k;
            }
            if (no_oracle_active) {
                no_oracle_active_steps += 1;
                adaptive_active_steps += 1;
            }
        }

        const double no_oracle_mu_feature =
            (1.0);

        double linearize_mu = 1.0;
        {
            linearize_mu = 1.0;
        }

        double w_authority = 0.0;
        {
            double xr_a, yr_a, thr_a, vr_a;
            traj_ref(t, opt, &xr_a, &yr_a, &thr_a, &vr_a);
            const double tube_err_now = std::hypot(x(0) - xr_a, x(1) - yr_a);

            double authority_analytic_h = 1.0e9;

            {
                const double raw_w = eval_authority_policy_car(
                    x(0), x(1), x(2), x(3), x(4), t, opt,
                    no_oracle_mu_feature, no_oracle_components,
                    no_oracle_score, authority_analytic_h,
                    authority_w_previous);
                w_authority = (k == 0)
                    ? raw_w
                    : opt.authority_policy_smoothing * authority_w_previous +
                          (1.0 - opt.authority_policy_smoothing) * raw_w;
            }
        }
        const double authority_w_effective =
            ((opt.authority_beta_track > 0.0))
                ? w_authority
                : 0.0;
        authority_w_sum += authority_w_effective;
        authority_switch_sum += std::abs(authority_w_effective - authority_w_previous);
        authority_w_previous = authority_w_effective;
        if (opt.authority_beta_track > 0.0) {
            const double rho = solver->cache->rho;
            const double s = std::max(0.1, 1.0 - opt.authority_beta_track * w_authority);
            solver->work->Q(0) = opt.q_x * s + rho;
            solver->work->Q(1) = opt.q_y * s + rho;
            solver->work->Q(2) = opt.q_theta + rho;
            solver->work->Q(3) = opt.q_v * s + rho;
            solver->work->Q(4) = opt.q_delta + rho;
        }
        relinearize_bicycle(solver, x(0), x(1), x(2), x(3), x(4), opt, linearize_mu);
        tiny_set_x0(solver, x);

        Trajectory traj = parse_trajectory(opt.trajectory);
        Scenario sc = parse_scenario(opt.scenario);
        if (traj == Trajectory::kLineX && sc != Scenario::kTrack) {
            const double dx_goal = opt.goal_x - x(0);
            const double dy_goal = opt.goal_y - x(1);
            const double theta_ref = std::atan2(dy_goal, dx_goal);
            for (int i = 0; i < kHorizon; ++i) {
                const double s_ahead = i * opt.dt * opt.v_ref;
                work->Xref(0, i) = static_cast<tinytype>(x(0) + s_ahead * std::cos(theta_ref));
                work->Xref(1, i) = static_cast<tinytype>(x(1) + s_ahead * std::sin(theta_ref));
                work->Xref(2, i) = static_cast<tinytype>(theta_ref);
                work->Xref(3, i) = static_cast<tinytype>(opt.v_ref);
                work->Xref(4, i) = 0.0;
            }
        } else {

            double thr_prev = static_cast<double>(x(2));
            for (int i = 0; i < kHorizon; ++i) {
                double xr, yr, thr, vr;
                traj_ref(t + i * opt.dt, opt, &xr, &yr, &thr, &vr);
                while (thr - thr_prev > M_PI) thr -= 2.0 * M_PI;
                while (thr - thr_prev < -M_PI) thr += 2.0 * M_PI;
                thr_prev = thr;
                work->Xref(0, i) = static_cast<tinytype>(xr);
                work->Xref(1, i) = static_cast<tinytype>(yr);
                work->Xref(2, i) = static_cast<tinytype>(thr);
                work->Xref(3, i) = static_cast<tinytype>(vr);

                double xrn, yrn, thrn, vrn;
                traj_ref(t + (i + 1) * opt.dt, opt, &xrn, &yrn, &thrn, &vrn);
                const double dtheta = wrap_angle(thrn - thr);
                const double curvature = (vr > 1e-3) ? (dtheta / opt.dt / vr) : 0.0;
                const double delta_ref = std::atan(opt.wheelbase * curvature);
                work->Xref(4, i) = static_cast<tinytype>(
                    clamp_value(delta_ref, -opt.delta_max, opt.delta_max));
            }
        }
        work->Uref.setZero();

        bool barrier_active = false;
        int barrier_constraint_count = 0;
        double tv_state_slack_pre = std::numeric_limits<double>::quiet_NaN();
        double tv_state_slack_post = std::numeric_limits<double>::quiet_NaN();
        double tv_input_slack_pre = std::numeric_limits<double>::quiet_NaN();
        double tv_input_slack_post = std::numeric_limits<double>::quiet_NaN();

        {
            constexpr int kStateConstraintRows = 6;
            constexpr int kInputConstraintRows = 6;
            tinyMatrix tv_Alin_x = tinyMatrix::Zero(kStateConstraintRows * kHorizon, kStateDim);
            tinyMatrix tv_blin_x = tinyMatrix::Constant(kStateConstraintRows, kHorizon, 1.0e9);
            tinyMatrix tv_Alin_u = tinyMatrix::Zero(kInputConstraintRows * (kHorizon - 1), kInputDim);
            tinyMatrix tv_blin_u = tinyMatrix::Constant(kInputConstraintRows, kHorizon - 1, 1.0e9);

            const int h_active = std::min(kHorizon, std::max(1, opt.robust_active_horizon));
            bool any_state_constraint = false;
            bool any_input_constraint = false;
            const bool analytic_state_method =
                false;

            for (int i = 0; i < h_active; ++i) {
                int row = 0;
                int input_row = 0;

                double xb = x(0), yb = x(1), thb = x(2), vb = x(3), db = x(4);
                if (i > 0) {
                    xb = static_cast<double>(work->x(0, i));
                    yb = static_cast<double>(work->x(1, i));
                    thb = static_cast<double>(work->x(2, i));
                    vb = static_cast<double>(work->x(3, i));
                    db = static_cast<double>(work->x(4, i));
                    if (!std::isfinite(xb) || !std::isfinite(yb)) {
                        xb = static_cast<double>(work->Xref(0, i));
                        yb = static_cast<double>(work->Xref(1, i));
                        thb = static_cast<double>(work->Xref(2, i));
                        vb = static_cast<double>(work->Xref(3, i));
                        db = static_cast<double>(work->Xref(4, i));
                    }
                }

                const SafetySnapshot safety_stage =
                    compute_safety_snapshot(xb, yb, thb, vb, db, opt);
                const double h_structural =
                    safety_stage.d_obs -
                    opt.robust_structural_vdot_gain * std::max(0.0, -safety_stage.d_obs_dot);

                double xr_g, yr_g, thr_g, vr_g;
                traj_ref(t + i * opt.dt, opt, &xr_g, &yr_g, &thr_g, &vr_g);
                const double tube_err_stage =
                    std::hypot(xb - xr_g, yb - yr_g);
                const double tube_margin_stage =
                    opt.fail_track_radius - tube_err_stage;

                const double tube_gate_margin =
                    (tube_err_stage > 0.5 * opt.fail_track_radius)
                        ? tube_margin_stage
                        : 1.0e9;
                double h_analytic =
                    std::min({h_structural, safety_stage.d_track, tube_gate_margin});

                const bool use_no_oracle_separate_barrier =
                    true;
                const double h_learned_gate =
                    (((((((((eval_no_oracle_learned_h_state_with_mu_feature(
                                            xb, yb, thb, vb, db, t + i * opt.dt, opt,
                                            no_oracle_mu_feature, &no_oracle_components))))))))));
                bool adaptive_learned_enabled = true;
                {

                    adaptive_learned_enabled =
                        (no_oracle_active);
                }

                const double h_gate =
                    ((adaptive_learned_enabled))
                        ? std::min(h_analytic, h_learned_gate)
                        : h_analytic;
                const double margin_target =
                    opt.robust_margin_base +
                    opt.robust_margin_gain *
                        std::max(0.0, opt.robust_activate_h - h_gate);

                const double activation_threshold =
                    std::max(opt.robust_activate_h,
                             margin_target + opt.robust_solver_activation_slack);
                const bool near_track =
                    safety_stage.d_track < opt.robust_track_band;
                const bool safe_skip =
                    (opt.robust_skip_safe_region != 0) &&
                    !near_track &&
                    (safety_stage.d_obs > opt.robust_skip_dobs) &&
                    (safety_stage.d_obs_dot > opt.robust_skip_dobs_dot) &&
                    (h_gate >= activation_threshold);
                const bool should_activate =
                    !safe_skip && (h_gate < activation_threshold);
                if (!should_activate) continue;

                auto add_state_constraint = [&](double a0, double a1, double a2, double a3,
                                                double a4, double b) {
                    if (row >= kStateConstraintRows) return;
                    const int cr = (kStateConstraintRows * i) + row;
                    tv_Alin_x(cr, 0) = a0;
                    tv_Alin_x(cr, 1) = a1;
                    tv_Alin_x(cr, 2) = a2;
                    tv_Alin_x(cr, 3) = a3;
                    tv_Alin_x(cr, 4) = a4;
                    tv_blin_x(row, i) = b;
                    row += 1;
                    any_state_constraint = true;
                    barrier_constraint_count += 1;
                };

                auto add_tracking_tube_constraint = [&]() {

                    double xr_i, yr_i, thr_i, vr_i;
                    traj_ref(t + i * opt.dt, opt, &xr_i, &yr_i, &thr_i, &vr_i);
                    const double dx_t = xb - xr_i;
                    const double dy_t = yb - yr_i;
                    const double tube_err_i = std::hypot(dx_t, dy_t);
                    if (tube_err_i > 0.5 * opt.fail_track_radius) {
                        const double tube_R = opt.fail_track_radius;
                        const double h_tube = tube_R * tube_R - dx_t * dx_t - dy_t * dy_t;
                        const double grad_tx = -2.0 * dx_t;
                        const double grad_ty = -2.0 * dy_t;
                        const double tube_margin_target = std::max(0.04, margin_target * tube_R);
                        const double b_const = h_tube - tube_margin_target -
                                               (grad_tx * xb + grad_ty * yb);
                        add_state_constraint(-grad_tx, -grad_ty, 0.0, 0.0, 0.0, b_const);
                    }
                };

                {

                    if (adaptive_learned_enabled) {
                        const double h_learned = h_learned_gate;
                        double grad[5] = {0, 0, 0, 0, 0};
                        {
                            finite_diff_learned_h_grad(
                                xb, yb, thb, vb, db, t + i * opt.dt, opt, grad,
                                (no_oracle_mu_feature),
                                (&no_oracle_components));
                        }
                        const double gnorm = std::sqrt(
                            grad[0] * grad[0] + grad[1] * grad[1] +
                            grad[2] * grad[2] + grad[3] * grad[3] + grad[4] * grad[4]);
                        if (gnorm > 1e-8) {
                            const double grad_dot_xbar =
                                grad[0] * xb + grad[1] * yb + grad[2] * thb +
                                grad[3] * vb + grad[4] * db;
                            const double b_const = h_learned - margin_target - grad_dot_xbar;
                            add_state_constraint(-grad[0], -grad[1], -grad[2],
                                                 -grad[3], -grad[4], b_const);
                        }
                    }

                    add_state_constraint(0.0, 1.0, 0.0, 0.0, 0.0,
                                         opt.track_half_width - margin_target);
                    add_state_constraint(0.0, -1.0, 0.0, 0.0, 0.0,
                                         opt.track_half_width - margin_target);
                    add_tracking_tube_constraint();
                }
            }

            if (!any_state_constraint && !any_input_constraint) {
                solver->settings->en_tv_state_linear = 0;
                solver->settings->en_tv_input_linear = 0;
                solver->settings->abs_pri_tol = inactive_abs_pri_tol;
                solver->settings->abs_dua_tol = inactive_abs_dua_tol;
                work->gl_tv.setZero();
                work->yl_tv.setZero();
            } else {
                tiny_set_tv_linear_constraints(solver, tv_Alin_x, tv_blin_x, tv_Alin_u, tv_blin_u);
                solver->settings->en_tv_state_linear = any_state_constraint ? 1 : 0;
                solver->settings->en_tv_input_linear = any_input_constraint ? 1 : 0;
                solver->settings->abs_pri_tol = opt.robust_abs_pri_tol;
                solver->settings->abs_dua_tol = opt.robust_abs_dua_tol;
                const double dual_decay = clamp_value(opt.robust_tv_dual_decay, 0.0, 1.0);
                work->gl_tv *= dual_decay;
                work->yl_tv *= dual_decay;
                barrier_active = true;
                tv_state_slack_pre = tv_state_min_slack(solver);
                tv_input_slack_pre = tv_input_min_slack(solver);
            }
        }
        if (barrier_active) barrier_active_steps += 1;

        const auto solve_start = std::chrono::steady_clock::now();
        tiny_solve(solver);
        if ((no_oracle_active) &&
            solver->solution != nullptr &&
            !solver->solution->solved) {
            solver->settings->en_tv_state_linear = 0;
            solver->settings->en_tv_input_linear = 0;
            solver->settings->abs_pri_tol = inactive_abs_pri_tol;
            solver->settings->abs_dua_tol = inactive_abs_dua_tol;
            work->u.setZero();
            work->z.setZero();
            work->znew.setZero();
            work->y.setZero();
            work->v = work->Xref;
            work->vnew = work->Xref;
            work->g.setZero();
            work->zl_tv.setZero();
            work->zlnew_tv.setZero();
            work->gl_tv.setZero();
            work->yl_tv.setZero();
            tiny_solve(solver);
        }
        const auto solve_end = std::chrono::steady_clock::now();
        const double solve_ms =
            std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
        solve_ms_sum += solve_ms;
        solve_ms_max = std::max(solve_ms_max, solve_ms);
        solve_ms_values.push_back(solve_ms);
        if (solver->solution->solved) solved_count += 1;
        tv_state_slack_post = tv_state_min_slack(solver);
        tv_input_slack_post = tv_input_min_slack(solver);

        double a_cmd = 0.0;
        double delta_dot_cmd = 0.0;
        if (solver->solution && solver->solution->u.size() >= kInputDim) {
            a_cmd = static_cast<double>(solver->solution->u(0, 0));
            delta_dot_cmd = static_cast<double>(solver->solution->u(1, 0));
        }

        if (!std::isfinite(a_cmd) || !std::isfinite(delta_dot_cmd)) {
            a_cmd = 0.0;
            delta_dot_cmd = 0.0;
        }

        a_cmd = clamp_value(a_cmd, opt.a_min, opt.a_max);
        delta_dot_cmd = clamp_value(delta_dot_cmd, -opt.delta_dot_max, opt.delta_dot_max);
        prev_a_cmd_for_trigger = a_cmd;
        prev_delta_dot_cmd_for_trigger = delta_dot_cmd;

        const double x_prev = x(0);
        const double y_prev = x(1);
        const double theta_prev = x(2);
        const double mass_inv = 1.0 / std::max(1e-3, opt.plant_mass_scale);
        const double L_actual = opt.wheelbase * opt.plant_wheelbase_scale;
        const double mu_eff = friction_mu_at(x(0), x(1), t, opt);
        const double a_eff = (a_cmd * mass_inv) - opt.plant_drag_coeff * x(3);
        const double v_next = clamp_value(x(3) + opt.dt * a_eff, opt.v_min, opt.v_max);
        const double delta_next = clamp_value(x(4) + opt.dt * delta_dot_cmd * mass_inv,
                                              -opt.delta_max, opt.delta_max);

        const double predicted_yaw_dot_step = (v_next / L_actual) * std::tan(delta_next);
        const double theta_nominal_next = wrap_angle(theta_prev + opt.dt * predicted_yaw_dot_step);
        prev_pred_x = x_prev + opt.dt * v_next * std::cos(theta_nominal_next);
        prev_pred_y = y_prev + opt.dt * v_next * std::sin(theta_nominal_next);
        prev_pred_theta = theta_nominal_next;
        prev_pred_v = v_next;

        const double theta_dot = mu_eff * predicted_yaw_dot_step;
        prev_predicted_yaw_dot = predicted_yaw_dot_step;
        prev_actual_yaw_dot = theta_dot;
        prev_state_valid = true;
        const double theta_next = wrap_angle(theta_prev + opt.dt * theta_dot);

        const double slip_mag = opt.slip_lateral_gain * (1.0 - mu_eff) * v_next *
                                std::abs(std::tan(delta_next));
        const double slip_sign = (delta_next >= 0.0) ? -1.0 : +1.0;
        const double slip_x = slip_mag * std::cos(theta_next + 0.5 * kPi) * slip_sign;
        const double slip_y = slip_mag * std::sin(theta_next + 0.5 * kPi) * slip_sign;
        const double x_next = x_prev + opt.dt * (v_next * std::cos(theta_next) + slip_x);
        const double y_next = y_prev + opt.dt * (v_next * std::sin(theta_next) + slip_y);
        x(0) = x_next;
        x(1) = y_next;
        x(2) = theta_next;
        x(3) = v_next;
        x(4) = delta_next;
        if (opt.process_noise_std > 0.0) {
            x(0) += process_noise(rng);
            x(1) += process_noise(rng);
        }

        const SafetySnapshot safety =
            compute_safety_snapshot(x(0), x(1), x(2), x(3), x(4), opt);
        min_margin = std::min(min_margin, safety.margin_combined);
        bool collided = safety.d_obs < 0.0;
        if (safety.margin_combined < 0.0) safety_fail_steps += 1;
        if (collided) collision_steps += 1;

        double xr_now, yr_now, thr_now, vr_now;
        traj_ref(t + opt.dt, opt, &xr_now, &yr_now, &thr_now, &vr_now);
        const double tube_err_now = std::hypot(x(0) - xr_now, x(1) - yr_now);
        const double tube_margin_now_for_metrics = opt.fail_track_radius - tube_err_now;
        min_tube_margin = std::min(min_tube_margin, tube_margin_now_for_metrics);
        const double slip_proxy_now = std::abs(predicted_yaw_dot_step - theta_dot);
        const double slip_margin_now = opt.fail_severe_slip_proxy - slip_proxy_now;
        min_slip_margin = std::min(min_slip_margin, slip_margin_now);
        tube_err_sum += tube_err_now;
        tube_err_max = std::max(tube_err_max, tube_err_now);
        if (tube_err_now > opt.fail_track_radius) {
            tube_violation_steps += 1;
            consecutive_tube_breach += 1;
            if (consecutive_tube_breach >= opt.fail_consecutive_steps && first_tube_failure_step < 0) {
                first_tube_failure_step = k;
            }
        } else {
            consecutive_tube_breach = 0;
        }
        if (std::abs(theta_dot) > opt.fail_attitude_rate) {
            spinout_steps += 1;
            if (first_spinout_step < 0) first_spinout_step = k;
        }
        if (slip_proxy_now > opt.fail_severe_slip_proxy) {
            severe_slip_steps += 1;
            if (first_severe_slip_step < 0) first_severe_slip_step = k;
        }
        if (safety.d_track < 0.0) {
            track_departure_steps += 1;
            if (first_track_departure_step < 0) first_track_departure_step = k;
        }

        const double goal_dist_now = std::hypot(opt.goal_x - x(0), opt.goal_y - x(1));
        if (reached_goal_step < 0 && goal_dist_now < 0.30) {
            reached_goal_step = k;
        }

        const double err = goal_dist_now;
        err_sum += err;
        err_max = std::max(err_max, err);

        if (csv.is_open()) {
            const double tube_margin_now = opt.fail_track_radius - tube_err_now;
            double feature_now[kMaxLearnedFeatureDim];
            learned_features(x(0), x(1), x(2), x(3), x(4), t + opt.dt, opt, feature_now);
            csv << std::setprecision(12)
                << k << "," << (t + opt.dt) << ","
                << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "," << x(4) << ","
                << xr_now << "," << yr_now << "," << a_cmd << "," << delta_dot_cmd << ","
                << friction_mu_at(x(0), x(1), t + opt.dt, opt) << "," << tube_margin_now << ","
                << (barrier_active ? 1 : 0) << "," << authority_w_effective << ","
                << no_oracle_score << "," << solver->solution->solved << "," << solve_ms << "\n";
        }

        {
            if (first_tube_failure_step >= 0 ||
                first_spinout_step >= 0 ||
                first_severe_slip_step >= 0 ||
                first_track_departure_step >= 0) {
                break;
            }
        }
    }

    int first_track_failure_step = first_tube_failure_step;
    for (int fs : {first_spinout_step, first_severe_slip_step, first_track_departure_step}) {
        if (fs >= 0 && (first_track_failure_step < 0 || fs < first_track_failure_step)) {
            first_track_failure_step = fs;
        }
    }
    const int actual_steps = (reached_goal_step >= 0)
        ? (reached_goal_step + 1)
        : (first_track_failure_step >= 0 ? first_track_failure_step + 1 : opt.steps);

    const double mean_tube_err = tube_err_sum / static_cast<double>(actual_steps);
    const int reached_goal = (reached_goal_step >= 0) ? 1 : 0;
    int success;
    {

        success = (first_track_failure_step < 0) ? 1 : 0;
    }
    const double mean_solve_ms = solve_ms_sum / static_cast<double>(std::max(1, actual_steps));
    double p95_solve_ms = solve_ms_max;
    if (!solve_ms_values.empty()) {
        std::sort(solve_ms_values.begin(), solve_ms_values.end());
        const size_t idx = std::min(
            solve_ms_values.size() - 1,
            static_cast<size_t>(std::floor(0.95 * static_cast<double>(solve_ms_values.size() - 1)))
        );
        p95_solve_ms = solve_ms_values[idx];
    }
    const double no_oracle_mean_score =
        no_oracle_score_sum / static_cast<double>(std::max(1, actual_steps));
    const double mean_authority_w =
        authority_w_sum / static_cast<double>(std::max(1, actual_steps));
    const double mean_authority_switch =
        authority_switch_sum / static_cast<double>(std::max(1, actual_steps));

    std::cout << std::setprecision(12) << "RAYA F1TENTH simulation\n";
    std::cout << "  method: " << "RAYA" << "\n";
    std::cout << "  scenario: " << opt.scenario << "\n";
    std::cout << "  seed: " << opt.seed << "\n";
    std::cout << "  steps: " << actual_steps << "\n";
    std::cout << "  reached_goal: " << reached_goal << "\n";
    std::cout << "  collisions: " << collision_steps << "\n";
    std::cout << "  tube_violation_steps: " << tube_violation_steps << "\n";
    std::cout << "  spinout_steps: " << spinout_steps << "\n";
    std::cout << "  severe_slip_steps: " << severe_slip_steps << "\n";
    std::cout << "  track_departure_steps: " << track_departure_steps << "\n";
    std::cout << "  success: " << success << "\n";
    std::cout << "  min_margin: " << min_margin << "\n";
    std::cout << "  min_tube_margin: " << min_tube_margin << "\n";
    std::cout << "  mean_tube_err: " << mean_tube_err << "\n";
    std::cout << "  min_slip_margin: " << min_slip_margin << "\n";
    std::cout << "  mean_solve_ms: " << mean_solve_ms << "\n";
    std::cout << "  p95_solve_ms: " << p95_solve_ms << "\n";
    std::cout << "  barrier_active_steps: " << barrier_active_steps << "\n";

    std::cout << "  no_oracle_active_steps: " << no_oracle_active_steps << "\n";
    std::cout << "  no_oracle_first_trigger_step: " << no_oracle_first_trigger_step << "\n";

    std::cout << "  no_oracle_mean_score: " << no_oracle_mean_score << "\n";
    std::cout << "  no_oracle_max_score: " << no_oracle_score_max << "\n";
    std::cout << "  mean_authority_w: " << mean_authority_w << "\n";
    std::cout << "  mean_authority_switch: " << mean_authority_switch << "\n";

    if (csv.is_open()) { csv.close(); if (!csv) { std::cerr << "Failed to write CSV\n"; return 5; } }
    return 0;
}
