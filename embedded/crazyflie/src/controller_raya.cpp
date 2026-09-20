/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2019 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * controller_tinympc.c - App layer application of TinyMPC.
 */



#ifndef TINYMPC_BENCH_PROFILE
#define TINYMPC_BENCH_PROFILE 0
#endif
#if TINYMPC_BENCH_PROFILE != 0 && TINYMPC_BENCH_PROFILE != 1
#error "TINYMPC_BENCH_PROFILE must be 0 (flight) or 1 (motors-off canned-state profiling)"
#endif
#ifndef TINYMPC_PROFILE_TANH
#define TINYMPC_PROFILE_TANH 0
#endif
#if TINYMPC_PROFILE_TANH && !TINYMPC_BENCH_PROFILE
#error "TINYMPC_PROFILE_TANH is permitted only in a motors-off bench build"
#endif

#ifndef TINYMPC_COMMON_Z_GUARD
#define TINYMPC_COMMON_Z_GUARD 1
#endif
#if TINYMPC_COMMON_Z_GUARD != 0 && TINYMPC_COMMON_Z_GUARD != 1
#error "TINYMPC_COMMON_Z_GUARD must be 0 (raw preview Z) or 1 (reference clamp)"
#endif

#define TINYMPC_TRAJECTORY_HOVER   0
#define TINYMPC_TRAJECTORY_LINE_X  1
#define TINYMPC_TRAJECTORY_LINE_Y  2
#define TINYMPC_TRAJECTORY_CIRCLE  3
#define TINYMPC_TRAJECTORY_FIGURE8 4

#ifndef TINYMPC_TRAJECTORY
#define TINYMPC_TRAJECTORY 4
#endif
#if TINYMPC_TRAJECTORY < TINYMPC_TRAJECTORY_HOVER ||      TINYMPC_TRAJECTORY > TINYMPC_TRAJECTORY_FIGURE8
#error "TINYMPC_TRAJECTORY must be 0 (hover), 1 (X-line), 2 (Y-line), 3 (circle), or 4 (figure eight)"
#endif

#define TINYMPC_OOT_TAKEOFF 1
#if TINYMPC_OOT_TAKEOFF != 0 && TINYMPC_OOT_TAKEOFF != 1
#error "TINYMPC_OOT_TAKEOFF must be 0 or 1"
#endif

#define TINYMPC_GLOBAL_MOTOR_CAP_PERCENT 100
#define TINYMPC_FAULT_MOTOR_INDEX 0
#define TINYMPC_FAULT_MOTOR_RETAINED_PERCENT 100
#define TINYMPC_FAULT_ONSET_MS 2000u

#if TINYMPC_GLOBAL_MOTOR_CAP_PERCENT < 1 ||      TINYMPC_GLOBAL_MOTOR_CAP_PERCENT > 100
#error "TINYMPC_GLOBAL_MOTOR_CAP_PERCENT must be in [1, 100]"
#endif
#if TINYMPC_FAULT_MOTOR_INDEX < 0 || TINYMPC_FAULT_MOTOR_INDEX >= 4
#error "TINYMPC_FAULT_MOTOR_INDEX must be in [0, 3]"
#endif
#if TINYMPC_FAULT_MOTOR_RETAINED_PERCENT < 1 ||      TINYMPC_FAULT_MOTOR_RETAINED_PERCENT > 100
#error "TINYMPC_FAULT_MOTOR_RETAINED_PERCENT must be in [1, 100]"
#endif

#define TINYMPC_TAKEOFF_TRIGGER_Z_M       0.15f
#define TINYMPC_TAKEOFF_ASCENT_MPS        0.15f
#define TINYMPC_TAKEOFF_TRAJECTORY_START_MS 2000u
#define TINYMPC_TAKEOFF_TIMEOUT_MS        12000u
#if TINYMPC_TAKEOFF_TRAJECTORY_START_MS >= TINYMPC_TAKEOFF_TIMEOUT_MS
#error "Trajectory start delay must be shorter than takeoff timeout"
#endif

#define TINYMPC_LANDING_DESCENT_MPS       0.15f
#define TINYMPC_LANDING_TARGET_Z_M       (-0.05f)
#define TINYMPC_LANDING_TOUCHDOWN_Z_M     0.08f
#define TINYMPC_LANDING_TOUCHDOWN_VZ_MPS  0.10f

#define TINYMPC_XYZ_CONSOLE_DECIMATION 2
#if TINYMPC_XYZ_CONSOLE_DECIMATION < 0
#error "TINYMPC_XYZ_CONSOLE_DECIMATION must be zero or positive"
#endif

#include "Eigen.h"

#include "tinympc/admm.hpp"
#include "tinympc/psd_support.hpp"
#include "raya_cache_seed.hpp"
#include "raya_barrier.hpp"
#include "raya_scheduler.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "stm32f4xx.h"

#include "app.h"
#include "config.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "sensors.h"
#include "static_mem.h"
#include "system.h"

#include "controller.h"
#include "power_distribution.h"
#include "physicalConstants.h"
#include "log.h"
#include "eventtrigger.h"
#include "param.h"
#include "num.h"
#include "math3d.h"

#include "cpp_compat.h" 

#include "controller_pid.h"

#include "quadrotor_50hz_line_9s_xyz.hpp"
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8

#include "traj_fig8_12.h"
#endif

#define DEBUG_MODULE "MPCTASK"
#include "debug.h"

#define MPC_RATE 20
#define LOWLEVEL_RATE RATE_500_HZ
constexpr uint32_t kMpcDeadlineCycles = 8400000u;  
constexpr uint16_t kGlobalMotorCapPwm = static_cast<uint16_t>(
    (65535u * TINYMPC_GLOBAL_MOTOR_CAP_PERCENT) / 100u);
constexpr uint16_t kFaultMotorEquivalentFullScalePwm =
    static_cast<uint16_t>(
        (65535u * TINYMPC_FAULT_MOTOR_RETAINED_PERCENT) / 100u);
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8
constexpr int kFigure8SourceRateHz = 100;
constexpr int kFigure8CyclesPerSource = 3;
constexpr int kFigure8TablePassCount = 2;
constexpr int kFigure8TotalCycleCount =
    kFigure8CyclesPerSource * kFigure8TablePassCount;
constexpr int kFigure8StateCount =
    static_cast<int>(sizeof(X_ref_data) / sizeof(X_ref_data[0]));
constexpr int kFigure8InputCount =
    static_cast<int>(sizeof(U_ref_data) / sizeof(U_ref_data[0]));
constexpr int kFigure8SourceIntervalsPerPass = kFigure8StateCount - 1;
constexpr int kFigure8TotalSourceIntervals =
    kFigure8TablePassCount * kFigure8SourceIntervalsPerPass;
constexpr int kFigure8MaxMpcIndex =
    (kFigure8TotalSourceIntervals * MPC_RATE + kFigure8SourceRateHz - 1) /
    kFigure8SourceRateHz;
static_assert(kFigure8StateCount > NHORIZON,
              "Figure-8 reference must cover the MPC horizon");
static_assert(kFigure8InputCount == kFigure8SourceIntervalsPerPass,
              "Figure-8 input reference must span every state interval");
static_assert(kFigure8TablePassCount > 0,
              "Figure-8 must execute at least one table pass");
#endif

constexpr int kPidBridgeIndex = 10;
static_assert(kPidBridgeIndex < NHORIZON,
              "kPidBridgeIndex must be inside the MPC horizon");

static SemaphoreHandle_t runTaskSemaphore;

static SemaphoreHandle_t dataMutex;
static StaticSemaphore_t dataMutexBuffer;

static void tinympcControllerTask(void *parameters);

STATIC_MEM_TASK_ALLOC(tinympcControllerTask, TINYMPC_TASK_STACKSIZE);

EVENTTRIGGER(horizon_x_part1, float, h0, float, h1, float, h2, float, h3, float, h4);
EVENTTRIGGER(horizon_x_part2, float, h5, float, h6, float, h7, float, h8, float, h9);
EVENTTRIGGER(horizon_x_part3, float, h10, float, h11, float, h12, float, h13, float, h14);
EVENTTRIGGER(horizon_x_part4, float, h15, float, h16, float, h17, float, h18, float, h19);
EVENTTRIGGER(horizon_y_part1, float, h0, float, h1, float, h2, float, h3, float, h4);
EVENTTRIGGER(horizon_y_part2, float, h5, float, h6, float, h7, float, h8, float, h9);
EVENTTRIGGER(horizon_y_part3, float, h10, float, h11, float, h12, float, h13, float, h14);
EVENTTRIGGER(horizon_y_part4, float, h15, float, h16, float, h17, float, h18, float, h19);
EVENTTRIGGER(horizon_z_part1, float, h0, float, h1, float, h2, float, h3, float, h4);
EVENTTRIGGER(horizon_z_part2, float, h5, float, h6, float, h7, float, h8, float, h9);
EVENTTRIGGER(horizon_z_part3, float, h10, float, h11, float, h12, float, h13, float, h14);
EVENTTRIGGER(horizon_z_part4, float, h15, float, h16, float, h17, float, h18, float, h19);
EVENTTRIGGER(problem_data_event, int32, solvetime_us, int32, iters, int32, cache_level);
EVENTTRIGGER(problem_residuals_event, float, prim_resid_state, float, prim_resid_input, float, dual_resid_state, float, dual_resid_input);

control_t control_data;
setpoint_t setpoint_data;
sensorData_t sensors_data;
state_t state_data;
tiny_VectorNx mpc_setpoint;
setpoint_t mpc_setpoint_pid;

setpoint_t setpoint_task;
sensorData_t sensors_task;
state_t state_task;
control_t control_task;
tiny_VectorNx mpc_setpoint_task;

static tinytype u_hover[4] = {.583, .583, .583, .583};
static struct tiny_params params;
static struct tiny_problem problem;
static tiny_MatrixNxNh problem_x;

static float init_vel_z;

static Eigen::Matrix<tinytype, 3, NTOTAL, Eigen::ColMajor> Xref_total;
static Eigen::Matrix<tinytype, NSTATES, 1, Eigen::ColMajor> Xref_origin; 
static Eigen::Matrix<tinytype, NSTATES, 1, Eigen::ColMajor> Xref_end; 
static tiny_VectorNu u_lqr;
static tiny_VectorNx current_state;

enum FlightPhase : uint8_t {
  FLIGHT_PHASE_WAITING_FOR_TAKEOFF = 0,
  FLIGHT_PHASE_TAKEOFF = 1,
  FLIGHT_PHASE_TRACKING = 2,
  FLIGHT_PHASE_LANDING = 3,
  FLIGHT_PHASE_COMPLETE = 4,
};

static bool enable_traj = TINYMPC_OOT_TAKEOFF == 0;
static bool mpc_has_run = false; 
static int traj_index = 0;
static int max_traj_index = 0;
static float traj_speed = 0.2f; 
static float traj_dist = 1.0f;  
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_HOVER ||      TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8

static float traj_height = 0.70f;
#else
static float traj_height = 1.0f;
#endif
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8

static float traj_duration =
    static_cast<float>(kFigure8TotalSourceIntervals) /
    kFigure8SourceRateHz;
#else
static float traj_duration = 12.0f;
#endif
static float traj_radius = 0.75f;
static float traj_omega = 0.45f;

static float trajectory_origin_x = 0.0f;
static float trajectory_origin_y = 0.0f;
static float trajectory_origin_z = traj_height;
static uint32_t last_controller_tick = 0;
static uint32_t controller_activate_tick = 0;

static uint32_t controller_activate_rtos_tick = 0;

static uint32_t experiment_start_rtos_tick = 0;
static uint8_t flight_phase =
    TINYMPC_OOT_TAKEOFF ? FLIGHT_PHASE_WAITING_FOR_TAKEOFF
                        : FLIGHT_PHASE_TRACKING;
static uint32_t takeoff_start_tick = 0;
static uint32_t takeoff_start_rtos_tick = 0;
static bool tracking_start_pending = false;
static bool takeoff_trigger_armed = false;
static float takeoff_start_z = 0.0f;
static float takeoff_hold_x = 0.0f;
static float takeoff_hold_y = 0.0f;
static float takeoff_hold_yaw = 0.0f;
static uint32_t landing_start_tick = 0;
static float landing_hold_x = 0.0f;
static float landing_hold_y = 0.0f;
static float landing_hold_yaw = 0.0f;
static float landing_start_z = 0.0f;
static float landing_reference_z = 0.0f;

static uint32_t mpc_start_timestamp;
static uint32_t mpc_time_us;
static struct vec phi; 
static bool isInit = false;
static int prev_cache_level = 0; 

static uint8_t controller_id = 3; // RAYA controller identifier.
static uint8_t trajectory_id = TINYMPC_TRAJECTORY;
static uint8_t previous_controller_id = 255;
static uint8_t previous_trajectory_id = 255;

static uint8_t benchmark_max_iter = 5;

static tinytype raya_margin = tinytype(0.01f);
static tinytype raya_margin_scale = tinytype(1.5f);
static tinytype raya_h_deadband = tinytype(0.30f);
static tinytype raya_act_slack = tinytype(0.20f);
static tinytype raya_demand_cap = tinytype(0.05f);
static tinytype raya_az_coeff = tinytype(8.0f);
static tinytype raya_gravity_comp = tinytype(0.0f);
static tinytype raya_fail_roll_deg = tinytype(50.0f);
static tinytype raya_fail_pitch_deg = tinytype(50.0f);
static tinytype raya_structural_guard_relax = tinytype(0.70f);
static tinytype raya_skip_z = tinytype(0.75f);
static tinytype raya_skip_vz = tinytype(-0.20f);

#ifndef RAYA_ACTIVE_HORIZON
#define RAYA_ACTIVE_HORIZON 1
#endif
#if RAYA_ACTIVE_HORIZON < 1 || RAYA_ACTIVE_HORIZON > NHORIZON
#error "RAYA_ACTIVE_HORIZON must be between 1 and NHORIZON"
#endif
static uint8_t raya_active_horizon = RAYA_ACTIVE_HORIZON;
static tinytype raya_dist_budget = tinytype(1.6f);
static tinytype raya_dist_margin_coeff = tinytype(0.018f);
static tinytype raya_dist_high_cut = tinytype(1.35f);
static tinytype raya_dist_high_bias = tinytype(0.030f);
static tinytype raya_dist_floor_band = tinytype(0.24f);
static tinytype raya_dist_floor_gain = tinytype(0.10f);
static tinytype raya_dist_desc_band = tinytype(0.20f);
static tinytype raya_dist_desc_gain = tinytype(0.02f);
static tinytype raya_horizon_margin_rate = tinytype(0.0008f);
static float raya_h = 0.0f;
static float raya_raw = 0.0f;
static float raya_grad_norm = 0.0f;
static float raya_margin_eff = 0.0f;
static float raya_threshold = 0.0f;
static uint32_t raya_eval_us = 0;

static uint32_t raya_barrier_cycles = 0;
static uint32_t raya_activation_cycles = 0;
static uint32_t raya_rl_cycles = 0;
static uint32_t raya_cache_cycles = 0;
static uint32_t mpc_solve_cycles = 0;
static uint32_t controller_total_cycles = 0;
static uint32_t max_step_cycles = 0;
static uint32_t deadline_overrun_count = 0;
static uint8_t raya_active = 0;
static uint8_t raya_active_count = 0;
static raya_embedded::Runtime raya_embedded_runtime = {};
static float raya_authority_w = 0.0f;
static float raya_authority_w_requested = 0.0f;
static float raya_qz = 1.0f;
static float raya_qz_requested = 1.0f;
static uint8_t raya_w_index = 0;
static uint8_t raya_qz_index = 0;
static uint8_t raya_cache_ok = 0;
static float raya_no_oracle_score = 0.0f;
static float raya_no_oracle_score_raw = 0.0f;
static uint8_t raya_no_oracle_active = 0;

static uint32_t benchmark_step = 0;
static uint8_t bench_state_index = 0;

#if TINYMPC_BENCH_PROFILE

static constexpr tinytype kBenchStates[][NSTATES] = {
    {0.00f,  0.00f, 1.00f,  0.00f,  0.00f, 0.00f,
     0.00f,  0.00f, 0.00f,  0.00f,  0.00f, 0.00f},
    {0.08f, -0.05f, 0.92f,  0.00f,  0.00f, 0.00f,
     0.30f, -0.20f, -0.35f, 0.00f,  0.00f, 0.00f},
    {0.12f, -0.08f, 0.82f,  0.10f, -0.08f, 0.00f,
     0.40f, -0.30f, -0.70f, 0.80f, -0.60f, 0.20f},
    {-0.10f, 0.06f, 0.75f,  0.22f, -0.18f, 0.03f,
     -0.20f, 0.25f, -1.00f, 2.00f, -1.50f, 0.60f},
    {0.00f,  0.00f, 0.90f, -0.08f,  0.06f, 0.00f,
     -0.10f, 0.10f, 0.30f, -0.50f, 0.40f, -0.20f},
};
static constexpr uint8_t kBenchStateCount =
    sizeof(kBenchStates) / sizeof(kBenchStates[0]);
static constexpr uint32_t kBenchStateHoldSteps = MPC_RATE;
#endif

static float tracking_pos_x = 0.0f;
static float tracking_pos_y = 0.0f;
static float tracking_pos_z = 0.0f;
static float tracking_vel_x = 0.0f;
static float tracking_vel_y = 0.0f;
static float tracking_vel_z = 0.0f;
static float tracking_roll_deg = 0.0f;
static float tracking_pitch_deg = 0.0f;
static float tracking_yaw_deg = 0.0f;
static float tracking_cmd_x = 0.0f;
static float tracking_cmd_y = 0.0f;
static float tracking_cmd_z = 0.0f;
static float mpc_yaw_setpoint_deg = 0.0f;
static float tracking_preview_z = 0.0f;
static uint32_t controller_total_us = 0;
static powerDistributionCapStats_t motor_cap_stats = {};
static uint8_t motor_fault_active = 0;
static uint32_t motor_fault_actual_onset_ms = 0;

static inline long console_scaled(float value, float scale)
{
  return static_cast<long>(value * scale);
}

static inline void refresh_motor_cap_stats()
{
  powerDistributionGetCapStats(&motor_cap_stats);
}

static inline void set_motor_fault_active(bool active)
{

  powerDistributionSetSingleMotorThrustScale(
      TINYMPC_FAULT_MOTOR_INDEX,
      active ? TINYMPC_FAULT_MOTOR_RETAINED_PERCENT : 100u);
  motor_fault_active = active ? 1u : 0u;
}

#if TINYMPC_BENCH_PROFILE
static void load_bench_state(tiny_MatrixNxNh *states, uint32_t step)
{
  bench_state_index = static_cast<uint8_t>(
      (step / kBenchStateHoldSteps) % kBenchStateCount);
  for (int i = 0; i < NSTATES; ++i) {
    (*states)(i, 0) = kBenchStates[bench_state_index][i];
  }
}
#endif

static Eigen::Matrix<tinytype, 3, 1> obs_center;
static Eigen::Matrix<tinytype, 3, 1> obs_start;     
static Eigen::Matrix<tinytype, 3, 1> obs_velocity;  
static Eigen::Matrix<tinytype, 3, 1> xc;
static Eigen::Matrix<tinytype, 3, 1> a_norm;
static Eigen::Matrix<tinytype, 3, 1> q_c;
static float r_obs = 0.35f;           
static float obs_activation_margin = 0.15f; 
static uint64_t obs_start_time = 0;   

static inline tinytype positive_part(tinytype value)
{
  return value > tinytype(0.0f) ? value : tinytype(0.0f);
}

static inline tinytype clamp_tiny(tinytype value,
                                  tinytype lower,
                                  tinytype upper)
{
  return value < lower ? lower : (value > upper ? upper : value);
}

static tinytype raya_effective_margin(const tiny_VectorNx &xbar, int stage)
{
  const tinytype budget = positive_part(raya_dist_budget);
  const tinytype high_budget = positive_part(budget - positive_part(raya_dist_high_cut));
  tinytype margin = positive_part(raya_margin);

  margin += positive_part(raya_dist_margin_coeff) * budget;
  margin += high_budget * positive_part(raya_dist_floor_gain) *
            positive_part(positive_part(raya_dist_floor_band) - xbar(2));
  margin += high_budget * positive_part(raya_dist_desc_gain) *
            positive_part(-xbar(8) - positive_part(raya_dist_desc_band));
  margin += positive_part(raya_horizon_margin_rate) * static_cast<tinytype>(stage);
  margin += positive_part(raya_dist_high_bias) * high_budget;
  margin *= positive_part(raya_margin_scale);

  return margin;
}

static inline tinytype structural_floor_h(const tiny_VectorNx &x)
{
  return x(2) - tinytype(0.20f) * positive_part(-x(8));
}

static inline tinytype guarded_raya_h(const tiny_VectorNx &x, tinytype learned_h)
{
  const tinytype guarded_structural =
      structural_floor_h(x) - positive_part(raya_structural_guard_relax);
  return learned_h > guarded_structural ? learned_h : guarded_structural;
}

static inline bool barrier_skipped_high_altitude(const tiny_VectorNx &x)
{
  return x(2) > raya_skip_z && x(8) > raya_skip_vz;
}

static void reset_solver_warm_start()
{
  problem.y.setZero();
  problem.g.setZero();
  problem.v.setZero();
  problem.vnew.setZero();
  problem.z.setZero();
  problem.znew.setZero();
  problem.cache_level = 0;
  prev_cache_level = 0;
}

static void reset_cost_cache()
{

  raya_cache_seed::install(&params);
}

static void reset_flight_state()
{
  traj_index = 0;
  benchmark_step = 0;
  enable_traj = TINYMPC_OOT_TAKEOFF == 0;
  mpc_has_run = false;
  flight_phase =
      TINYMPC_OOT_TAKEOFF ? FLIGHT_PHASE_WAITING_FOR_TAKEOFF
                          : FLIGHT_PHASE_TRACKING;
  takeoff_start_tick = 0;
  takeoff_start_rtos_tick = 0;
  tracking_start_pending = false;
  takeoff_trigger_armed = false;
  takeoff_start_z = 0.0f;
  takeoff_hold_x = 0.0f;
  takeoff_hold_y = 0.0f;
  takeoff_hold_yaw = 0.0f;
  trajectory_origin_z = traj_height;
  landing_start_tick = 0;
  landing_reference_z = traj_height;
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8
  max_traj_index = kFigure8MaxMpcIndex;
#else
  max_traj_index =
      static_cast<int>(positive_part(traj_duration) * MPC_RATE);
#endif
  raya_embedded_runtime = {};
  raya_cache_ok = 0;
  raya_no_oracle_score = 0.0f;
  raya_no_oracle_score_raw = 0.0f;
  raya_no_oracle_active = 0;

  bench_state_index = 0;
  max_step_cycles = 0;
  deadline_overrun_count = 0;
  set_motor_fault_active(false);
  motor_fault_actual_onset_ms = 0;
  powerDistributionResetCapStats();
  refresh_motor_cap_stats();
  reset_solver_warm_start();
}

static void reset_tracking_measurements()
{
  benchmark_step = 0;
  max_step_cycles = 0;
  deadline_overrun_count = 0;

}

static inline float quat_dot(quaternion_t a, quaternion_t b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static inline quaternion_t make_quat(float x, float y, float z, float w)
{
  quaternion_t q;
  q.x = x;
  q.y = y;
  q.z = z;
  q.w = w;
  return q;
}

static inline quaternion_t normalize_quat(quaternion_t q)
{
  float s = 1.0f / sqrtf(quat_dot(q, q));
  return make_quat(s * q.x, s * q.y, s * q.z, s * q.w);
}

static inline struct vec quat_2_rp(quaternion_t q)
{
  struct vec v;
  v.x = q.x / q.w;
  v.y = q.y / q.w;
  v.z = q.z / q.w;
  return v;
}

static inline float rp_yaw_degrees(const tiny_VectorNx &x)
{
  const float rx = x(3);
  const float ry = x(4);
  const float rz = x(5);
  const float inv_norm =
      1.0f / sqrtf(1.0f + rx * rx + ry * ry + rz * rz);
  const float qx = rx * inv_norm;
  const float qy = ry * inv_norm;
  const float qz = rz * inv_norm;
  const float qw = inv_norm;
  const float sin_yaw = 2.0f * (qw * qz + qx * qy);
  const float cos_yaw = 1.0f - 2.0f * (qy * qy + qz * qz);
  return atan2f(sin_yaw, cos_yaw) * 57.2957795f;
}

static inline void fill_hold_setpoint(setpoint_t *sp, const state_t *state)
{
  memset(sp, 0, sizeof(setpoint_t));
  sp->mode.yaw = modeAbs;
  sp->mode.x = modeAbs;
  sp->mode.y = modeAbs;
  sp->mode.z = modeAbs;
  sp->position.x = state->position.x;
  sp->position.y = state->position.y;
  sp->position.z = state->position.z;
  sp->attitude.yaw = state->attitude.yaw;
}

static inline bool oot_takeoff_requested(const setpoint_t *setpoint)
{
  const bool absolute_request =
      setpoint->mode.z == modeAbs &&
      setpoint->position.z >= TINYMPC_TAKEOFF_TRIGGER_Z_M;
  const bool upward_velocity_request =
      setpoint->mode.z == modeVelocity &&
      setpoint->velocity.z > 0.05f;
  return absolute_request || upward_velocity_request;
}

static inline void stop_motors(control_t *control)
{
  memset(control, 0, sizeof(control_t));
  control->controlMode = controlModeLegacy;
}

void controllerOutOfTreeInit(void);

void appMain()
{

  controllerOutOfTreeInit();
  DEBUG_PRINT("Waiting for activation ...\n");

  while (1)
  {
    vTaskDelay(M2T(2000));
  }
}

static void resetProblem(void) {
  
  problem.x = tiny_MatrixNxNh::Zero();
  problem.q = tiny_MatrixNxNh::Zero();
  problem.p = tiny_MatrixNxNh::Zero();
  problem.v = tiny_MatrixNxNh::Zero();
  problem.vnew = tiny_MatrixNxNh::Zero();
  problem.g = tiny_MatrixNxNh::Zero();

  problem.u = tiny_MatrixNuNhm1::Zero();
  problem.r = tiny_MatrixNuNhm1::Zero();
  problem.d = tiny_MatrixNuNhm1::Zero();
  problem.z = tiny_MatrixNuNhm1::Zero();
  problem.znew = tiny_MatrixNuNhm1::Zero();
  problem.y = tiny_MatrixNuNhm1::Zero();
}

void controllerOutOfTreeInit(void)
{
  if (isInit) {

    controllerPidInit();
    reset_flight_state();
    previous_controller_id = controller_id;
    previous_trajectory_id = trajectory_id;
    return;
  }

  controllerPidInit();
  powerDistributionSetMaxMotorThrust(kGlobalMotorCapPwm);
  set_motor_fault_active(false);
  motor_fault_actual_onset_ms = 0;
  powerDistributionResetCapStats();
  refresh_motor_cap_stats();

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  reset_cost_cache();
  params.u_min = tiny_VectorNu(-u_hover[0], -u_hover[1], -u_hover[2], -u_hover[3]).replicate<1, NHORIZON - 1>();
  params.u_max = tiny_VectorNu(1 - u_hover[0], 1 - u_hover[1], 1 - u_hover[2], 1 - u_hover[3]).replicate<1, NHORIZON - 1>();
  for (int i = 0; i < NHORIZON; i++)
  {
    params.x_min[i] = tiny_VectorNc::Constant(-1000); 
    params.x_max[i] = tiny_VectorNc::Constant(1000);
    params.A_constraints[i] = tiny_MatrixNcNx::Zero();
  }
  params.Xref = tiny_MatrixNxNh::Zero();
  params.Uref = tiny_MatrixNuNhm1::Zero();

  resetProblem();

  problem.primal_residual_state = 0;
  problem.primal_residual_input = 0;
  problem.dual_residual_state = 0;
  problem.dual_residual_input = 0;
  problem.abs_tol = 0.001;
  problem.status = 0;
  problem.iter = 0;
  problem.max_iter = benchmark_max_iter;
  problem.iters_check_rho_update = 10;
  problem.cache_level = 0; 

  Xref_origin << 0, 0, traj_height, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  Xref_end << traj_dist, 0, traj_height, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  params.Xref = Xref_origin.replicate<1, NHORIZON>();

  mpc_setpoint = Xref_origin;

  enable_traj = TINYMPC_OOT_TAKEOFF == 0;
  mpc_has_run = false;
  traj_index = 0;
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8
  max_traj_index = kFigure8MaxMpcIndex;
#else
  max_traj_index = static_cast<int>(traj_duration * MPC_RATE);
#endif
  flight_phase =
      TINYMPC_OOT_TAKEOFF ? FLIGHT_PHASE_WAITING_FOR_TAKEOFF
                          : FLIGHT_PHASE_TRACKING;
  landing_start_tick = 0;
  landing_reference_z = traj_height;
  experiment_start_rtos_tick = xTaskGetTickCount();

  obs_start << 0.7f, 0.3f, 0.5f;      
  obs_velocity << 0.0f, -0.1f, 0.0f; 
  obs_center = obs_start;             
  obs_start_time = 0;                 

  problem.en_psd = 0;
  DEBUG_PRINT("Benchmark mode %u, maneuver %u, max_iter %u, task_pri %u\n",
              (unsigned int)controller_id,
              (unsigned int)trajectory_id,
              (unsigned int)benchmark_max_iter,
              (unsigned int)TINYMPC_TASK_PRI);
  DEBUG_PRINT("Reference altitude: %.2f m; barrier skip when z>%.2f and vz>%.2f\n",
              (double)traj_height, (double)raya_skip_z,
              (double)raya_skip_vz);
#if TINYMPC_OOT_TAKEOFF
  DEBUG_PRINT("OOT takeoff: cfclient trigger>=%.2fm ramp=%.2fm/s figure8_start=%lums_after_OOT_takeoff no_z_vz_settle target=%.2fm timeout=%lums\n",
              (double)TINYMPC_TAKEOFF_TRIGGER_Z_M,
              (double)TINYMPC_TAKEOFF_ASCENT_MPS,
              (unsigned long)TINYMPC_TAKEOFF_TRAJECTORY_START_MS,
              (double)traj_height,
              (unsigned long)TINYMPC_TAKEOFF_TIMEOUT_MS);
#else
  DEBUG_PRINT("OOT takeoff: disabled; switch to controller 6 after external takeoff\n");
#endif

#if TINYMPC_COMMON_Z_GUARD
  DEBUG_PRINT("RAYA initial cost: mpc_hz=20 model_hz=20 rho=5 base_w=0 base_qz=1 Qz=100 R=4 bridge=x%u guards=z[ref,ref+0.20] xy=free cap=0.05\n",
              (unsigned int)kPidBridgeIndex);
#else
  DEBUG_PRINT("RAYA initial cost: mpc_hz=20 model_hz=20 rho=5 base_w=0 base_qz=1 Qz=100 R=4 bridge=x%u z_guard=off(raw_preview) xy=free cap=0.05\n",
              (unsigned int)kPidBridgeIndex);
#endif
  DEBUG_PRINT("MPC->PID: xyz=x%u, z_guard=[ref,ref+0.20], yaw=raw_rz, output_hold=200ms, safety_hold=500ms\n",
              (unsigned int)kPidBridgeIndex);
#if TINYMPC_FAULT_MOTOR_RETAINED_PERCENT < 100
  DEBUG_PRINT("Motor fault schedule: model=multiplicative_step pre_onset=100%% onset=%lums_after_TRAJ_START global=%u%% fault=M%u(index=%u) retained=%u%% equiv_fullscale_pwm=%u ramp=none placement=post_mixer_post_battery model_bounds=unchanged\n",
              (unsigned long)TINYMPC_FAULT_ONSET_MS,
              (unsigned int)TINYMPC_GLOBAL_MOTOR_CAP_PERCENT,
              (unsigned int)(TINYMPC_FAULT_MOTOR_INDEX + 1),
              (unsigned int)TINYMPC_FAULT_MOTOR_INDEX,
              (unsigned int)TINYMPC_FAULT_MOTOR_RETAINED_PERCENT,
              (unsigned int)kFaultMotorEquivalentFullScalePwm);
#else
  DEBUG_PRINT("Motor handicap: disabled; global=100%% all_motors_retained=100%%\n");
#endif
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_HOVER
  DEBUG_PRINT("Reference origin: xy=estimator_zero z=%.2f m\n",
              (double)traj_height);
#elif TINYMPC_OOT_TAKEOFF
  DEBUG_PRINT("Reference origin: xyz=measured_at_forced_timed_start nominal_takeoff_target=%.2f m\n",
              (double)traj_height);
#else
  DEBUG_PRINT("Reference origin: xy=OOT_activation z=%.2f m\n",
              (double)traj_height);
#endif
  DEBUG_PRINT("RAYA adaptive: cache=16x4 evals=1 activeH=%u timing=bar/rl/cache_cycles\n",
              (unsigned int)raya_active_horizon);
#if TINYMPC_BENCH_PROFILE
  DEBUG_PRINT("BENCH PROFILE: MOTORS FORCED OFF, canned_states=%u hold_steps=%lu tanhf_subcounter=%u\n",
              (unsigned int)kBenchStateCount,
              (unsigned long)kBenchStateHoldSteps,
              (unsigned int)TINYMPC_PROFILE_TANH);
#endif
#if TINYMPC_XYZ_CONSOLE_DECIMATION > 0
  DEBUG_PRINT("RAYA record: step,time_ms,xyz_mm,ref_xyz_mm,barrier_milli,margin_milli,authority_milli,qz_milli,solve_us,total_us,fault,overruns\n");
#endif
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8
#if TINYMPC_OOT_TAKEOFF
  DEBUG_PRINT("Figure8: source=traj_fig8_12.h source_hz=%u samples=%u passes=%u cycles=%u lobes=%u duration=%.2f s origin=forced_timed_start start_delay=%lums z_origin=measured_at_start feedforward=U_ref_data\n",
              (unsigned int)kFigure8SourceRateHz,
              (unsigned int)kFigure8StateCount,
              (unsigned int)kFigure8TablePassCount,
              (unsigned int)kFigure8TotalCycleCount,
              (unsigned int)(2 * kFigure8TotalCycleCount),
              (double)traj_duration,
              (unsigned long)TINYMPC_TAKEOFF_TRAJECTORY_START_MS);
#else
  DEBUG_PRINT("Figure8: source=traj_fig8_12.h source_hz=%u samples=%u passes=%u cycles=%u lobes=%u duration=%.2f s origin=OOT_activation z_offset=%.2f m\n",
              (unsigned int)kFigure8SourceRateHz,
              (unsigned int)kFigure8StateCount,
              (unsigned int)kFigure8TablePassCount,
              (unsigned int)kFigure8TotalCycleCount,
              (unsigned int)(2 * kFigure8TotalCycleCount),
              (double)traj_duration,
              (double)(traj_height - X_ref_data[0][2]));
#endif
#endif

  reset_flight_state();
  previous_controller_id = controller_id;
  previous_trajectory_id = trajectory_id;
  
  runTaskSemaphore = xSemaphoreCreateBinary();

  dataMutex = xSemaphoreCreateMutexStatic(&dataMutexBuffer);

  STATIC_MEM_TASK_CREATE(tinympcControllerTask, tinympcControllerTask, TINYMPC_TASK_NAME, NULL, TINYMPC_TASK_PRI);

  isInit = true;
  
}

static void UpdateHorizonReference(const setpoint_t *setpoint)
{
  (void)setpoint;
  if (flight_phase == FLIGHT_PHASE_TAKEOFF)
  {
    params.Uref.setZero();
    const float elapsed_s =
        0.001f * static_cast<float>(
            T2M(xTaskGetTickCount() - takeoff_start_rtos_tick));
    const float current_reference_z =
        fminf(traj_height,
              takeoff_start_z + TINYMPC_TAKEOFF_ASCENT_MPS * elapsed_s);
    const float yaw_radians = radians(takeoff_hold_yaw);
    const float yaw_rodrigues = tanf(0.5f * yaw_radians);
    for (int i = 0; i < NHORIZON; ++i) {
      const float horizon_reference_z =
          fminf(traj_height,
                current_reference_z +
                    TINYMPC_TAKEOFF_ASCENT_MPS *
                        (static_cast<float>(i) / MPC_RATE));
      params.Xref.col(i).setZero();
      params.Xref(0, i) = takeoff_hold_x;
      params.Xref(1, i) = takeoff_hold_y;
      params.Xref(2, i) = horizon_reference_z;
      params.Xref(5, i) = yaw_rodrigues;
      params.Xref(8, i) =
          horizon_reference_z < traj_height
              ? TINYMPC_TAKEOFF_ASCENT_MPS
              : 0.0f;
    }
    Xref_end = params.Xref.col(NHORIZON - 1);
    return;
  }

  if (enable_traj)
  {
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_FIGURE8

#if TINYMPC_OOT_TAKEOFF
    const float z_offset = trajectory_origin_z - X_ref_data[0][2];
#else
    const float z_offset = traj_height - X_ref_data[0][2];
#endif
    for (int i = 0; i < NHORIZON; ++i) {
      const int unwrapped_source_index =
          ((traj_index + i) * kFigure8SourceRateHz) / MPC_RATE;
      int source_index;
      if (unwrapped_source_index >= kFigure8TotalSourceIntervals) {
        source_index = kFigure8StateCount - 1;
      } else {
        source_index =
            unwrapped_source_index % kFigure8SourceIntervalsPerPass;
      }
      for (int state_index = 0; state_index < NSTATES; ++state_index) {
        params.Xref(state_index, i) =
            X_ref_data[source_index][state_index];
      }
      params.Xref(0, i) += trajectory_origin_x;
      params.Xref(1, i) += trajectory_origin_y;
      params.Xref(2, i) += z_offset;
    }
    for (int i = 0; i < NHORIZON - 1; ++i) {
      const int unwrapped_source_index =
          ((traj_index + i) * kFigure8SourceRateHz) / MPC_RATE;
      const int source_index =
          unwrapped_source_index >= kFigure8TotalSourceIntervals
              ? kFigure8InputCount - 1
              : unwrapped_source_index % kFigure8SourceIntervalsPerPass;
      for (int input_index = 0; input_index < NINPUTS; ++input_index) {
        params.Uref(input_index, i) =
            U_ref_data[source_index][input_index];
      }
    }
#else
    params.Uref.setZero();
    const float dt = 1.0f / MPC_RATE;
    const float base_t = traj_index * dt;
    for (int i = 0; i < NHORIZON; ++i) {
      const float t = base_t + i * dt;
      float x = 0.0f;
      float y = 0.0f;
      float vx = 0.0f;
      float vy = 0.0f;
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_HOVER
      
      (void)t;
#elif TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_LINE_X
      x = fminf(traj_speed * t, traj_dist);
      vx = traj_speed * t < traj_dist ? traj_speed : 0.0f;
#elif TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_LINE_Y
      y = fminf(traj_speed * t, traj_dist);
      vy = traj_speed * t < traj_dist ? traj_speed : 0.0f;
#elif TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_CIRCLE
      
      x = traj_radius * sinf(traj_omega * t);
      y = traj_radius * (1.0f - cosf(traj_omega * t));
      vx = traj_radius * traj_omega * cosf(traj_omega * t);
      vy = traj_radius * traj_omega * sinf(traj_omega * t);
#endif
      params.Xref.col(i).setZero();
#if TINYMPC_TRAJECTORY == TINYMPC_TRAJECTORY_HOVER
      params.Xref(0, i) = x;
      params.Xref(1, i) = y;
#else
      params.Xref(0, i) = trajectory_origin_x + x;
      params.Xref(1, i) = trajectory_origin_y + y;
#endif
      params.Xref(2, i) = traj_height;
      params.Xref(6, i) = vx;
      params.Xref(7, i) = vy;
    }
#endif
    Xref_end = params.Xref.col(NHORIZON - 1);

    if (traj_index < max_traj_index) {
      traj_index++;
    } else {

      DEBUG_PRINT("TRAJ DONE: idx=%d, max=%d; requesting landing\n",
                  traj_index, max_traj_index);
      enable_traj = false;

      params.Xref = Xref_end.replicate<1, NHORIZON>();
      params.Uref.setZero();
      landing_start_tick = 0;
      flight_phase = FLIGHT_PHASE_LANDING;
    }
  }
  else
  {
    params.Xref = Xref_end.replicate<1, NHORIZON>();
    params.Uref.setZero();
  }
}

bool controllerOutOfTreeTest()
{
  
  return true;
}

static void tinympcControllerTask(void *parameters)
{

  uint32_t nowMs = T2M(xTaskGetTickCount());
  uint32_t nextMpcMs = nowMs;

  static uint32_t task_loop_count = 0;
  while (true)
  {
    
    xSemaphoreTake(runTaskSemaphore, portMAX_DELAY);
    
    task_loop_count++;
    if (task_loop_count <= 3) {
      DEBUG_PRINT("MPC task loop %lu\n", task_loop_count);
    }

    xSemaphoreTake(dataMutex, portMAX_DELAY);
    memcpy(&setpoint_task, &setpoint_data, sizeof(setpoint_t));
    memcpy(&sensors_task, &sensors_data, sizeof(sensorData_t));
    memcpy(&state_task, &state_data, sizeof(state_t));
    memcpy(&control_task, &control_data, sizeof(control_t));
    xSemaphoreGive(dataMutex);

    nowMs = T2M(xTaskGetTickCount());
    if (nowMs >= nextMpcMs)
    {
      nextMpcMs = nowMs + (1000.0f / MPC_RATE);

      if (controller_id != previous_controller_id ||
          trajectory_id != previous_trajectory_id) {
        previous_controller_id = controller_id;
        previous_trajectory_id = trajectory_id;
        reset_cost_cache();
        reset_flight_state();
        DEBUG_PRINT("Benchmark reset: mode=%u maneuver=%u\n",
                    (unsigned int)controller_id,
                    (unsigned int)trajectory_id);
      }
      problem.max_iter = benchmark_max_iter > 0 ? benchmark_max_iter : 1;

      if (flight_phase != FLIGHT_PHASE_TAKEOFF &&
          flight_phase != FLIGHT_PHASE_TRACKING) {
        continue;
      }
      if (flight_phase == FLIGHT_PHASE_TAKEOFF &&
          tracking_start_pending) {

        enable_traj = true;
        flight_phase = FLIGHT_PHASE_TRACKING;
        traj_index = 0;
        experiment_start_rtos_tick = xTaskGetTickCount();
        reset_tracking_measurements();

        set_motor_fault_active(false);
        motor_fault_actual_onset_ms = 0;
        powerDistributionResetCapStats();
        refresh_motor_cap_stats();
        tracking_start_pending = false;
        DEBUG_PRINT(
            "TRAJ START: z=%.2f origin=(%.2f,%.2f,%.2f) EXP clock reset; Z metric begins at t+3s\n",
            (double)state_task.position.z,
            (double)trajectory_origin_x,
            (double)trajectory_origin_y,
            (double)trajectory_origin_z);
      }

      if (TINYMPC_FAULT_MOTOR_RETAINED_PERCENT < 100 &&
          flight_phase == FLIGHT_PHASE_TRACKING && !motor_fault_active) {
        const uint32_t experiment_elapsed_ms =
            T2M(xTaskGetTickCount() - experiment_start_rtos_tick);
        if (experiment_elapsed_ms >= TINYMPC_FAULT_ONSET_MS) {
          set_motor_fault_active(true);
          motor_fault_actual_onset_ms = experiment_elapsed_ms;

          powerDistributionResetCapStats();
          refresh_motor_cap_stats();
          DEBUG_PRINT(
              "MOTOR FAULT ON: t=%lums after_TRAJ_START model=multiplicative_step fault=M%u(index=%u) retained=%u%% equiv_fullscale_pwm=%u ramp=none\n",
              (unsigned long)motor_fault_actual_onset_ms,
              (unsigned int)(TINYMPC_FAULT_MOTOR_INDEX + 1),
              (unsigned int)TINYMPC_FAULT_MOTOR_INDEX,
              (unsigned int)TINYMPC_FAULT_MOTOR_RETAINED_PERCENT,
              (unsigned int)kFaultMotorEquivalentFullScalePwm);
        }
      }

      const uint32_t controller_cycle_start_cycles = DWT->CYCCNT;
      const uint32_t controller_cycle_start_us = usecTimestamp();

      phi = quat_2_rp(normalize_quat(state_task.attitudeQuaternion)); 
      problem.x.col(0) << state_task.position.x, state_task.position.y, state_task.position.z,
          phi.x, phi.y, phi.z,
          state_task.velocity.x, state_task.velocity.y, state_task.velocity.z,
          radians(sensors_task.gyro.x), radians(sensors_task.gyro.y), radians(sensors_task.gyro.z);
#if TINYMPC_BENCH_PROFILE
      load_bench_state(&problem.x, benchmark_step);
#endif

      if (task_loop_count <= 3) {
        DEBUG_PRINT("x0: pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f)\n",
                    (double)problem.x(0, 0), (double)problem.x(1, 0), (double)problem.x(2, 0),
                    (double)problem.x(6, 0), (double)problem.x(7, 0), (double)problem.x(8, 0));
      }

      UpdateHorizonReference(&setpoint_task);

      tracking_pos_x = problem.x(0, 0);
      tracking_pos_y = problem.x(1, 0);
      tracking_pos_z = problem.x(2, 0);
      tracking_vel_x = problem.x(6, 0);
      tracking_vel_y = problem.x(7, 0);
      tracking_vel_z = problem.x(8, 0);
#if TINYMPC_BENCH_PROFILE

      tracking_roll_deg = problem.x(3, 0) * 57.2957795f;
      tracking_pitch_deg = problem.x(4, 0) * 57.2957795f;
      tracking_yaw_deg = problem.x(5, 0) * 57.2957795f;
#else
      tracking_roll_deg = state_task.attitude.roll;
      tracking_pitch_deg = state_task.attitude.pitch;
      tracking_yaw_deg = state_task.attitude.yaw;
#endif
      
      if (task_loop_count <= 3) {
        DEBUG_PRINT("ref: (%.2f,%.2f,%.2f)\n",
                    (double)params.Xref(0,0), (double)params.Xref(1,0), (double)params.Xref(2,0));
      }
      raya_barrier_cycles = 0;
      raya_activation_cycles = 0;
      raya_rl_cycles = 0;
      raya_cache_cycles = 0;
      mpc_solve_cycles = 0;
      controller_total_cycles = 0;
      raya_cache_ok = 0;
      raya_authority_w = 0.0f;
      raya_authority_w_requested = 0.0f;
      raya_qz = 1.0f;
      raya_qz_requested = 1.0f;
      raya_w_index = 0;
      raya_qz_index = 0;
      raya_no_oracle_score = 0.0f;
      raya_no_oracle_score_raw = 0.0f;
      raya_no_oracle_active = 0;

      RayaBarrierEval embedded_eval_x0;
      uint32_t embedded_eval_us = 0;
      {
        const uint32_t embedded_eval_start_us = usecTimestamp();
        const uint32_t embedded_barrier_start_cycles = DWT->CYCCNT;
        raya_eval_barrier(
            problem.x.col(0), raya_az_coeff, raya_gravity_comp,
            radians(raya_fail_roll_deg), radians(raya_fail_pitch_deg),
            &embedded_eval_x0);
        raya_barrier_cycles =
            DWT->CYCCNT - embedded_barrier_start_cycles;
        raya_activation_cycles = embedded_eval_x0.activation_cycles;
        embedded_eval_us = usecTimestamp() - embedded_eval_start_us;
        const tinytype embedded_h =
            guarded_raya_h(problem.x.col(0), embedded_eval_x0.h);
        const tinytype embedded_margin =
            raya_effective_margin(problem.x.col(0), 0);
        raya_embedded::observe_no_oracle(
            &raya_embedded_runtime, problem.x.col(0), embedded_h,
            embedded_margin, radians(raya_fail_roll_deg),
            radians(raya_fail_pitch_deg));
        const uint32_t embedded_rl_start_cycles = DWT->CYCCNT;
        raya_embedded::update(
            &raya_embedded_runtime, problem.x.col(0), params.Xref.col(0),
            embedded_h, embedded_margin, radians(raya_fail_roll_deg),
            radians(raya_fail_pitch_deg));
        raya_rl_cycles = DWT->CYCCNT - embedded_rl_start_cycles;
        const uint32_t embedded_cache_start_cycles = DWT->CYCCNT;
        raya_cache_ok =
            raya_embedded::install_cache(raya_embedded_runtime, &params) ? 1 : 0;
        raya_cache_cycles =
            DWT->CYCCNT - embedded_cache_start_cycles;
        raya_authority_w = raya_embedded_runtime.applied_w;
        raya_authority_w_requested = raya_embedded_runtime.requested_w;
        raya_qz = raya_embedded_runtime.applied_qz;
        raya_qz_requested = raya_embedded_runtime.requested_qz;
        raya_w_index =
            static_cast<uint8_t>(raya_embedded_runtime.w_index);
        raya_qz_index =
            static_cast<uint8_t>(raya_embedded_runtime.qz_index);
        raya_no_oracle_score = raya_embedded_runtime.no_oracle_score;
        raya_no_oracle_score_raw =
            raya_embedded_runtime.no_oracle_score_raw;
        raya_no_oracle_active =
            raya_embedded_runtime.no_oracle_active ? 1 : 0;
        if (!raya_cache_ok) {
          DEBUG_PRINT("RAYA embedded cache validation failed\n");

          enable_traj = false;
          mpc_has_run = true;
          landing_start_tick = 0;
          flight_phase = FLIGHT_PHASE_LANDING;
          reset_cost_cache();
          reset_solver_warm_start();
        }
      }

      float obs_elapsed = 0.0f;

      const bool constraint_hold =
          (!mpc_has_run) ||
          ((xTaskGetTickCount() - controller_activate_rtos_tick) < M2T(500));
      static uint32_t cstr_log_cnt = 0;
      int cstr_active_count = 0;
      const float dt_horizon = 1.0f / MPC_RATE;  
      raya_eval_us = embedded_eval_us;
      raya_h = 0.0f;
      raya_raw = 0.0f;
      raya_active = 0;
      raya_active_count = 0;
      raya_grad_norm = 0.0f;
      raya_margin_eff = 0.0f;
      raya_threshold = 0.0f;

      for (int i = 0; i < NHORIZON; i++)
      {
        params.x_min[i] = tiny_VectorNc::Constant(-1000);
        params.x_max[i] = tiny_VectorNc::Constant(1000);
        params.A_constraints[i] = tiny_MatrixNcNx::Zero();

        const bool install_safety_row =
            !constraint_hold && i < raya_active_horizon;
        if (install_safety_row) {
          const tiny_VectorNx xbar = problem.x.col(i);
          tiny_VectorNx grad = tiny_VectorNx::Zero();
          tinytype barrier_h = 0.0f;
          tinytype constraint_h = 0.0f;
          tinytype raw_h = 0.0f;
          {

            grad = embedded_eval_x0.grad;
            const tinytype linearized_delta =
                grad.dot(xbar - problem.x.col(0));
            raw_h = embedded_eval_x0.raw + linearized_delta;
            constraint_h = embedded_eval_x0.h + linearized_delta;
            barrier_h = guarded_raya_h(xbar, constraint_h);
          }
          for (int j = 0; j < NSTATES; ++j) {
            grad(j) = clamp_tiny(grad(j), tinytype(-2.0f), tinytype(2.0f));
          }
          const tinytype grad_norm = grad.norm();
          const tinytype margin_eff = raya_effective_margin(xbar, i);
          const tinytype activation_threshold =
              raya_h_deadband > (margin_eff + raya_act_slack) ? raya_h_deadband : (margin_eff + raya_act_slack);
          const bool skipped = barrier_skipped_high_altitude(xbar);
          const bool active = !skipped && (grad_norm > tinytype(1e-6f)) &&
                              (barrier_h < activation_threshold);

          if (i == 0) {
            raya_h = barrier_h;
            raya_raw = raw_h;
            raya_grad_norm = grad_norm;
            raya_margin_eff = margin_eff;
            raya_threshold = activation_threshold;
            raya_active = active ? 1 : 0;
          }

          if (active) {
            params.A_constraints[i] = -grad.transpose();
            tinytype demand = margin_eff - constraint_h;

            if (demand > raya_demand_cap) {
              demand = raya_demand_cap;
            }
            params.x_max[i](0) = -(grad.dot(xbar) + demand);
            cstr_active_count++;
            if (raya_active_count < 255) {
              raya_active_count++;
            }
          }
        }

      }
      
      if (task_loop_count <= 3) {
        DEBUG_PRINT("SAFE TV mode=%u: h=%.3f raw=%.3f grad=%.3f margin=%.3f thr=%.3f active0=%u active_count=%u eval=%lu us\n",
                    (unsigned int)controller_id,
                    (double)raya_h, (double)raya_raw, (double)raya_grad_norm,
                    (double)raya_margin_eff, (double)raya_threshold,
                    (unsigned int)raya_active, (unsigned int)raya_active_count, raya_eval_us);
      }
      
      const int requested_cache_level = cstr_active_count > 0 ? 1 : 0;
      if (requested_cache_level != prev_cache_level) {
        DEBUG_PRINT("Cache level changed: %d -> %d\n",
                    prev_cache_level, requested_cache_level);
        reset_solver_warm_start();
        problem.cache_level = requested_cache_level;
        prev_cache_level = requested_cache_level;
      } else {
        problem.cache_level = requested_cache_level;
      }
      if (problem.cache_level == 0) {
        problem.y.setZero();
        problem.g.setZero();
      }

      problem.iter = 0;

      if (task_loop_count <= 3) {
        DEBUG_PRINT("MPC solve start\n");
      }
      mpc_start_timestamp = usecTimestamp();
      const uint32_t mpc_solve_start_cycles = DWT->CYCCNT;
      solve_admm(&problem, &params);
      mpc_solve_cycles = DWT->CYCCNT - mpc_solve_start_cycles;
      mpc_time_us = usecTimestamp() - mpc_start_timestamp;
      {
        raya_embedded::update_saturation(
            &raya_embedded_runtime, problem.u.col(0),
            params.u_min.col(0), params.u_max.col(0));
        raya_embedded::save_prediction(
            &raya_embedded_runtime, problem.x.col(1));
      }
      ++benchmark_step;
      if (task_loop_count <= 3) {
        DEBUG_PRINT("MPC solve done, iter=%d\n", problem.iter);
      }
      if (task_loop_count <= 3) {
        DEBUG_PRINT("MPC time=%lu us\n", mpc_time_us);
      }
      if (task_loop_count == 1) {
        DEBUG_PRINT("MPC stack free=%lu words\n",
                    (unsigned long)uxTaskGetStackHighWaterMark(NULL));
      }

      static uint32_t cert_log_cnt = 0;
      bool certified_k0 = true;
      float trace_gap_k0 = 0.0f;
      float eta_min_k0 = 1000.0f;

      const tinytype preview_z = problem.x(2, kPidBridgeIndex);
      mpc_setpoint_task = problem.x.col(kPidBridgeIndex);

      float command_yaw_deg = mpc_setpoint_task(5);
      if (flight_phase == FLIGHT_PHASE_TAKEOFF) {

        mpc_setpoint_task(0) = takeoff_hold_x;
        mpc_setpoint_task(1) = takeoff_hold_y;
        command_yaw_deg = takeoff_hold_yaw;
      }
      tracking_preview_z = preview_z;
#if TINYMPC_COMMON_Z_GUARD
      const tinytype reference_z = params.Xref(2, 0);

      mpc_setpoint_task(2) =
          clamp_tiny(preview_z, reference_z, reference_z + tinytype(0.20f));
#else

      mpc_setpoint_task(2) = preview_z;
#endif
      tracking_cmd_z = mpc_setpoint_task(2);

      tracking_cmd_x = mpc_setpoint_task(0);
      tracking_cmd_y = mpc_setpoint_task(1);
      controller_total_us = usecTimestamp() - controller_cycle_start_us;
      controller_total_cycles =
          DWT->CYCCNT - controller_cycle_start_cycles;
      if (controller_total_cycles > max_step_cycles) {
        max_step_cycles = controller_total_cycles;
      }
      if (controller_total_cycles > kMpcDeadlineCycles) {
        ++deadline_overrun_count;
      }
      refresh_motor_cap_stats();

#if TINYMPC_XYZ_CONSOLE_DECIMATION > 0
      if ((benchmark_step % TINYMPC_XYZ_CONSOLE_DECIMATION) == 0) {
        if (flight_phase == FLIGHT_PHASE_TRACKING) {
          const uint32_t experiment_elapsed_ms =
              T2M(xTaskGetTickCount() - experiment_start_rtos_tick);
          DEBUG_PRINT(
            "RAYA,%lu,%lu,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%lu,%lu,%u,%lu\n",
            (unsigned long)benchmark_step,
            (unsigned long)experiment_elapsed_ms,
            console_scaled(tracking_pos_x, 1000.0f),
            console_scaled(tracking_pos_y, 1000.0f),
            console_scaled(tracking_pos_z, 1000.0f),
            console_scaled(params.Xref(0, 0), 1000.0f),
            console_scaled(params.Xref(1, 0), 1000.0f),
            console_scaled(params.Xref(2, 0), 1000.0f),
            console_scaled(raya_h, 1000.0f),
            console_scaled(raya_margin_eff, 1000.0f),
            console_scaled(raya_authority_w, 1000.0f),
            console_scaled(raya_qz, 1000.0f),
            (unsigned long)mpc_time_us,
            (unsigned long)controller_total_us,
            (unsigned int)motor_fault_active,
            (unsigned long)deadline_overrun_count);
        }
      }
#endif

      if (flight_phase == FLIGHT_PHASE_LANDING ||
          flight_phase == FLIGHT_PHASE_COMPLETE) {
        refresh_motor_cap_stats();
        DEBUG_PRINT(
            "TIMING SUMMARY: max_step_cycles=%lu deadline_cycles=%lu overruns=%lu\n",
            (unsigned long)max_step_cycles,
            (unsigned long)kMpcDeadlineCycles,
            (unsigned long)deadline_overrun_count);
        DEBUG_PRINT(
            "MOTOR FAULT SUMMARY: model=multiplicative_step schedule_onset=%lums active=%u actual_onset=%lums global=%u fault=M%u index=%u retained=%u equiv_fullscale_pwm=%u samples=%lu affected=%lu peak_req=%ld last_req=%ld last_app=%u\n",
            (unsigned long)TINYMPC_FAULT_ONSET_MS,
            (unsigned int)motor_fault_active,
            (unsigned long)motor_fault_actual_onset_ms,
            (unsigned int)TINYMPC_GLOBAL_MOTOR_CAP_PERCENT,
            (unsigned int)(TINYMPC_FAULT_MOTOR_INDEX + 1),
            (unsigned int)TINYMPC_FAULT_MOTOR_INDEX,
            (unsigned int)TINYMPC_FAULT_MOTOR_RETAINED_PERCENT,
            (unsigned int)motor_cap_stats.maxAllowed,
            (unsigned long)motor_cap_stats.sampleCount,
            (unsigned long)motor_cap_stats.capHitCount,
            (long)motor_cap_stats.peakRequested,
            (long)motor_cap_stats.latestRequested,
            (unsigned int)motor_cap_stats.latestApplied);
      }

      if (task_loop_count <= 3) {
        DEBUG_PRINT("setpoint: x=%.2f y=%.2f z=%.2f preview_z=%.2f\n",
                    (double)mpc_setpoint_task(0), (double)mpc_setpoint_task(1),
                    (double)mpc_setpoint_task(2),
                    (double)preview_z);
      }

      xSemaphoreTake(dataMutex, portMAX_DELAY);
      mpc_setpoint = mpc_setpoint_task;
      mpc_yaw_setpoint_deg = command_yaw_deg;
      init_vel_z = problem.x(8, 0);
      mpc_has_run = true; 
      xSemaphoreGive(dataMutex);
    }
  }
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick)
{
  setpoint_t hold_sp;
  fill_hold_setpoint(&hold_sp, state);

  if (!isInit || (dataMutex == NULL) || (runTaskSemaphore == NULL)) {
    controllerPid(control, &hold_sp, sensors, state, tick);
    return;
  }

  if (xSemaphoreTake(dataMutex, M2T(2)) != pdTRUE) {
    controllerPid(control, &hold_sp, sensors, state, tick);
    return;
  }
  memcpy(&setpoint_data, setpoint, sizeof(setpoint_t));
  memcpy(&sensors_data, sensors, sizeof(sensorData_t));
  memcpy(&state_data, state, sizeof(state_t));

  const bool controller_reactivated =
      (last_controller_tick == 0) || ((tick - last_controller_tick) > M2T(200));
  if (controller_reactivated) {
    controller_activate_tick = tick;
    controller_activate_rtos_tick = xTaskGetTickCount();
    experiment_start_rtos_tick = controller_activate_rtos_tick;
    mpc_has_run = false;
    takeoff_trigger_armed = false;
#if TINYMPC_TRAJECTORY != TINYMPC_TRAJECTORY_HOVER
    trajectory_origin_x = state->position.x;
    trajectory_origin_y = state->position.y;
    trajectory_origin_z = state->position.z;
    DEBUG_PRINT("Trajectory origin: x=%.2f y=%.2f z=%.2f\n",
                (double)trajectory_origin_x,
                (double)trajectory_origin_y,
                (double)trajectory_origin_z);
#endif
    
    mpc_setpoint = tiny_VectorNx::Zero();
    mpc_setpoint(0) = state->position.x;
    mpc_setpoint(1) = state->position.y;
    mpc_setpoint(2) = state->position.z;
    mpc_yaw_setpoint_deg = state->attitude.yaw;
    DEBUG_PRINT("OOT activated at z=%.2f\n", (double)state->position.z);
  }
  last_controller_tick = tick;

#if TINYMPC_BENCH_PROFILE

  xSemaphoreGive(dataMutex);
  xSemaphoreGive(runTaskSemaphore);
  stop_motors(control);
  return;
#endif

#if TINYMPC_OOT_TAKEOFF
  if (flight_phase == FLIGHT_PHASE_WAITING_FOR_TAKEOFF) {
    const bool takeoff_requested = oot_takeoff_requested(setpoint);
    if (!takeoff_requested) {

      takeoff_trigger_armed = true;
    }
    if (!takeoff_requested || !takeoff_trigger_armed) {
      xSemaphoreGive(dataMutex);
      stop_motors(control);
      return;
    }

    takeoff_trigger_armed = false;
    takeoff_start_tick = tick;
    takeoff_start_rtos_tick = xTaskGetTickCount();
    takeoff_start_z = fmaxf(0.0f, state->position.z);
    takeoff_hold_x = state->position.x;
    takeoff_hold_y = state->position.y;
    takeoff_hold_yaw = state->attitude.yaw;
    trajectory_origin_x = takeoff_hold_x;
    trajectory_origin_y = takeoff_hold_y;
    trajectory_origin_z = takeoff_start_z;
    enable_traj = false;
    mpc_has_run = false;
    tracking_start_pending = false;
    controller_activate_tick = tick;
    controller_activate_rtos_tick = takeoff_start_rtos_tick;
    experiment_start_rtos_tick = takeoff_start_rtos_tick;
    reset_solver_warm_start();
    DEBUG_PRINT(
        "TAKEOFF START: trigger_z=%.2f start_z=%.2f target=%.2f hold=(%.2f,%.2f) ramp=%.2fm/s\n",
        (double)setpoint->position.z,
        (double)takeoff_start_z,
        (double)traj_height,
        (double)takeoff_hold_x,
        (double)takeoff_hold_y,
        (double)TINYMPC_TAKEOFF_ASCENT_MPS);
    flight_phase = FLIGHT_PHASE_TAKEOFF;
  }
#endif

  if (RATE_DO_EXECUTE(LOWLEVEL_RATE, tick))
  {
    if (flight_phase == FLIGHT_PHASE_TAKEOFF) {
      const uint32_t takeoff_elapsed_ms =
          T2M(tick - takeoff_start_tick);

      if (!tracking_start_pending &&
          flight_phase == FLIGHT_PHASE_TAKEOFF &&
          takeoff_elapsed_ms >= TINYMPC_TAKEOFF_TRAJECTORY_START_MS) {
        trajectory_origin_x = state->position.x;
        trajectory_origin_y = state->position.y;
        trajectory_origin_z = state->position.z;
        tracking_start_pending = true;
        obs_start_time = 0;
        DEBUG_PRINT(
            "TAKEOFF COMPLETE: forced_start t=%lums z=%.2f vz=%.2f; trajectory queued origin=(%.2f,%.2f,%.2f) no_z_vz_gate\n",
            (unsigned long)takeoff_elapsed_ms,
            (double)state->position.z,
            (double)state->velocity.z,
            (double)trajectory_origin_x,
            (double)trajectory_origin_y,
            (double)trajectory_origin_z);
      }

      if (!tracking_start_pending &&
          flight_phase == FLIGHT_PHASE_TAKEOFF &&
          takeoff_elapsed_ms >= TINYMPC_TAKEOFF_TIMEOUT_MS) {
        enable_traj = false;
        landing_start_tick = 0;
        flight_phase = FLIGHT_PHASE_LANDING;
        DEBUG_PRINT(
            "TAKEOFF ABORT: timeout=%lums z=%.2f vz=%.2f; requesting landing\n",
            (unsigned long)takeoff_elapsed_ms,
            (double)state->position.z,
            (double)state->velocity.z);
      }
    }

    if (flight_phase == FLIGHT_PHASE_LANDING) {
      if (landing_start_tick == 0) {
        landing_start_tick = tick;
        landing_hold_x = state->position.x;
        landing_hold_y = state->position.y;
        landing_hold_yaw = state->attitude.yaw;
        landing_start_z = state->position.z;
        landing_reference_z = landing_start_z;
        DEBUG_PRINT("LANDING: start z=%.2f, hold=(%.2f,%.2f)\n",
                    (double)landing_start_z,
                    (double)landing_hold_x,
                    (double)landing_hold_y);
      }

      const float landing_elapsed_s =
          0.001f * static_cast<float>(T2M(tick - landing_start_tick));
      landing_reference_z =
          fmaxf(TINYMPC_LANDING_TARGET_Z_M,
                landing_start_z -
                    TINYMPC_LANDING_DESCENT_MPS * landing_elapsed_s);

      memset(&mpc_setpoint_pid, 0, sizeof(mpc_setpoint_pid));
      mpc_setpoint_pid.mode.yaw = modeAbs;
      mpc_setpoint_pid.mode.x = modeAbs;
      mpc_setpoint_pid.mode.y = modeAbs;
      mpc_setpoint_pid.mode.z = modeAbs;
      mpc_setpoint_pid.position.x = landing_hold_x;
      mpc_setpoint_pid.position.y = landing_hold_y;
      mpc_setpoint_pid.position.z = landing_reference_z;
      mpc_setpoint_pid.attitude.yaw = landing_hold_yaw;
      controllerPid(control, &mpc_setpoint_pid, sensors, state, tick);

      const bool landing_reference_complete =
          landing_reference_z <= TINYMPC_LANDING_TARGET_Z_M + 1e-4f;
      const bool near_floor =
          state->position.z <= TINYMPC_LANDING_TOUCHDOWN_Z_M;
      const bool vertically_settled =
          fabsf(state->velocity.z) <= TINYMPC_LANDING_TOUCHDOWN_VZ_MPS;
      if (landing_reference_complete && near_floor && vertically_settled) {
        flight_phase = FLIGHT_PHASE_COMPLETE;
        stop_motors(control);
        DEBUG_PRINT("LANDING COMPLETE: t=%lu ms, x=%.3f, y=%.3f, z=%.3f, vz=%.3f; motors off\n",
                    (unsigned long)T2M(tick - controller_activate_tick),
                    (double)state->position.x,
                    (double)state->position.y,
                    (double)state->position.z,
                    (double)state->velocity.z);
      }
    } else if (flight_phase == FLIGHT_PHASE_COMPLETE) {
      stop_motors(control);
    } else {
      memset(&mpc_setpoint_pid, 0, sizeof(mpc_setpoint_pid));
      mpc_setpoint_pid.mode.yaw = modeAbs;
      mpc_setpoint_pid.mode.x = modeAbs;
      mpc_setpoint_pid.mode.y = modeAbs;
      mpc_setpoint_pid.mode.z = modeAbs;

      const bool hold_output =
          (!mpc_has_run) || ((tick - controller_activate_tick) < M2T(200));
      if (!hold_output) {
        mpc_setpoint_pid.position.x = mpc_setpoint(0);
        mpc_setpoint_pid.position.y = mpc_setpoint(1);
        mpc_setpoint_pid.position.z = mpc_setpoint(2);
        mpc_setpoint_pid.attitude.yaw = mpc_yaw_setpoint_deg;
      } else {
        
        mpc_setpoint_pid.position.x = state->position.x;
        mpc_setpoint_pid.position.y = state->position.y;
        mpc_setpoint_pid.position.z = state->position.z;
        mpc_setpoint_pid.attitude.yaw = state->attitude.yaw;
      }

      controllerPid(control, &mpc_setpoint_pid, sensors, state, tick);
    }
  }

  xSemaphoreGive(dataMutex);

  xSemaphoreGive(runTaskSemaphore);
  
  static uint32_t oot_loop_count = 0;
  oot_loop_count++;
  if (oot_loop_count <= 3) {
    DEBUG_PRINT("OOT loop %lu done\n", oot_loop_count);
  }
}

LOG_GROUP_START(tinympc)

LOG_ADD(LOG_FLOAT, initial_velocity, &init_vel_z)
LOG_ADD(LOG_UINT8, mode, &controller_id)
LOG_ADD(LOG_UINT8, maneuver, &trajectory_id)
LOG_ADD(LOG_UINT8, phase, &flight_phase)
LOG_ADD(LOG_FLOAT, land_zref, &landing_reference_z)
LOG_ADD(LOG_UINT32, step, &benchmark_step)
LOG_ADD(LOG_FLOAT, raya_h, &raya_h)
LOG_ADD(LOG_FLOAT, raya_raw, &raya_raw)
LOG_ADD(LOG_FLOAT, raya_grad, &raya_grad_norm)
LOG_ADD(LOG_FLOAT, raya_margin, &raya_margin_eff)
LOG_ADD(LOG_FLOAT, raya_thresh, &raya_threshold)
LOG_ADD(LOG_UINT32, raya_eval_us, &raya_eval_us)
LOG_ADD(LOG_UINT32, bar_cyc, &raya_barrier_cycles)
LOG_ADD(LOG_UINT32, tanh_cyc, &raya_activation_cycles)
LOG_ADD(LOG_UINT32, rl_cyc, &raya_rl_cycles)
LOG_ADD(LOG_UINT32, cache_cyc, &raya_cache_cycles)
LOG_ADD(LOG_UINT32, solve_cyc, &mpc_solve_cycles)
LOG_ADD(LOG_UINT32, total_cyc, &controller_total_cycles)
LOG_ADD(LOG_UINT32, max_step_cyc, &max_step_cycles)
LOG_ADD(LOG_UINT32, overruns, &deadline_overrun_count)
LOG_ADD(LOG_UINT8, bench_state, &bench_state_index)
LOG_ADD(LOG_UINT8, raya_active, &raya_active)
LOG_ADD(LOG_UINT8, raya_active_count, &raya_active_count)
LOG_ADD(LOG_FLOAT, raya_w, &raya_authority_w)
LOG_ADD(LOG_FLOAT, raya_w_req, &raya_authority_w_requested)
LOG_ADD(LOG_FLOAT, raya_qz, &raya_qz)
LOG_ADD(LOG_FLOAT, raya_qz_req, &raya_qz_requested)
LOG_ADD(LOG_UINT8, raya_w_idx, &raya_w_index)
LOG_ADD(LOG_UINT8, raya_qz_idx, &raya_qz_index)
LOG_ADD(LOG_UINT8, raya_cache, &raya_cache_ok)
LOG_ADD(LOG_FLOAT, no_or_score, &raya_no_oracle_score)
LOG_ADD(LOG_FLOAT, no_or_raw, &raya_no_oracle_score_raw)
LOG_ADD(LOG_UINT8, no_or_act, &raya_no_oracle_active)

LOG_ADD(LOG_UINT16, motCap, &motor_cap_stats.maxAllowed)
LOG_ADD(LOG_INT32, motReq, &motor_cap_stats.latestRequested)
LOG_ADD(LOG_UINT16, motApp, &motor_cap_stats.latestApplied)
LOG_ADD(LOG_INT32, motPeak, &motor_cap_stats.peakRequested)
LOG_ADD(LOG_UINT32, capHits, &motor_cap_stats.capHitCount)
LOG_ADD(LOG_UINT32, capSamples, &motor_cap_stats.sampleCount)
LOG_ADD(LOG_UINT8, faultOn, &motor_fault_active)
LOG_ADD(LOG_UINT32, faultAtMs, &motor_fault_actual_onset_ms)

LOG_ADD(LOG_FLOAT, posX, &tracking_pos_x)
LOG_ADD(LOG_FLOAT, posY, &tracking_pos_y)
LOG_ADD(LOG_FLOAT, posZ, &tracking_pos_z)
LOG_ADD(LOG_FLOAT, velX, &tracking_vel_x)
LOG_ADD(LOG_FLOAT, velY, &tracking_vel_y)
LOG_ADD(LOG_FLOAT, velZ, &tracking_vel_z)
LOG_ADD(LOG_FLOAT, cmdX, &tracking_cmd_x)
LOG_ADD(LOG_FLOAT, cmdY, &tracking_cmd_y)
LOG_ADD(LOG_FLOAT, cmdZ, &tracking_cmd_z)
LOG_ADD(LOG_FLOAT, cmdYaw, &mpc_yaw_setpoint_deg)
LOG_ADD(LOG_FLOAT, previewZ, &tracking_preview_z)
LOG_ADD(LOG_UINT32, mpc_us, &mpc_time_us)
LOG_ADD(LOG_UINT32, total_us, &controller_total_us)

LOG_GROUP_STOP(tinympc)

#define PARAM_TOC_TYPE(TYPE)    static_cast<uint8_t>(((TYPE) <= 0xFF) ? ((TYPE) & 0xFF) : (((TYPE) | PARAM_EXTENDED) & 0xFF))
#define PARAM_TOC_EXT_TYPE(TYPE)    static_cast<uint8_t>((((TYPE) & 0xFF00) >> 8))
#define PARAM_GROUP_ENTRY(TYPE, NAME)    { .type = static_cast<uint8_t>(TYPE), .extended_type = 0, .name = const_cast<char *>(#NAME),      .address = NULL, .callback = NULL, .getter = NULL, },
#define PARAM_VALUE_ENTRY(TYPE, NAME, ADDRESS)    { .type = PARAM_TOC_TYPE(TYPE), .extended_type = PARAM_TOC_EXT_TYPE(TYPE),      .name = const_cast<char *>(#NAME), .address = static_cast<void *>(ADDRESS),      .callback = NULL, .getter = NULL, },

static struct param_s __params_raya[] __attribute__((section(".param.raya"), used)) = {
  PARAM_GROUP_ENTRY(PARAM_GROUP | PARAM_START, raya)
  PARAM_VALUE_ENTRY(PARAM_UINT8, maxIter, &benchmark_max_iter)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, duration, &traj_duration)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, radius, &traj_radius)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, omega, &traj_omega)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, speed, &traj_speed)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, distance, &traj_dist)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, height, &traj_height)
  PARAM_VALUE_ENTRY(PARAM_UINT8, activeH, &raya_active_horizon)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, margin, &raya_margin)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, mScale, &raya_margin_scale)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, hDeadband, &raya_h_deadband)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, actSlack, &raya_act_slack)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, azCoeff, &raya_az_coeff)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, gComp, &raya_gravity_comp)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, failRoll, &raya_fail_roll_deg)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, failPitch, &raya_fail_pitch_deg)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, guard, &raya_structural_guard_relax)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, skipZ, &raya_skip_z)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, skipVz, &raya_skip_vz)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, distBudget, &raya_dist_budget)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, distCoeff, &raya_dist_margin_coeff)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, highCut, &raya_dist_high_cut)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, highBias, &raya_dist_high_bias)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, floorBand, &raya_dist_floor_band)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, floorGain, &raya_dist_floor_gain)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, descBand, &raya_dist_desc_band)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, descGain, &raya_dist_desc_gain)
  PARAM_VALUE_ENTRY(PARAM_FLOAT, hRate, &raya_horizon_margin_rate)
  PARAM_GROUP_ENTRY(PARAM_GROUP | PARAM_STOP, stop_raya)
};

#undef PARAM_VALUE_ENTRY
#undef PARAM_GROUP_ENTRY
#undef PARAM_TOC_EXT_TYPE
#undef PARAM_TOC_TYPE

#ifdef __cplusplus
} 
#endif
