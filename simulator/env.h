#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GRID_SIZE 10.0f
#define DT 0.01

#define MASS      1.0f     // kg
#define IXX       0.005f   // kgm^2
#define IYY       0.005f   // kgm^2
#define IZZ       0.009f   // kgm^2
#define ARM_LEN   0.225f   // m
#define K_THRUST  3e-5f    // thrust coefficient
#define K_ANG_DAMP 0.05f   // tune this
#define K_DRAG    1e-7f    // drag (torque) coefficient
#define GRAVITY   9.81f    // m/s^2

#define MAX_RPM   1000.0f  // rad/s
#define MAX_VEL   5.0f     // m/s
#define MAX_OMEGA 10.0f    // rad/s

// ------------------------------------------------------------
// Logging functions for training loop
// ------------------------------------------------------------
#define LOG_BUFFER_SIZE 4096

typedef struct Log Log;
struct Log {
  float episode_return;
  float episode_length;
  float score;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
  Log *logs;
  int length;
  int idx;
};

LogBuffer *allocate_logbuffer(int size) {
  LogBuffer *logs = (LogBuffer *)calloc(1, sizeof(LogBuffer));
  logs->logs = (Log *)calloc(size, sizeof(Log));
  logs->length = size;
  logs->idx = 0;
  return logs;
}

void free_logbuffer(LogBuffer *buffer) {
  free(buffer->logs);
  free(buffer);
}

void add_log(LogBuffer *logs, Log *log) {
  if (logs->idx == logs->length) {
    return;
  }
  logs->logs[logs->idx] = *log;
  logs->idx += 1;
  // printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length,
  // log->score);
}

Log aggregate_and_clear(LogBuffer *logs) {
  Log log = {0};
  if (logs->idx == 0) {
    return log;
  }
  for (int i = 0; i < logs->idx; i++) {
    log.episode_return += logs->logs[i].episode_return;
    log.episode_length += logs->logs[i].episode_length;
    log.score += logs->logs[i].score;
  }
  log.episode_return /= logs->idx;
  log.episode_length /= logs->idx;
  log.score /= logs->idx;
  logs->idx = 0;
  return log;
}

// ------------------------------------------------------------
// Helper functions for vector math in ℝ³
// ------------------------------------------------------------
static inline float clampf(float v, float min, float max) {
  if (v < min)
    return min;
  if (v > max)
    return max;
  return v;
}

static inline float rndf(float a, float b) {
  return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline int rndi(int a, int b) { return a + rand() % (b - a + 1); }

static inline float dot3(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline void cross3(const float a[3], const float b[3], float out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

static inline float norm3(const float a[3]) { return sqrtf(dot3(a, a)); }

static inline void normalize3(float a[3]) {
  float n = norm3(a);
  if (n > 0) {
    a[0] /= n;
    a[1] /= n;
    a[2] /= n;
  }
}

static inline void add3(const float a[3], const float b[3], float out[3]) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
}

static inline void sub3(const float a[3], const float b[3], float out[3]) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

static inline void scalmul3(const float a[3], float s, float out[3]) {
  out[0] = a[0] * s;
  out[1] = a[1] * s;
  out[2] = a[2] * s;
}

// In-place clamp of a vector
static inline void clamp3(float a[3], float min, float max) {
  a[0] = clampf(a[0], min, max);
  a[1] = clampf(a[1], min, max);
  a[2] = clampf(a[2], min, max);
}

// In-place clamp of a vector
static inline void clamp4(float a[4], float min, float max) {
  a[0] = clampf(a[0], min, max);
  a[1] = clampf(a[1], min, max);
  a[2] = clampf(a[2], min, max);
  a[3] = clampf(a[3], min, max);
}

static inline void quat_mul(const float q1[4], const float q2[4], float out[4]) {
    out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
    out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
    out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
    out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
}

static inline void quat_normalize(float q[4]) {
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 0.0f) {
        q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
    }
}

// Rotate vector v by quaternion q: out = q * [0,v] * q_conj
static inline void quat_rotate(const float q[4], const float v[3], float out[3]) {
    float qv[4] = {0.0f, v[0], v[1], v[2]};
    float tmp[4], res[4];
    quat_mul(q, qv, tmp);
    float q_conj[4] = { q[0], -q[1], -q[2], -q[3] };
    quat_mul(tmp, q_conj, res);
    out[0] = res[1]; out[1] = res[2]; out[2] = res[3];
}

// ------------------------------------------------------------

typedef struct Drone Drone;
struct Drone {
  float *observations;
  float *actions;
  float *rewards;
  unsigned char *terminals;
  LogBuffer *log_buffer;
  Log log;
  int tick;

  int n_targets;
  int moves_left;

  float pos[3];          // global position (X, Y, Z)
  float vel[3];          // linear velocity (U, V, W)
  float quat[4];       // roll (phi), pitch (theta), yaw (psi)
  float omega[3];      // angular velocities (P, Q, R)

  float move_target[3];
  float look_target[3];
  float vec_to_target[3];
};

void init(Drone *env) {
  // logging
  env->tick = 0;
  srand(time(NULL));
}

void allocate(Drone *env) {
  init(env);
  env->observations = (float *)calloc(16, sizeof(float));
  env->actions = (float *)calloc(4, sizeof(float));
  env->rewards = (float *)calloc(1, sizeof(float));
  env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
  env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_allocated(Drone *env) {
  free(env->observations);
  free(env->actions);
  free(env->rewards);
  free(env->terminals);
  free_logbuffer(env->log_buffer);
}

void compute_observations(Drone *env) {
  float scaled_move_target[3];
  scalmul3(env->move_target, 1.0f / GRID_SIZE, scaled_move_target);
  env->observations[0] = scaled_move_target[0];
  env->observations[1] = scaled_move_target[1];
  env->observations[2] = scaled_move_target[2];

  float scaled_pos[3];
  scalmul3(env->pos, 1.0f / GRID_SIZE, scaled_pos);
  env->observations[3] = scaled_pos[0];
  env->observations[4] = scaled_pos[1];
  env->observations[5] = scaled_pos[2];

  env->observations[6] = env->quat[0];
  env->observations[7] = env->quat[1];
  env->observations[8] = env->quat[2];
  env->observations[9] = env->quat[3];

  env->observations[10] = env->vel[0] / MAX_VEL;
  env->observations[11] = env->vel[1] / MAX_VEL;
  env->observations[12] = env->vel[2] / MAX_VEL;

  env->observations[13] = env->omega[0] / MAX_OMEGA;
  env->observations[14] = env->omega[1] / MAX_OMEGA;
  env->observations[15] = env->omega[2] / MAX_OMEGA;
}

void c_reset(Drone *env) {
  env->log = (Log){0};

  // env
  env->n_targets = 5;
  env->moves_left = 1000;

  env->move_target[0] = rndf(-9, 9);
  env->move_target[1] = rndf(-9, 9);
  env->move_target[2] = rndf(-9, 9);

  env->look_target[0] = rndf(-9, 9);
  env->look_target[1] = rndf(-9, 9);
  env->look_target[2] = rndf(-9, 9);

  // state
  env->pos[0] = rndf(-9, 9);
  env->pos[1] = rndf(-9, 9);
  env->pos[2] = rndf(-9, 9);

  env->vel[0] = 0.0f;
  env->vel[1] = 0.0f;
  env->vel[2] = 0.0f;
  
  env->quat[0] = 1.0f;
  env->quat[1] = 0.0f;
  env->quat[2] = 0.0f;
  env->quat[3] = 0.0f;

  env->omega[0] = 0.0f;
  env->omega[1] = 0.0f;
  env->omega[2] = 0.0f;

  compute_observations(env);
}

void c_step(Drone *env) {
  clamp4(env->actions, -1.0f, 1.0f);

  float prev_vec[3];
  sub3(env->pos, env->move_target, prev_vec);
  float prev_dist = norm3(prev_vec);

  env->tick += 1;
  env->log.episode_length += 1;
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  float T[4];
  for (int i = 0; i < 4; i++) {
    float rpm = (env->actions[i] + 1.0f)*0.5f * MAX_RPM;
    T[i] = K_THRUST * rpm * rpm;
  }

  float F_body[3] = {0.0f, 0.0f, T[0]+T[1]+T[2]+T[3]};

  float M[3] = {
        ARM_LEN*(T[1]-T[3]),
        ARM_LEN*(T[2]-T[0]),
        K_DRAG*(T[0]-T[1]+T[2]-T[3])
    };

  for (int i = 0; i < 3; i++) {
    M[i] -= K_ANG_DAMP * env->omega[i];
  }

    float F_world[3], a[3];
    quat_rotate(env->quat, F_body, F_world);
    scalmul3(F_world, 1.0f/MASS, a);
    a[2] -= GRAVITY;

    float I_omega[3] = {IXX*env->omega[0], IYY*env->omega[1], IZZ*env->omega[2]};
    float omega_x_Iw[3]; cross3(env->omega, I_omega, omega_x_Iw);
    float omega_dot[3] = {
        (M[0]-omega_x_Iw[0]) / IXX,
        (M[1]-omega_x_Iw[1]) / IYY,
        (M[2]-omega_x_Iw[2]) / IZZ
    };

    float omega_q[4] = {0.0f, env->omega[0], env->omega[1], env->omega[2]};
    float q_dot[4]; quat_mul(env->quat, omega_q, q_dot);
    for (int i = 0; i < 4; i++) q_dot[i] *= 0.5f;

    for (int i = 0; i < 3; i++) {
        env->pos[i]   += env->vel[i]  * DT;
        env->vel[i]   += a[i]         * DT;
        env->omega[i] += omega_dot[i] * DT;
    }

    clamp3(env->vel,   -MAX_VEL,   MAX_VEL);
    clamp3(env->omega, -MAX_OMEGA, MAX_OMEGA);

    for (int i = 0; i < 4; i++) {
        env->quat[i]  += q_dot[i]    * DT;
    }
    quat_normalize(env->quat);

  bool out_of_bounds = false;
  for (int i = 0; i < 3; i++) {
      if (env->pos[i] < -10.0f || env->pos[i] > 10.0f) {
          out_of_bounds = true;
          break;
      }
  }
  
  if (out_of_bounds) {
      env->rewards[0] -= 1;
      env->log.episode_return -= 1;
      env->terminals[0] = 1;
      add_log(env->log_buffer, &env->log);
      c_reset(env);
      compute_observations(env);
      return;
  }

  sub3(env->pos, env->move_target, env->vec_to_target);

  float curr_dist = norm3(env->vec_to_target);
  float dist = prev_dist - curr_dist;
  env->rewards[0] += dist;
  env->log.episode_return += dist;

  if (norm3(env->vec_to_target) < 1.5) {
    env->rewards[0] += 1;
    env->log.episode_return += 1;
    env->log.score += 1;
    env->n_targets -= 1;

    env->move_target[0] = rndf(-10, 10);
    env->move_target[1] = rndf(-10, 10);
    env->move_target[2] = rndf(-10, 10);
  }

  env->moves_left -= 1;
  if (env->moves_left == 0 || env->n_targets == 0) {
    env->terminals[0] = 1;
    add_log(env->log_buffer, &env->log);
    c_reset(env);
  }

  compute_observations(env);
}
