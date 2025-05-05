#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GRID_SIZE 10.0f

// Physical constants
#define K 0.00000298f
#define B 0.000000114f
#define L 0.225f

// Intertial constants
#define IX 0.004856f
#define IY 0.004856f
#define IZ 0.008801f
#define IR 0.00003357f

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
  float next_pos[3];     // global position of next step
  float vel[3];          // linear velocity (U, V, W)
  float angles[3];       // roll (phi), pitch (theta), yaw (psi)
  float angular_vel[3];  // angular velocities (P, Q, R)
  float rotor_speeds[4]; // speed of quadcopter rotors (omega_i)

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
  env->observations = (float *)calloc(18, sizeof(float));
  env->actions = (float *)calloc(3, sizeof(float));
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

  env->observations[6] = sin(env->angles[0]);
  env->observations[7] = cos(env->angles[0]);
  env->observations[8] = sin(env->angles[1]);
  env->observations[9] = cos(env->angles[1]);
  env->observations[10] = sin(env->angles[2]);
  env->observations[11] = cos(env->angles[2]);

  env->observations[12] = env->vel[0];
  env->observations[13] = env->vel[1];
  env->observations[14] = env->vel[2];

  env->observations[15] = env->angular_vel[0];
  env->observations[16] = env->angular_vel[1];
  env->observations[17] = env->angular_vel[2];
}

void c_reset(Drone *env) {
  env->log = (Log){0};

  env->n_targets = 5;
  env->moves_left = 1500;

  env->pos[0] = rndf(-10, 10);
  env->pos[1] = rndf(-10, 10);
  env->pos[2] = rndf(-10, 10);

  env->move_target[0] = rndf(-10, 10);
  env->move_target[1] = rndf(-10, 10);
  env->move_target[2] = rndf(-10, 10);

  env->look_target[0] = rndf(-10, 10);
  env->look_target[1] = rndf(-10, 10);
  env->look_target[2] = rndf(-10, 10);

  env->angles[0] = 0;
  env->angles[1] = 0;
  env->angles[2] = 0;

  env->vel[0] = 0;
  env->vel[1] = 0;
  env->vel[2] = 0;

  env->angular_vel[0] = 0;
  env->angular_vel[1] = 0;
  env->angular_vel[2] = 0;

  compute_observations(env);
}

void calc_to_nearest_collider(Drone *env);

void c_step(Drone *env) {
  clamp3(env->actions, -1, 1);

  env->tick += 1;
  env->log.episode_length += 1;
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  clamp3(env->next_pos, -10, 10);

  sub3(env->next_pos, env->move_target, env->vec_to_target);
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
