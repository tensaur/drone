#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRID_SIZE 10.0f
#define COL_RAD 0.55f
#define N_COLS 8
#define N_RAYS 6

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
  float pos[3];
  float next_pos[3];
  float vel[3];
  float yaw;
  float move_target[3];
  float look_target[3];
  float vec_to_target[3];

  float closest_collider_dist;
  float near_collision[3];

  float rays[N_RAYS][3];
  float projections[N_RAYS];

  float colliders[N_COLS][4][3];
};

void init(Drone *env) {
  // logging
  env->tick = 0;

  // precompute
  for (int i = 0; i < N_RAYS; i++) {
    for (int j = 0; j < 3; j++) {
      env->rays[i][j] = 0;
    }
  }

  for (int i = 0; i < N_RAYS / 2; i++) {
    env->rays[i][i] = -1;
    env->rays[N_RAYS - 1 - i][i] = 1;
  }

  float cols[N_COLS][4][3] = {
      {{-10, -10, -10}, {10, -10, -10}, {10, -10, 10}, {-10, -10, 10}},
      {{10, -10, -10}, {10, 10, -10}, {10, 10, 10}, {10, -10, 10}},
      {{10, 10, -10}, {10, 10, 10}, {-10, 10, 10}, {-10, 10, -10}},
      {{-10, 10, -10}, {-10, 10, 10}, {-10, -10, 10}, {-10, -10, -10}},
      {{-10, -10, 10}, {10, -10, 10}, {10, 10, 10}, {-10, 10, 10}},
      {{-10, -10, -10}, {10, -10, -10}, {10, 10, -10}, {-10, 10, -10}},
      {{0, 0, -5}, {0, 10, -5}, {0, 10, 10}, {0, 0, 10}},
      {{0, 0, 5}, {0, -10, 5}, {0, -10, -10}, {0, 0, -10}}};

  memcpy(env->colliders, cols, sizeof(cols));
}

void allocate(Drone *env) {
  init(env);
  env->observations = (float *)calloc(8 + N_RAYS, sizeof(float));
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

  env->observations[6] = sin(env->yaw);
  env->observations[7] = cos(env->yaw);

  for (int i = 0; i < N_RAYS; i++) {
    env->observations[8 + i] = env->projections[i];
  }
}

void c_reset(Drone *env) {
  env->log = (Log){0};

  env->n_targets = 5;
  env->moves_left = 1500;
  env->yaw = 0;

  env->pos[0] = rndf(-10, 10);
  env->pos[1] = rndf(-10, 10);
  env->pos[2] = rndf(-10, 10);

  env->move_target[0] = rndf(-10, 10);
  env->move_target[1] = rndf(-10, 10);
  env->move_target[2] = rndf(-10, 10);

  env->look_target[0] = rndf(-10, 10);
  env->look_target[1] = rndf(-10, 10);
  env->look_target[2] = rndf(-10, 10);

  env->closest_collider_dist = MAXFLOAT;

  compute_observations(env);
}

void c_step(Drone *env) {
  clamp3(env->actions, -1, 1);

  env->closest_collider_dist = MAXFLOAT;
  for (int r = 0; r < N_RAYS; r++) {
    env->projections[r] = 0;
  }

  env->tick += 1;
  env->log.episode_length += 1;
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  env->vel[0] =
      env->actions[0] * cos(env->yaw) - env->actions[1] * sin(env->yaw);
  env->vel[1] =
      env->actions[0] * sin(env->yaw) + env->actions[1] * cos(env->yaw);
  env->vel[2] = env->actions[2];

  add3(env->pos, env->vel, env->next_pos);
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

  for (int i = 0; i < N_COLS; i++) {
    float plane_vecs[3][3];
    sub3(env->colliders[i][1], env->colliders[i][0], plane_vecs[0]);
    sub3(env->colliders[i][3], env->colliders[i][0], plane_vecs[1]);
    sub3(env->colliders[i][2], env->colliders[i][0], plane_vecs[2]);

    float collider_norm[3];
    cross3(plane_vecs[0], plane_vecs[2], collider_norm);
    normalize3(collider_norm);

    float d = -dot3(env->colliders[i][0], collider_norm);
    float dist_to_plane = fabsf(collider_norm[0] * env->next_pos[0] +
                                collider_norm[1] * env->next_pos[1] +
                                collider_norm[2] * env->next_pos[2] + d) /
                          norm3(collider_norm);

    float mu = -(d + dot3(collider_norm, env->next_pos)) /
               dot3(collider_norm, collider_norm);

    float point_on_plane[3];
    point_on_plane[0] = env->next_pos[0] + mu * collider_norm[0];
    point_on_plane[1] = env->next_pos[1] + mu * collider_norm[1];
    point_on_plane[2] = env->next_pos[2] + mu * collider_norm[2];

    float p[3];
    sub3(env->next_pos, env->colliders[i][0], p);

    float close_point[3];
    float close_dist;

    if ((0 < dot3(p, plane_vecs[0]) &&
         dot3(p, plane_vecs[0]) < dot3(plane_vecs[0], plane_vecs[0])) &&
        (0 < dot3(p, plane_vecs[1]) &&
         dot3(p, plane_vecs[1]) < dot3(plane_vecs[1], plane_vecs[1]))) {
      if (dist_to_plane < env->closest_collider_dist) {
        env->closest_collider_dist = dist_to_plane;
        close_dist = dist_to_plane;
        env->near_collision[0] = point_on_plane[0];
        env->near_collision[1] = point_on_plane[1];
        env->near_collision[2] = point_on_plane[2];
        close_point[0] = point_on_plane[0];
        close_point[1] = point_on_plane[1];
        close_point[2] = point_on_plane[2];
      }
    } else {
      float dist_to_corners[4];
      for (int cidx = 0; cidx < 4; cidx++) {
        float v[3];
        sub3(env->colliders[i][cidx], point_on_plane, v);
        dist_to_corners[cidx] = norm3(v);
      }

      float closest_corners[2][3];
      int min_idx = 0, min2_idx = 1;

      if (dist_to_corners[1] < dist_to_corners[0]) {
        min_idx = 1;
        min2_idx = 0;
      }

      for (int k = 2; k < 4; k++) {
        if (dist_to_corners[k] < dist_to_corners[min_idx]) {
          min2_idx = min_idx;
          min_idx = k;
        } else if (dist_to_corners[k] < dist_to_corners[min2_idx]) {
          min2_idx = k;
        }
      }

      closest_corners[0][0] = env->colliders[i][min_idx][0];
      closest_corners[0][1] = env->colliders[i][min_idx][1];
      closest_corners[0][2] = env->colliders[i][min_idx][2];

      closest_corners[1][0] = env->colliders[i][min2_idx][0];
      closest_corners[1][1] = env->colliders[i][min2_idx][1];
      closest_corners[1][2] = env->colliders[i][min2_idx][2];

      float closest_edge[3];
      sub3(closest_corners[1], closest_corners[0], closest_edge);

      float plane_to_corner[3];
      sub3(closest_corners[0], point_on_plane, plane_to_corner);

      float omega = (-dot3(closest_edge, plane_to_corner)) /
                    dot3(closest_edge, closest_edge);

      if (omega < 0 || omega > 1) {
        omega = 0;
      }

      close_point[0] = closest_corners[0][0] + omega * closest_edge[0];
      close_point[1] = closest_corners[0][1] + omega * closest_edge[1];
      close_point[2] = closest_corners[0][2] + omega * closest_edge[2];

      float vec_to_point[3];
      sub3(close_point, env->next_pos, vec_to_point);
      close_dist = norm3(vec_to_point);

      if (close_dist < env->closest_collider_dist) {
        env->closest_collider_dist = close_dist;
        env->near_collision[0] = close_point[0];
        env->near_collision[1] = close_point[1];
        env->near_collision[2] = close_point[2];
      }
    }

    for (int r = 0; r < N_RAYS; r++) {
      float dir_unit[3];
      sub3(close_point, env->next_pos, dir_unit);
      normalize3(dir_unit);

      float projection = clampf(dot3(env->rays[r], dir_unit), 0, 1);
      if ((projection > env->projections[r]) && close_dist < 1) {
        env->projections[r] = projection * (1 - close_dist);
      }
    }
  }

  if (env->closest_collider_dist < COL_RAD) {
    env->rewards[0] -= 0.25;
    env->log.episode_return -= 0.25;
  } else if (COL_RAD < env->closest_collider_dist &&
             env->closest_collider_dist < COL_RAD + 0.2) {
    env->rewards[0] -= 0.1 + ((env->closest_collider_dist - COL_RAD) / 2);
    env->log.episode_return -=
        0.1 + ((env->closest_collider_dist - COL_RAD) / 2);
  }

  env->moves_left -= 1;
  if (env->moves_left == 0 || env->n_targets == 0) {
    env->terminals[0] = 1;
    add_log(env->log_buffer, &env->log);
    c_reset(env);
  }

  env->pos[0] = env->next_pos[0];
  env->pos[1] = env->next_pos[1];
  env->pos[2] = env->next_pos[2];

  env->yaw = rndf(0, 2 * M_PI);

  compute_observations(env);
}
