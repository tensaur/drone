#include "drone.h"
#include "render.h"

#define OBS_SIZE DRONE_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"

#include "task_hover.h"
#include "task_race.h"

static const TaskDef* LOG_TASK = NULL;

static void hover_init(DroneEnv* env, Dict* kwargs) {
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist = dict_get(kwargs, "hover_target_dist")->value;
    cfg->hover_dist = dict_get(kwargs, "hover_dist")->value;
    cfg->hover_omega = dict_get(kwargs, "hover_omega")->value;
    cfg->hover_vel = dict_get(kwargs, "hover_vel")->value;
    cfg->alpha_dist = dict_get(kwargs, "alpha_dist")->value;
    cfg->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    cfg->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    cfg->alpha_omega = dict_get(kwargs, "alpha_omega")->value;
    env->task_config = cfg;

    HoverState* state = (HoverState*)calloc(1, sizeof(HoverState));
    state->prev_potential = (float*)calloc(env->num_agents, sizeof(float));
    state->score = (float*)calloc(env->num_agents, sizeof(float));
    state->perf = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_dist = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_vel = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_omega = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;
}

static void race_init(DroneEnv* env, Dict* kwargs) {
    RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
    cfg->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    cfg->ring_reward = dict_get(kwargs, "ring_reward")->value;
    cfg->collision_penalty = dict_get(kwargs, "collision_penalty")->value;
    cfg->time_penalty = dict_get(kwargs, "time_penalty")->value;
    cfg->alpha_dist = dict_get(kwargs, "alpha_dist")->value;
    env->task_config = cfg;
    env->ring_buffer = (Target*)calloc(cfg->max_rings, sizeof(Target));

    RaceState* state = (RaceState*)calloc(1, sizeof(RaceState));
    state->ring_idx = (int*)calloc(env->num_agents, sizeof(int));
    state->rings_passed = (int*)calloc(env->num_agents, sizeof(int));
    state->collisions = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    int task = (int)dict_get(kwargs, "task")->value;
    if (task == 7) {
        env->task = &TASK_RACE;
        race_init(env, kwargs);
    } else {
        env->task = &TASK_HOVER;
        hover_init(env, kwargs);
    }

    LOG_TASK = env->task;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    for (int i = 0; i < LOG_TASK->num_log_keys; i++)
        dict_set(out, LOG_TASK->log_keys[i], log->task[i]);
}