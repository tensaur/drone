#pragma once

#include "drone.h"

typedef struct {
    float target_dist;
    float hover_dist;
    float hover_omega;
    float hover_vel;
    float alpha_dist;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;
} HoverConfig;

typedef struct {
    float* prev_potential;
} HoverState;

static inline float hover_potential(float dist, float vel, float omega, HoverConfig* cfg) {
    float d = 1.0f / (1.0f + dist / cfg->hover_dist);
    float v = 1.0f / (1.0f + vel / cfg->hover_vel);
    float w = 1.0f / (1.0f + omega / cfg->hover_omega);
    return d * (0.7f + 0.15f * v + 0.15f * w);
}

static void hover_init(DroneEnv* env, void* kwargs_) {
    Dict* kwargs = (Dict*)kwargs_;
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist   = dict_get(kwargs, "hover_target_dist")->value;
    cfg->hover_dist    = dict_get(kwargs, "hover_dist")->value;
    cfg->hover_omega   = dict_get(kwargs, "hover_omega")->value;
    cfg->hover_vel     = dict_get(kwargs, "hover_vel")->value;
    cfg->alpha_dist    = dict_get(kwargs, "alpha_dist")->value;
    cfg->alpha_hover   = dict_get(kwargs, "alpha_hover")->value;
    cfg->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    cfg->alpha_omega   = dict_get(kwargs, "alpha_omega")->value;
    env->task_config = cfg;

    HoverState* state = (HoverState*)calloc(1, sizeof(HoverState));
    state->prev_potential = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;

    log_register(&env->log, "hover_score"); // 0
    log_register(&env->log, "oob");         // 1
    log_register(&env->log, "timeout");     // 2
}

static void hover_free(DroneEnv* env) {
    free(((HoverState*)env->task_state)->prev_potential);
    free(env->task_state);
    free(env->task_config);
}

static void hover_reset(DroneEnv* env, Drone* agent, int idx) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    HoverState* state = (HoverState*)env->task_state;

    agent->state.pos = random_pos(&env->rng);
    agent->target->pos = random_pos(&env->rng);

    state->prev_potential[idx] = hover_potential(
        norm3(sub3(agent->target->pos, agent->state.pos)),
        norm3(agent->state.vel), norm3(agent->state.omega), cfg);
}

static float hover_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    HoverState* state = (HoverState*)env->task_state;

    float curr = hover_potential(cache->dist, cache->vel, cache->omega, cfg);
    float reward = cfg->alpha_dist * (cache->prev_dist - cache->dist)
                 + cfg->alpha_hover * curr
                 + cfg->alpha_shaping * (curr - state->prev_potential[idx])
                 - cfg->alpha_omega * cache->omega;
    state->prev_potential[idx] = curr;
    return reward;
}

static bool hover_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    return cache->dist > (cfg->target_dist + 1.0f)
        || agent->episode_length >= HORIZON;
}

static void hover_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    HoverConfig* cfg = (HoverConfig*)env->task_config;
    log_add(log, 0, 1.0f / (1.0f + cache->dist));
    log_add(log, 1, cache->dist > (cfg->target_dist + 1.0f) ? 1.0f : 0.0f);
    log_add(log, 2, agent->episode_length >= HORIZON ? 1.0f : 0.0f);
}

static const TaskDef TASK_HOVER = {
    .name      = "hover",
    .init      = hover_init,
    .free      = hover_free,
    .env_reset = NULL,
    .reset     = hover_reset,
    .reward    = hover_reward,
    .done      = hover_done,
    .log       = hover_log,
    .render    = NULL,
};