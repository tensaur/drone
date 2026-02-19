// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "dronelib.h"
#include "tasks.h"

#define HORIZON 1024

typedef struct Client Client;
typedef struct DroneEnv DroneEnv;

struct DroneEnv {
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;

    Log log;
    int tick;

    DroneTask task;
    char* task_arg;
    int num_agents;
    Drone* agents;

    int max_rings;
    Target* ring_buffer;

    Client* client;

    int env_index;
    int num_envs;

    // reward scaling
    float alpha_dist;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;

    // hover task parameters
    float hover_target_dist;
    float hover_dist;
    float hover_omega;
    float hover_vel;

    int num_obs;
};

void init(DroneEnv* env) {
    if (env->task_arg) {
        env->task = get_task(env->task_arg);
    } else {
        env->task = HOVER;
    }

    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = (Target*)calloc(env->max_rings, sizeof(Target));

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));
        env->agents[i].buffer_idx = 0;
    }

    env->log = (Log){0};
    env->tick = (HORIZON * env->env_index) / env->num_envs;
}

void add_log(DroneEnv* env, int idx, bool oob, bool timeout) {
    Drone* agent = &env->agents[idx];

    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collisions += agent->collisions;

    if (oob) env->log.oob += 1.0f;
    if (timeout) env->log.timeout += 1.0f;

    env->log.score += agent->hover_score;
    env->log.perf += agent->hover_ema;
    env->log.rings_passed += agent->rings_passed;

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->rings_passed = 0.0f;
}

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        compute_drone_observations(&env->agents[i], env->observations + i*env->num_obs);
    }
}

void reset_agent(DroneEnv* env, Drone* agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->rings_passed = 0;
    agent->score = 0.0f;
    agent->hover_score = 0.0f;
    agent->hover_ema = 0.0f;

    agent->buffer = env->ring_buffer;
    agent->buffer_size = env->max_rings;

    init_drone(agent, 0.05f);

    agent->state.pos =
        (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), rndf(-MARGIN_Z, MARGIN_Z)};

    if (env->task == RACE) {
        while (norm3(sub3(agent->state.pos, env->ring_buffer[0].pos)) < 2.0f * RING_RADIUS) {
            agent->state.pos = (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y),
                                      rndf(-MARGIN_Z, MARGIN_Z)};
        }
    }

    agent->prev_pos = agent->state.pos;
    agent->prev_potential = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_rings(env->ring_buffer, env->max_rings);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(env->task, env->agents, i, env->num_agents, env->hover_target_dist);
    }

    compute_observations(env);
}

void c_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];

        agent->prev_pos = agent->state.pos;
        move_drone(agent, &env->actions[4 * i]);
        agent->episode_length++;

        bool oob = norm3(sub3(agent->target->pos, agent->state.pos)) > (env->hover_target_dist + 1.0f);
        bool timeout = (agent->episode_length >= HORIZON);

        float curr = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
        float prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
        float curr_dist = norm3(sub3(agent->target->pos, agent->state.pos));
        float omega = norm3(agent->state.omega);

        float reward = env->alpha_dist * (prev_dist - curr_dist)
                     + env->alpha_hover * curr
                     + env->alpha_shaping * (curr - agent->prev_potential)
                     - env->alpha_omega * omega;
        
        agent->prev_potential = curr;

        float h = check_hover(agent, env->hover_dist, env->hover_omega, env->hover_vel);
        agent->hover_score += h;
        agent->hover_ema = (1.0f - 0.02f) * agent->hover_ema + 0.02f * h;
        agent->episode_return += reward;
        env->rewards[i] = reward;

        bool reset = oob || timeout;
        env->terminals[i] = reset ? 1 : 0;

        if (reset) {
            add_log(env, i, oob, timeout);
            reset_agent(env, agent, i);
            set_target(env->task, env->agents, i, env->num_agents, env->hover_target_dist);
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        free(env->agents[i].target);
    }

    free(env->agents);
    free(env->ring_buffer);

    if (env->client != NULL) {
        c_close_client(env->client);
    }
}
