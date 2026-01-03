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

#define SUCCESS_HOVER_STEPS 1

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
    int num_agents;
    Drone* agents;

    int max_rings;
    Target* ring_buffer;

    Client* client;

    int env_index;
    int num_envs;
};

void init(DroneEnv* env) {
    env->task = HOVER;
    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = (Target*)calloc(env->max_rings, sizeof(Target));

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));
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

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
}

void compute_observations(DroneEnv* env) {
    int idx = 0;

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];

        Quat q_inv = quat_inverse(agent->state.quat);
        Vec3 linear_vel_body = quat_rotate(q_inv, agent->state.vel);
        Vec3 to_target_world = sub3(agent->target->pos, agent->state.pos);
        Vec3 to_target = quat_rotate(q_inv, to_target_world);

        // we should probably clamp the overall velocity
        float denom = agent->params.max_vel * 1.7320508f; // sqrt(3)
        env->observations[idx++] = linear_vel_body.x / denom;
        env->observations[idx++] = linear_vel_body.y / denom;
        env->observations[idx++] = linear_vel_body.z / denom;

        env->observations[idx++] = agent->state.omega.x / agent->params.max_omega;
        env->observations[idx++] = agent->state.omega.y / agent->params.max_omega;
        env->observations[idx++] = agent->state.omega.z / agent->params.max_omega;

        env->observations[idx++] = agent->state.quat.w;
        env->observations[idx++] = agent->state.quat.x;
        env->observations[idx++] = agent->state.quat.y;
        env->observations[idx++] = agent->state.quat.z;

        env->observations[idx++] = agent->state.rpms[0] / agent->params.max_rpm;
        env->observations[idx++] = agent->state.rpms[1] / agent->params.max_rpm;
        env->observations[idx++] = agent->state.rpms[2] / agent->params.max_rpm;
        env->observations[idx++] = agent->state.rpms[3] / agent->params.max_rpm;

        // this is body frame so we have to be careful about scaling
        // because distances are relative to the drone orientation
        env->observations[idx++] = to_target.x / MAX_DIST;
        env->observations[idx++] = to_target.y / MAX_DIST;
        env->observations[idx++] = to_target.z / MAX_DIST;

        env->observations[idx++] = clampf(to_target.x, -1.0f, 1.0f);
        env->observations[idx++] = clampf(to_target.y, -1.0f, 1.0f);
        env->observations[idx++] = clampf(to_target.z, -1.0f, 1.0f);

        Vec3 normal_body = quat_rotate(q_inv, agent->target->normal);
        env->observations[idx++] = normal_body.x;
        env->observations[idx++] = normal_body.y;
        env->observations[idx++] = normal_body.z;

        Drone* nearest = nearest_drone(agent, env->agents, env->num_agents);
        if (env->num_agents > 1) {
            Vec3 to_nearest_world = sub3(nearest->state.pos, agent->state.pos);
            Vec3 to_nearest = quat_rotate(q_inv, to_nearest_world);
            env->observations[idx++] = clampf(to_nearest.x, -1.0f, 1.0f);
            env->observations[idx++] = clampf(to_nearest.y, -1.0f, 1.0f);
            env->observations[idx++] = clampf(to_nearest.z, -1.0f, 1.0f);
        } else {
            env->observations[idx++] = MAX_DIST;
            env->observations[idx++] = MAX_DIST;
            env->observations[idx++] = MAX_DIST;
        }
    }
}

void reset_agent(DroneEnv* env, Drone* agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->hover_steps = 0;
    agent->rings_passed = 0;

    agent->buffer = env->ring_buffer;
    agent->buffer_size = env->max_rings;
    agent->buffer_idx = -1;

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
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_rings(env->ring_buffer, env->max_rings);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(env->task, env->agents, i, env->num_agents);
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

        bool oob = agent->state.pos.x < -GRID_X || agent->state.pos.x > GRID_X ||
                   agent->state.pos.y < -GRID_Y || agent->state.pos.y > GRID_Y ||
                   agent->state.pos.z < -GRID_Z || agent->state.pos.z > GRID_Z;
        bool collision = check_collision(agent, env->agents, env->num_agents);
        // bool timeout = (agent->episode_length >= HORIZON);
        bool timeout = false;
        if (collision) agent->collisions += 1.0f;

        float reward = shaping_reward(agent);
        // if (collision) reward -= 0.1f;
        if (oob) reward -= 1.0f;

        bool succeeded = false;
        if (env->task == RACE) {
            int ring_result = check_ring(agent, &env->ring_buffer[agent->buffer_idx]);
            succeeded = (ring_result > 0);
            if (succeeded) env->log.rings_passed += 1.0f;
            if (ring_result < 0) env->log.ring_collision += 1.0f;
        } else {
            bool hovering = check_success(agent);
            if (hovering)
                agent->hover_steps++;
            else
                agent->hover_steps = 0;
            succeeded = (agent->hover_steps >= SUCCESS_HOVER_STEPS);
        }

        if (succeeded) {
            reward += 1.0f;
            agent->hover_steps = 0;
            env->log.score += 1.0f;
            set_target(env->task, env->agents, i, env->num_agents);
        }

        agent->episode_return += reward;
        env->rewards[i] = reward;

        bool failed = oob || timeout;
        env->terminals[i] = failed ? 1 : 0;

        if (failed) {
            add_log(env, i, oob, timeout);
            reset_agent(env, agent, i);
            set_target(env->task, env->agents, i, env->num_agents);
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
