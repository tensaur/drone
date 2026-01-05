// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>
#include <stdlib.h>
#include <strings.h>

#include "dronelib.h"

#define HOVER_SPAWN_RADIUS 10.0f

typedef enum {
    IDLE,
    HOVER,
    ORBIT,
    FOLLOW,
    CUBE,
    CONGO,
    FLAG,
    RACE,
    TASK_N // Should always be last
} DroneTask;

static char const* TASK_NAMES[TASK_N] = {"idle", "hover", "orbit", "follow",
                                         "cube", "congo", "flag",  "race"};

DroneTask get_task(char* task_name) {
    for (size_t i = 0; i < TASK_N; i++) {
        if (strcasecmp(TASK_NAMES[i], task_name) == 0) {
            return (DroneTask)i;
        }
    }

    return HOVER;
}

void move_target(Drone* agent) {
    agent->target->pos.x += agent->target->vel.x;
    agent->target->pos.y += agent->target->vel.y;
    agent->target->pos.z += agent->target->vel.z;

    if (agent->target->pos.x < -MARGIN_X || agent->target->pos.x > MARGIN_X) {
        agent->target->vel.x = -agent->target->vel.x;
    }
    if (agent->target->pos.y < -MARGIN_Y || agent->target->pos.y > MARGIN_Y) {
        agent->target->vel.y = -agent->target->vel.y;
    }
    if (agent->target->pos.z < -MARGIN_Z || agent->target->pos.z > MARGIN_Z) {
        agent->target->vel.z = -agent->target->vel.z;
    }
}

void set_target_idle(Drone* agent) {
    agent->target->pos =
        (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), rndf(-MARGIN_Z, MARGIN_Z)};
    agent->target->vel =
        (Vec3){rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET)};
}

void set_target_hover(Drone* agent) {
    agent->target->pos =
        (Vec3){clampf(agent->state.pos.x + rndf(-HOVER_SPAWN_RADIUS, HOVER_SPAWN_RADIUS), -MARGIN_X,
                      MARGIN_X),
               clampf(agent->state.pos.y + rndf(-HOVER_SPAWN_RADIUS, HOVER_SPAWN_RADIUS), -MARGIN_Y,
                      MARGIN_Y),
               clampf(agent->state.pos.z + rndf(-HOVER_SPAWN_RADIUS, HOVER_SPAWN_RADIUS), -MARGIN_Z,
                      MARGIN_Z)};
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_orbit(Drone* agent, int idx, int num_agents) {
    // Fibbonacci sphere algorithm
    float R = 8.0f;
    float phi = M_PI * (sqrt(5.0f) - 1.0f);
    float y = 1.0f - 2 * ((float)idx / (float)num_agents);
    float radius = sqrtf(1.0f - y * y);

    float theta = phi * idx;
    float x = cos(theta) * radius;
    float z = sin(theta) * radius;

    agent->target->pos = (Vec3){R * x, R * z, R * y}; // convert to z up
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_follow(Drone* agents, int idx) {
    Drone* agent = &agents[idx];

    if (idx == 0) {
        set_target_idle(agent);
    } else {
        agent->target->pos = agents[0].target->pos;
        agent->target->vel = agents[0].target->vel;
    }
}

void set_target_cube(Drone* agent, int idx) {
    float z = idx / 16;
    idx = idx % 16;
    float x = (float)(idx % 4);
    float y = (float)(idx / 4);
    agent->target->pos = (Vec3){4 * x - 6, 4 * y - 6, 4 * z - 6};
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_congo(Drone* agents, int idx) {
    if (idx == 0) {
        set_target_idle(&agents[0]);
        return;
    }

    Drone* follow = &agents[idx - 1];
    Drone* lead = &agents[idx];
    lead->target->pos = follow->target->pos;
    lead->target->vel = follow->target->vel;

    // TODO: Slow hack
    for (int i = 0; i < 40; i++) {
        move_target(lead);
    }
}

void set_target_flag(Drone* agent, int idx) {
    float x = (float)(idx % 8);
    float y = (float)(idx / 8);
    x = 2.0f * x - 7;
    y = 5 - 1.5f * y;
    agent->target->pos = (Vec3){0.0f, x, y};
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_race(Drone* agent) { *agent->target = agent->buffer[agent->buffer_idx]; }

void set_target(DroneTask task, Drone* agents, int idx, int num_agents) {
    Drone* agent = &agents[idx];

    if (task == IDLE) set_target_idle(agent);
    else if (task == HOVER) set_target_hover(agent);
    else if (task == ORBIT) set_target_orbit(agent, idx, num_agents);
    else if (task == FOLLOW) set_target_follow(agents, idx);
    else if (task == CUBE) set_target_cube(agent, idx);
    else if (task == CONGO) set_target_congo(agents, idx);
    else if (task == FLAG) set_target_flag(agent, idx);
    else if (task == RACE) set_target_race(agent);
}

int check_success(Drone* agent, float hover_dist, float hover_omega, float hover_vel) {
    Vec3 to_target = sub3(agent->target->pos, agent->state.pos);
    float dist = norm3(to_target);
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);

    if (dist < hover_dist && omega < hover_omega && vel < hover_vel) {
        return 1;
    }
    return 0;
}

float shaping_reward(Drone* agent, float alpha_dist, float alpha_omega) {
    Vec3 to_target = sub3(agent->target->pos, agent->state.pos);
    float dist = norm3(to_target);
    float omega = norm3(agent->state.omega);

    float prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
    float dist_delta = prev_dist - dist;

    return (alpha_dist * dist_delta) - (alpha_omega * omega);
}
