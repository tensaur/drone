#include "drone.h"
#include "render.h"

#include <Python.h>

#define Env DroneEnv
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* val = PyDict_GetItemString(kwargs, "task");
    if (val && PyUnicode_Check(val)) {
        env->task_arg = PyUnicode_AsUTF8(val);
    }

    env->num_agents = unpack(kwargs, "num_agents");
    env->max_rings = unpack(kwargs, "max_rings");
    env->env_index = unpack(kwargs, "env_index");
    env->num_envs = unpack(kwargs, "num_envs");
    env->alpha_hover = unpack(kwargs, "alpha_hover");
    env->alpha_shaping = unpack(kwargs, "alpha_shaping");
    env->alpha_omega = unpack(kwargs, "alpha_omega");
    env->alpha_dist = unpack(kwargs, "alpha_dist");
    env->hover_dist = unpack(kwargs, "hover_dist");
    env->hover_omega = unpack(kwargs, "hover_omega");
    env->hover_vel = unpack(kwargs, "hover_vel");
    env->hover_target_dist = unpack(kwargs, "hover_target_dist");
    env->num_obs = unpack(kwargs, "num_obs");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "rings_passed", log->rings_passed);
    assign_to_dict(dict, "ring_collisions", log->ring_collision);
    assign_to_dict(dict, "collisions", log->collisions);
    assign_to_dict(dict, "oob", log->oob);
    assign_to_dict(dict, "timeout", log->timeout);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
