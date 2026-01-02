#include "drone.h"
#include "render.h"

#define Env DroneEnv
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = unpack(kwargs, "num_agents");
    env->max_rings = unpack(kwargs, "max_rings");
    env->env_index = unpack(kwargs, "env_index");
    env->num_envs = unpack(kwargs, "num_envs");
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
