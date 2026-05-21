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

#define TASK_ID_HOVER 0
#define TASK_ID_RACE  1

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    int task_id = (int)dict_get(kwargs, "task")->value;
    if (task_id == TASK_ID_RACE)
        env->task = &TASK_RACE;
    else
        env->task = &TASK_HOVER;

    init(env);
    env->task->init(env, kwargs);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    for (int i = 0; i < log->n; i++)
        dict_set(out, log->keys[i], log->values[i]);
}