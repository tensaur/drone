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

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;

    int task = (int)dict_get(kwargs, "task")->value;
    if (task == 7) env->task = &TASK_RACE;
    else env->task = &TASK_HOVER;

    LOG_TASK = env->task;

    init(env);
    env->task->init(env, kwargs);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    if (LOG_TASK == NULL) return;

    int n = LOG_TASK->num_log_keys;
    if (n > MAX_TASK_LOG_ENTRIES) n = MAX_TASK_LOG_ENTRIES;
    for (int i = 0; i < n; i++)
        dict_set(out, LOG_TASK->log_keys[i], log->task[i]);
}
