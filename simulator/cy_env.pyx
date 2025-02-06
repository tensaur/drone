cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

NUM_ENVS = 0

cdef extern from "env.h":
    int LOG_BUFFER_SIZE
    float GRID_SIZE
    float COL_RAD
    int N_COLS
    int N_RAYS

    ctypedef struct Log:
        float episode_return
        float episode_length
        float score

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Drone:
        float* observations;
        float* actions;
        float* rewards;
        unsigned char* terminals;
        LogBuffer* log_buffer;
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
        # TODO: change to consts
        float rays[6][3];
        float projections[6];
        float colliders[7][4][3];

    void init(Drone* env)
    void c_reset(Drone* env)
    void c_step(Drone* env)

cdef class CyDrone:
    cdef:
        Drone* envs
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, float[:, :] actions,
                 float[:] rewards, unsigned char[:] terminals, int num_envs):
        NUM_ENVS = num_envs;
        self.num_envs = num_envs;

        self.envs = <Drone*> calloc(num_envs, sizeof(Drone));
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE);

        cdef int i;
        for i in range(num_envs):
            self.envs[i] = Drone(
                observations = &observations[i, 0],
                actions = &actions[i, 0],
                rewards = &rewards[i],
                terminals = &terminals[i],
                log_buffer=self.logs,
            );
            init(&self.envs[i])

    def reset(self):
        cdef int i;
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i;
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def close(self):
        free(self.envs)
        free(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log

    # Make properties available which are needed for the visualisation
    if NUM_ENVS == 1:
        @property
        def pos(self):
            return self.envs[0].pos;

        @property
        def move_target(self):
            return self.envs[0].move_target;

        @property
        def look_target(self):
            return self.envs[0].look_target;

        @property
        def yaw(self):
            return self.envs[0].yaw;

        @property
        def near_collision(self):
            return self.envs[0].near_collision;

        @property
        def colliders(self):
            return self.envs[0].colliders;

        @property
        def rays(self):
            return self.envs[0].rays;
