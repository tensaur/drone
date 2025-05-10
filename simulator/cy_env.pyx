cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os
import math

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
        float vel[3];
        float quat[4];
        float omega[3];

        float move_target[3];
        float look_target[3];
        float vec_to_target[3];

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
        def roll(self):
            cdef float q0 = self.envs[0].quat[0]
            cdef float q1 = self.envs[0].quat[1]
            cdef float q2 = self.envs[0].quat[2]
            cdef float q3 = self.envs[0].quat[3]
            return math.atan2(2.0*(q0*q1 + q2*q3),
                              1.0 - 2.0*(q1*q1 + q2*q2))

        @property
        def pitch(self):
            cdef float q0 = self.envs[0].quat[0]
            cdef float q1 = self.envs[0].quat[1]
            cdef float q2 = self.envs[0].quat[2]
            cdef float q3 = self.envs[0].quat[3]
            cdef float t = 2.0*(q0*q2 - q3*q1)
            if t >  1.0:
                t =  1.0
            elif t < -1.0:
                t = -1.0
            return math.asin(t)

        @property
        def yaw(self):
            cdef float q0 = self.envs[0].quat[0]
            cdef float q1 = self.envs[0].quat[1]
            cdef float q2 = self.envs[0].quat[2]
            cdef float q3 = self.envs[0].quat[3]
            return math.atan2(2.0*(q0*q3 + q1*q2),
                              1.0 - 2.0*(q2*q2 + q3*q3))