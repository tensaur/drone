#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

#include "motors.h"
#include "param.h"
#include "stabilizer_types.h"

#define DEBUG_MODULE "PUFFERDRONE"
#include "debug.h"
#include "log.h"

#include "controller.h"
#include "controller_pid.h"

#include "dronelib.h"
#include "puffernet.h"
#include "weights.h"

#define toRad 3.14159265358979323846f / 180.0f

bool puffer_use_direct_motor_output = false;

static uint64_t controller_tick = 0;
static uint8_t use_rl = 0;

static float observations[23];
static float actions[4];

static uint16_t motor_cmd[4];
const uint8_t motors[4] = {MOTOR_M1, MOTOR_M2, MOTOR_M3, MOTOR_M4};

static Weights w;
static LinearContLSTM* puffer_controller;

Drone drone;
Target target;

// We still need an appMain() function, but we will not really use it. Just let it quietly sleep.
void appMain() {
    DEBUG_PRINT("Waiting for activation ...\n");

    while (1) {
        vTaskDelay(M2T(2000));
    }
}

void controllerOutOfTreeInit() {
    // create drone
    drone.target = &target;
    init_drone(&drone, 0.0f);

    // create backup pid controller
    controllerPidInit();

    // create puffer controller
    w.data = puffer_weights;
    w.size = sizeof(puffer_weights) / sizeof(puffer_weights[0]);
    w.idx = 0;

    int logit_sizes[1] = {4};
    // this will not error if you pick the wrong input dim!
    puffer_controller = make_linearcontlstm(&w, 1, 23, logit_sizes, 1);

    DEBUG_PRINT("Puffer drone controller initialized.\n");
    DEBUG_PRINT("Weights used: %d / %d\n", w.idx, w.size);
}

bool controllerOutOfTreeTest() {
    // Always return true
    return true;
}

void controllerOutOfTree(control_t* control, const setpoint_t* setpoint,
                         const sensorData_t* sensors, const state_t* state, const uint32_t tick) {
    
    if (use_rl) {
        puffer_use_direct_motor_output = true;

        // 100 Hz
        if (controller_tick % 10 == 0) {
            drone.state.pos = (Vec3){state->position.x, state->position.y, state->position.z};
            drone.state.vel = (Vec3){state->velocity.x, state->velocity.y, state->velocity.z};
            drone.state.quat = (Quat){state->attitudeQuaternion.w, state->attitudeQuaternion.x, state->attitudeQuaternion.y, state->attitudeQuaternion.z};
            drone.state.omega = (Vec3){sensors->gyro.x * toRad, sensors->gyro.y * toRad, sensors->gyro.z * toRad};
            // rpms are zeroed

            target.pos = (Vec3){setpoint->position.x, setpoint->position.y, setpoint->position.z};

            compute_drone_observations(&drone, observations);
            forward_linearcontlstm(puffer_controller, observations, actions);
        }

        for (int i = 0; i < 4; i++) {
            float scaled = (actions[i] + 1) / 2;
            if (scaled < 0) scaled = 0;
            if (scaled > 1) scaled = 1;
            scaled = 0.15f;

            motor_cmd[i] = scaled * UINT16_MAX;
            motorsSetRatio(motors[i], motor_cmd[i]);
        }

    } else {
        puffer_use_direct_motor_output = false;
        controllerPid(control, setpoint, sensors, state, tick);
    }

    if (controller_tick % 1000 == 0) {
        DEBUG_PRINT("Last setpoint: x disposition/mode %f/%f/%d\n", (double)setpoint->position.x,
                    (double)setpoint->velocity.x, setpoint->mode.x);
        DEBUG_PRINT("Last setpoint: y disposition/mode %f/%f/%d\n", (double)setpoint->position.y,
                    (double)setpoint->velocity.y, setpoint->mode.y);
        DEBUG_PRINT("Last setpoint: z disposition/mode %f/%f/%d\n\n", (double)setpoint->position.z,
                    (double)setpoint->velocity.z, setpoint->mode.z);
    }

    controller_tick++;
}

/*LOG_GROUP_START(motor_commands)
LOG_ADD(LOG_UINT16, m1, &motor_cmd[0])
LOG_ADD(LOG_UINT16, m2, &motor_cmd[1])
LOG_ADD(LOG_UINT16, m3, &motor_cmd[2])
LOG_ADD(LOG_UINT16, m4, &motor_cmd[3])
LOG_GROUP_STOP(motor_commands)*/

PARAM_GROUP_START(pufferdrone)
PARAM_ADD(PARAM_UINT8, use_rl, &use_rl)
PARAM_GROUP_STOP(pufferdrone)
