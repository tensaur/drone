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
static float state_input[16];
static float actions_output[4];
static uint16_t motor_cmd[4];
const uint8_t motors[4] = {MOTOR_M1, MOTOR_M2, MOTOR_M3, MOTOR_M4};

static Weights w;
static LinearContLSTM* puffer_controller;

// We still need an appMain() function, but we will not really use it. Just let it quietly sleep.
void appMain() {
    DEBUG_PRINT("Waiting for activation ...\n");

    while (1) {
        vTaskDelay(M2T(2000));
    }
}

void controllerOutOfTreeInit() {
    // create backup pid controller
    controllerPidInit();

    // create puffer controller
    w.data = puffer_weights;
    w.size = sizeof(puffer_weights) / sizeof(puffer_weights[0]);
    w.idx = 0;

    int logit_sizes[1] = {4};
    // this will not error if you pick the wrong input dim!
    puffer_controller = make_linearcontlstm(&w, 1, 16 /*input dim*/, logit_sizes, 1);

    DEBUG_PRINT("Puffer drone controller initialized.\n");
    DEBUG_PRINT("Weights used: %d / %d\n", w.idx, w.size);
}

bool controllerOutOfTreeTest() {
    // Always return true
    return true;
}

void controllerOutOfTree(control_t* control, const setpoint_t* setpoint,
                         const sensorData_t* sensors, const state_t* state, const uint32_t tick) {
    Quat q = {state->attitudeQuaternion.w, state->attitudeQuaternion.x, state->attitudeQuaternion.y,
              state->attitudeQuaternion.z};
    Vec3 v = {state->velocity.x, state->velocity.y, state->velocity.z};

    Vec3 toTarget = {setpoint->position.x - state->position.x,
                     setpoint->position.y - state->position.y,
                     setpoint->position.z - state->position.z};

    Quat q_inv = quat_inverse(q);
    Vec3 linear_vel_body = quat_rotate(q_inv, v);

    state_input[0] = linear_vel_body.x / BASE_MAX_VEL;
    state_input[1] = linear_vel_body.y / BASE_MAX_VEL;
    state_input[2] = linear_vel_body.z / BASE_MAX_VEL;

    state_input[3] = sensors->gyro.x * toRad / BASE_MAX_OMEGA;
    state_input[4] = sensors->gyro.y * toRad / BASE_MAX_OMEGA;
    state_input[5] = sensors->gyro.z * toRad / BASE_MAX_OMEGA;

    state_input[6] = q.w;
    state_input[7] = q.x;
    state_input[8] = q.y;
    state_input[9] = q.z;

    state_input[10] = toTarget.x / GRID_X;
    state_input[11] = toTarget.y / GRID_Y;
    state_input[12] = toTarget.z / GRID_Z;

    state_input[13] = clampf(toTarget.x, -1.0f, 1.0f);
    state_input[14] = clampf(toTarget.y, -1.0f, 1.0f);
    state_input[15] = clampf(toTarget.z, -1.0f, 1.0f);

    if (use_rl) {
        puffer_use_direct_motor_output = true;

        forward_linearcontlstm(puffer_controller, state_input, actions_output);

        for (int i = 0; i < 4; i++) {
            float scaled = (actions_output[i] + 1) / 2;
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
