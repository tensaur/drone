#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

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

#define toRad (3.14159265358979323846f / 180.0f)

static inline float action_to_target_rpm(const Params* p, float action) {
    float min_rpm = rpm_min_for_centered_hover(p);
    float u = 0.5f * (clampf(action, -1.0f, 1.0f) + 1.0f); // [-1,1] -> [0,1]
    return min_rpm + u * (p->max_rpm - min_rpm);
}

bool puffer_use_direct_motor_output = false;

static uint64_t controller_tick = 0;
static uint8_t use_rl = 0;

static float observations[23];
static float actions[4];

const uint8_t motors[4] = {MOTOR_M1, MOTOR_M2, MOTOR_M3, MOTOR_M4};

static Weights w;
static LinearContLSTM* puffer_controller;

Drone drone;
Target target;

void controllerOutOfTreeInit() {
    drone.target = &target;
    init_drone(&drone, 0.0f);
    controllerPidInit();

    w.data = (float*)puffer_drone_weights_bin;
    w.size = puffer_drone_weights_bin_size;
    w.idx = 0;

    int logit_sizes[1] = {4};
    puffer_controller = make_linearcontlstm(&w, 1, 23, logit_sizes, 1);

    for (int i = 0; i < 4; i++) {
        drone.state.rpms[i] = 0.0f;
        actions[i] = 0.0f;
    }

    DEBUG_PRINT("Puffer drone controller initialized.\n");
    DEBUG_PRINT("Weights used: %d / %d\n", w.idx, w.size);

    // test forward pass with ones
    // PUFFERDRONE: Test logits: -60.580 -7.712 -1.272 -5.215
    // PUFFERDRONE: Test logits: -60.580 -7.712 -1.272 -5.215
    float test_obs[23];
    for (int i = 0; i < 23; i++)
        test_obs[i] = 1.0f;
    test_obs[22] = 0.0f;
    test_obs[21] = 0.0f;
    test_obs[20] = 0.0f;
    test_obs[19] = 0.0f;
    forward_linearcontlstm(puffer_controller, test_obs);
    DEBUG_PRINT("Test logits: %.3f %.3f %.3f %.3f\n", (double)puffer_controller->actor->output[0],
                (double)puffer_controller->actor->output[1],
                (double)puffer_controller->actor->output[2],
                (double)puffer_controller->actor->output[3]);
}

bool controllerOutOfTreeTest() { return true; }

void controllerOutOfTree(control_t* control, const setpoint_t* setpoint,
                         const sensorData_t* sensors, const state_t* state, const uint32_t tick) {

    if (use_rl) {
        puffer_use_direct_motor_output = true;

        // run policy at 100 Hz
        if (controller_tick % 10 == 0) {
            uint32_t start = xTaskGetTickCount();

            drone.state.pos = (Vec3){state->position.x, state->position.y, state->position.z};
            drone.state.vel = (Vec3){state->velocity.x, state->velocity.y, state->velocity.z};
            drone.state.quat = (Quat){state->attitudeQuaternion.w, state->attitudeQuaternion.x,
                                      state->attitudeQuaternion.y, state->attitudeQuaternion.z};
            drone.state.omega =
                (Vec3){sensors->gyro.x * toRad, sensors->gyro.y * toRad, sensors->gyro.z * toRad};
            target.pos = (Vec3){setpoint->position.x, setpoint->position.y, setpoint->position.z};

            compute_drone_observations(&drone, observations);
            observations[19] = 0.0f;
            observations[20] = 0.0f;
            observations[21] = 0.0f;
            observations[22] = 0.0f;

            forward_linearcontlstm(puffer_controller, observations);
            sample_linearcontlstm(puffer_controller, actions, 1); // 0 -> stochastic

            for (int i = 0; i < 4; i++) {
                float a = actions[i];
                if (!isfinite(a)) a = 0.0f;
                actions[i] = clampf(a, -1.0f, 1.0f);
            }

            uint32_t elapsed = xTaskGetTickCount() - start;
            if (controller_tick % 1000 == 0) {
                DEBUG_PRINT("tick: %d, elapsed: %d ms\n", (int)controller_tick, (int)elapsed);
            }
        }

        // set motors at 1kHz
        for (int i = 0; i < 4; i++) {
            float a = actions[i];
            if (!isfinite(a)) a = 0.0f;
            a = clampf(a, -1.0f, 1.0f);

            float target_rpm = action_to_target_rpm(&drone.params, a);

            float ratio_f = target_rpm / drone.params.max_rpm; // [0,1]
            ratio_f = clampf(ratio_f, 0.0f, 1.0f);

            uint16_t ratio_u16 = (uint16_t)(ratio_f * (float)UINT16_MAX + 0.5f);
            motorsSetRatio(motors[i], ratio_u16);
        }

    } else {
        puffer_use_direct_motor_output = false;
        controllerPid(control, setpoint, sensors, state, tick);
    }

    controller_tick++;
}

PARAM_GROUP_START(pufferdrone)
PARAM_ADD(PARAM_UINT8, use_rl, &use_rl)
PARAM_GROUP_STOP(pufferdrone)
