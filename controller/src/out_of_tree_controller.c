#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

#include "stabilizer_types.h"
#include "motors.h"
#include "param.h"

#define DEBUG_MODULE "PUFFERDRONE"
#include "debug.h"

#include "controller.h"
#include "controller_pid.h"

#include "puffernet.h"
#include "weights.h"

bool puffer_use_direct_motor_output = false;

static uint64_t controller_tick = 0;
static uint8_t use_rl = 0;
static float state_input[41];
static float actions_output[4];
static uint16_t motor_cmd[4];
const uint8_t motors[4] = {MOTOR_M1, MOTOR_M2, MOTOR_M3, MOTOR_M4};

static Weights w;
static LinearContLSTM* puffer_controller;

// We still need an appMain() function, but we will not really use it. Just let it quietly sleep.
void appMain() {
  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) {
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
  puffer_controller = make_linearcontlstm(&w, 1, 29/*input dim*/, logit_sizes, 1);

}

bool controllerOutOfTreeTest() {
  // Always return true
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
  state_input[0] = state->position.x;

  if (use_rl){
    puffer_use_direct_motor_output = true;
      
    forward_linearcontlstm(puffer_controller, state_input, actions_output);
      
    for (int i = 0; i < 4; i++) {
      float scaled = (actions_output[i] + 1)/2;
      if (scaled < 0) scaled = 0;
      if (scaled > 1) scaled = 1;
      scaled = 0.0f;

      if (i == 3) {
        scaled = 0.2f;
      }

      motor_cmd[i] = scaled * UINT16_MAX;
      motorsSetRatio(motors[i], motor_cmd[i]);
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