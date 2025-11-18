#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

#include "stabilizer_types.h"
#include "param.h"

#include "puffernet.h"
#include "weights.h"

// Edit the debug name to get nice debug prints
#define DEBUG_MODULE "MYCONTROLLER"
#include "debug.h"


static uint64_t controller_tick = 0; 
static uint8_t use_rl = 0;
static float state_input[29];
static float actions[4];
static setpoint_t last_setpoint;



// We still need an appMain() function, but we will not really use it. Just let it quietly sleep.
void appMain() {
  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) {
    vTaskDelay(M2T(2000));
  }
}

static inline void every_500ms(){
  DEBUG_PRINT("q: %5.2f, %5.2f, %5.2f, %5.2f\n",
            (double)state_input[3], (double)state_input[4],
            (double)state_input[5], (double)state_input[6]);
}

static inline void every_1000ms(){
    DEBUG_PRINT("Last setpoint: x disposition/mode %f/%f/%d\n", (double)last_setpoint.position.x, (double)last_setpoint.velocity.x, last_setpoint.mode.x);
    DEBUG_PRINT("Last setpoint: y disposition/mode %f/%f/%d\n", (double)last_setpoint.position.y, (double)last_setpoint.velocity.y, last_setpoint.mode.y);
    DEBUG_PRINT("Last setpoint: z disposition/mode %f/%f/%d\n", (double)last_setpoint.position.z, (double)last_setpoint.velocity.z, last_setpoint.mode.z);
}


static inline void trigger_every(uint64_t controller_tick){
  if(controller_tick > 3000){
    if(controller_tick % 500 == 0){
      every_500ms();
    }
    if(controller_tick % 1000 == 0){
      every_1000ms();
    }
  }
}


// The new controller goes here --------------------------------------------
// Move the includes to the the top of the file if you want to
#include "controller.h"

// Call the PID controller in this example to make it possible to fly. When you implement you own controller, there is
// no need to include the pid controller.
#include "controller_pid.h"

static Weights w;
static LinearContLSTM* puffer_controller;

void controllerOutOfTreeInit() {
  // Initialize your controller data here...

  // Call the PID controller instead in this example to make it possible to fly
  controllerPidInit();

  DEBUG_PRINT("ur nan is fat brev\n");

  w.data = (float*)puffer_weights;
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
  // Implement your controller here...
  last_setpoint = *setpoint;


  // Call the PID controller instead in this example to make it possible to fly
  state_input[0] = state->position.x;
  if (use_rl){
    if (controller_tick % 200 == 0) { 
      DEBUG_PRINT("using rl controller\n");
      DEBUG_PRINT("weight 0: %d\n", (int)w.data[0]);
      forward_linearcontlstm(puffer_controller, state_input, actions);
      DEBUG_PRINT("weight 0: %f\n", (double)actions[0]);
    }
  }
  controllerPid(control, setpoint, sensors, state, tick);
  trigger_every(controller_tick);
  controller_tick++;
}

PARAM_GROUP_START(pufferdrone)
PARAM_ADD(PARAM_UINT8, use_rl, &use_rl)
PARAM_GROUP_STOP(pufferdrone)