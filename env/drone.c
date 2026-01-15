// Standalone C demo for drone environment
// Compile using: ./scripts/build_ocean.sh drone [local|fast]
// Run with: ./drone

#include "drone.h"
#include "puffernet.h"
#include "render.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

double randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

#ifndef LINEAR_DIM
#define LINEAR_DIM 64
#endif

#ifndef LSTM_DIM
#define LSTM_DIM 16
#endif

typedef struct LinearContLSTM LinearContLSTM;
struct LinearContLSTM {
    int num_agents;
    float* obs;
    int num_actions;
    float* log_std;
    Linear* encoder1;
    GELU*  gelu1;
    Linear* encoder2;
    GELU*  gelu2;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
};

LinearContLSTM* make_linearcontlstm(Weights* weights, int num_agents, int input_dim,
                                    int logit_sizes[], int num_actions) {
    LinearContLSTM* net = calloc(1, sizeof(LinearContLSTM));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents * input_dim, sizeof(float));
    net->num_actions = logit_sizes[0];
    
    // Must match export order exactly:
    net->log_std  = get_weights(weights, net->num_actions);                      // 1. decoder_logstd
    net->encoder1 = make_linear(weights, num_agents, input_dim, LINEAR_DIM);     // 2-3. encoder.0
    net->gelu1    = make_gelu(num_agents, LINEAR_DIM);
    net->encoder2 = make_linear(weights, num_agents, LINEAR_DIM, LSTM_DIM);      // 4-5. encoder.2
    net->gelu2    = make_gelu(num_agents, LSTM_DIM);
    net->actor    = make_linear(weights, num_agents, LSTM_DIM, net->num_actions);// 6-7. decoder_mean
    net->value_fn = make_linear(weights, num_agents, LSTM_DIM, 1);               // 8-9. value (FIX: was LSTM_DIM+4)
    net->lstm     = make_lstm(weights, num_agents, LSTM_DIM, LSTM_DIM);          // 10-13. lstm

    return net;
}

void free_linearcontlstm(LinearContLSTM* net) {
    free(net->obs);
    free(net->encoder1);
    free(net->gelu1);
    free(net->encoder2);
    free(net->gelu2);
    free(net->lstm);
    free(net->actor);
    free(net->value_fn);
    free(net);
}

void forward_linearcontlstm(LinearContLSTM* net, float* observations) {
    linear(net->encoder1, observations);
    gelu(net->gelu1, net->encoder1->output);
    linear(net->encoder2, net->gelu1->output);
    gelu(net->gelu2, net->encoder2->output);
    lstm(net->lstm, net->gelu2->output);
    linear(net->actor, net->lstm->state_h);
}

void sample_linearcontlstm(LinearContLSTM* net, float* actions, int deterministic) {
    for (int b = 0; b < net->num_agents; b++) {
        for (int i = 0; i < net->num_actions; i++) {
            int idx = b * net->num_actions + i;
            float mean = net->actor->output[idx];
            if (deterministic) {
                actions[idx] = mean;
            } else {
                float std = expf(net->log_std[i]);
                actions[idx] = (float)randn(mean, std);
            }
        }
    }
}

void generate_dummy_actions(DroneEnv* env) {
    // Generate random floats in [-1, 1] range
    env->actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[3] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneEnv* env;
    LinearContLSTM* net;
    Weights* weights;
} WebRenderArgs;

void emscriptenStep(void* e) {
    WebRenderArgs* args = (WebRenderArgs*)e;
    DroneEnv* env = args->env;
    LinearContLSTM* net = args->net;

    for (int i = 0; i < env->num_agents; i++) {
        int base = i * obs_size;
        env->observations[base + 19] = 0.0f;
        env->observations[base + 20] = 0.0f;
        env->observations[base + 21] = 0.0f;
        env->observations[base + 22] = 0.0f;
    }

    forward_linearcontlstm(net, env->observations);
    sample_linearcontlstm(net, env->actions, 0);
    c_step(env);
    c_render(env);
    return;
}

WebRenderArgs* web_args = NULL;
#endif

int main() {
    srand(time(NULL)); // Seed random number generator

    DroneEnv* env = calloc(1, sizeof(DroneEnv));
    size_t obs_size = 23;
    size_t act_size = 4;

    env->num_agents = 64;
    env->max_rings = 10;
    env->task = HOVER;
    env->hover_target_dist = 0.5f;
    env->hover_dist = 0.05f;
    env->hover_omega = 0.05;
    env->hover_vel = 0.01;
    env->num_obs = obs_size;
    env->num_envs = 1;
    env->env_index = 0;
    init(env);

    env->observations = (float*)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_agents, sizeof(float));

    Weights* weights = load_weights("resources/drone/puffer_drone_weights.bin", 4841);
    int logit_sizes[1] = {4};
    LinearContLSTM* net = make_linearcontlstm(weights, env->num_agents, obs_size, logit_sizes, 1);

    if (!env->observations || !env->actions || !env->rewards) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return 0;
    }

    init(env);
    c_reset(env);

#ifdef __EMSCRIPTEN__
    WebRenderArgs* args = calloc(1, sizeof(WebRenderArgs));
    args->env = env;
    args->net = net;
    args->weights = weights;
    web_args = args;

    emscripten_set_main_loop_arg(emscriptenStep, args, 0, true);
#else
    c_render(env);

    while (!WindowShouldClose()) {
        for (int i = 0; i < env->num_agents; i++) {
            int base = i * obs_size;
            env->observations[base + 19] = 0.0f;
            env->observations[base + 20] = 0.0f;
            env->observations[base + 21] = 0.0f;
            env->observations[base + 22] = 0.0f;
        }
        forward_linearcontlstm(net, env->observations);
        sample_linearcontlstm(net, env->actions, 0);
        c_step(env);
        c_render(env);
    }

    c_close(env);
    free_linearcontlstm(net);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
#endif

    return 0;
}
