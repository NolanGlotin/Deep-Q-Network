#include "dqn/dqn.h"
#include "dqn/dnn.h"
#include "dqn/utils.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


// --- Target tracking : a simple test of DQN on a custom environment ---
// The agent must learn to track a target on a 1D line.
// To run : gcc target_tracking.c -o main -ldqn -lm -lblas -fsanitize=address && ./main


// Test environment
void init_env(vec state) {
    state[0] = randf() * 2.0f - 1.0f;
    state[1] = randf() * 2.0f - 1.0f;
}

bool update_env(vec state, int action, float *reward) {
    float step_size = 0.1f;
    if (action == 0) state[0] -= step_size; // Left
    if (action == 1) state[0] += step_size; // Right
    
    // Distance
    float dist = fabsf(state[0] - state[1]);
    
    if (dist < 0.1f) {
        *reward = 10.0f;  // Success
        return true;
    } else {
        *reward = -0.1f;  // Time penalty
        if (state[0] < -1.5f || state[0] > 1.5f) return true;
        return false;
    }
}

int main() {
    // 1. Create model neural network
    dnn_t *model = dnn_create(2);
    dnn_add_layer(model, 24, ACTIVATION_RELU);
    dnn_add_layer(model, 24, ACTIVATION_RELU);
    dnn_add_layer(model, 3, ACTIVATION_LINEAR);

    // 2. Configure training
    config_t config = {
        .episode_nb = 500,
        .step_nb = 100,
        .learning_rate = 0.001f,
        .discount_factor = 0.95f,
        .replay_buffer_capacity = 10000,
        .batch_size = 32,
        .epsilon = 1.0f,
        .epsilon_decay = 0.995f,
        .epsilon_min = 0.01f,
        .learn_step = 4,
        .target_update_step = 100
    };

    // 3. Train model
    printf("Training in progress...\n");
    dqn_train(model, init_env, update_env, config);
    printf("Training done.\n");

    // 4. Test model
    vec state = (vec)malloc(sizeof(float) * 2);
    for (float pos = -0.5f; pos < 1.0f; pos += 0.5f) {
        state[0] = pos;
        state[1] = -pos;
        int action = dnn_predict(model, state);
        printf("Test : Agent at %.1f, Target at %.1f. Action : %d (0: Left, 1: Right, 2:Stay)\n", state[0], state[1], action);
    }

    dnn_destroy(model);
    free(state);
    return 0;
}