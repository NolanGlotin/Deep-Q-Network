# DQN - Deep Q-Network in C

A small implementation of the Deep Q-Network algorithm in C for reinforcement learning, with an integrated neural network library.

## Features

- Fully configurable multi-layer neural network (DNN)
- DQN algorithm with experience replay
- Epsilon-greedy exploration strategy
- Model save/load functionality

## Project Structure

```
examples/
  ├── target_tracking.c  # Training example
include/
  ├── dnn.h      # Neural network API
  ├── dqn.h      # DQN API
  ├── types.h    # Type definitions
  └── utils.h    # Utilities
src/
  ├── dnn.c      # Network implementation
  └── dqn.c      # DQN implementation
LICENSE         # License file
makefile        # Build configuration
README.md       # This file
```

## Installation

### Build and install the library

```bash
make build
sudo make install
make clean
```

### To uninstall :

```bash
sudo make uninstall
```

## Usage

### Train a model with DQN on your environment:

```c
dqn_train(model, init_env, update_env, config);
```

- `model` : The neural network model to train (see DNN section below)
- `init_env` : Function to initialize the environment state (signature: `void init_env(vec state)`, where `state` is a vector of floats which will be filled with the initial state)
- `update_env` : Function to update the environment based on the action taken (signature: `bool update_env(vec state, int action, float *reward)`, where `state` is the current state vector, `action` is the action taken, and `reward` is a pointer to a float where the reward for this action will be stored. The function should return `true` if the episode has ended, and `false` otherwise)
- `config` : DQN configuration structure (see below)

## DQN Configuration

Customize DQN behavior via the `config_t` structure:

```c
config_t config = {
    .episode_nb = 100,              // Number of training episodes
    .learn_step = 4,                // Learning frequency (every N steps)
    .step_nb = 100,                 // Steps per episode
    .epsilon = 1.0f,                // Initial exploration probability
    .epsilon_decay = 0.995f,        // Epsilon decay factor
    .epsilon_min = 0.01f,           // Minimum epsilon
    .discount_factor = 0.95f,       // Gamma (discount factor)
    .learning_rate = 0.1f,          // Learning rate
    .batch_size = 32,               // Batch size for gradient descent
    .replay_buffer_capacity = 1000, // Replay buffer capacity
    .target_update_step = 50        // Target network update frequency
};
```

## Neural Network Functions

### Creation and destruction

```c
dnn_t *dnn_create(int input_nb);           // Create a new network
void dnn_destroy(dnn_t *dnn);               // Destroy a network
void dnn_add_layer(dnn_t *dnn, int out_nb, int activation_function);
dnn_t *dnn_copy(dnn_t *dnn);               // Copy a network
void dnn_copy_weights(dnn_t *dest, const dnn_t *src);
```

### Save/Load

```c
void dnn_save(dnn_t *dnn, char *file_name);  // Save the model
dnn_t *dnn_load(char *file_name);            // Load a model
```

### Predictions

```c
vec dnn_forward(dnn_t *dnn, vec inputs);          // Forward propagation
int dnn_predict(dnn_t *dnn, vec input);           // Greedy prediction
int dnn_predict_epsilon(dnn_t *dnn, vec input, float epsilon);  // Exploration
float dnn_max_output(dnn_t *dnn, vec input);      // Maximum Q value
```

## Activation Functions

- `ACTIVATION_RELU` : ReLU
- `ACTIVATION_SIGMOID` : Sigmoid
- `ACTIVATION_TANH` : Tanh
- `ACTIVATION_LINEAR` : Linear (for output layer)

## Complete Example

A simple target tracking environment where an agent learns to move left or right to follow a target. The state consists of the agent's position and the target's position, and the reward is based on how close the agent is to the target.
See `target_tracking.c` in `examples` folder:

```c
#include "dqn/dqn.h"
#include "dqn/dnn.h"
#include "dqn/utils.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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
```

## Compilation

```bash
gcc target_tracking.c -o main -ldqn -lm -lblas -fsanitize=address
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Nolan Glotin 2025
