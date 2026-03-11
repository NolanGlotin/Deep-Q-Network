#ifndef DQN_H
#define DQN_H

#include "types.h"
#include <stdbool.h>

#define CONFIG_DEFAULT (config_t){.episode_nb=100, .learn_step=4, .step_nb=100, .epsilon=1.0f, .epsilon_decay=0.995f, .epsilon_min=0.01, .discount_factor=0.95f, .learning_rate=0.1f, .batch_size=32, .replay_buffer_capacity=1000, .target_update_step=50}

void dqn_train(dnn_t *model, void (*init_env)(vec), bool (*update_env)(vec, int, float *), config_t config);

#endif