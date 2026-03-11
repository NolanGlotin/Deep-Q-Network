#include "../include/dqn.h"
#include "../include/dnn.h"
#include "../include/utils.h"
#include <assert.h>

typedef struct {
    vec state;
    int action;
    float reward;
    bool terminate;
    vec next_state;
} transition_t;

typedef struct {
    int capacity;
    int current;
    int size;
    transition_t *transitions;
} buffer_t;

buffer_t *buffer_init(int capacity, int state_size) {
    buffer_t *buffer = (buffer_t *)malloc(sizeof(buffer_t));
    buffer->capacity = capacity;
    buffer->transitions = (transition_t *)malloc(sizeof(transition_t)*capacity);
    buffer->current = 0;
    buffer->size = 0;
    for (int i = 0; i < capacity; i++) {
        buffer->transitions[i].state = (vec)malloc(sizeof(float)*state_size);
        buffer->transitions[i].next_state = (vec)malloc(sizeof(float)*state_size);
    }

    return buffer;
}

void buffer_destroy(buffer_t *buffer) {
    for (int i = 0; i < buffer->capacity; i++) {
        free(buffer->transitions[i].state);
        free(buffer->transitions[i].next_state);
    }
    free(buffer->transitions);
    free(buffer);
}

void buffer_init_transition(buffer_t *buffer, vec state, int state_size) {
    vec_copy(buffer->transitions[buffer->current].state, state, state_size);
}

void buffer_update_transition(buffer_t *buffer, vec next_state, int action, float reward, bool terminate, int state_size) {
    transition_t *transition = &buffer->transitions[buffer->current];

    vec_copy(transition->next_state, next_state, state_size);
    transition->action = action;
    transition->reward = reward;
    transition->terminate = terminate;

    buffer->size = minint(buffer->size + 1, buffer->capacity);
    buffer->current = (buffer->current + 1)%buffer->capacity;
}

void generate_batch(int *batch, bool *used, int size, int nb) {
    assert(size <= nb);
    for (int i = 0; i < nb; i++)
        used[i] = false;
    for (int i = 0; i < size; i++) {
        int r;
        do
            r = randint(0, nb - 1);
        while (used[r]);
        used[r] = true;
        batch[i] = r;
    }
}

void dqn_train(dnn_t *model, void (*init_env)(vec), bool (*update_env)(vec, int, float *), config_t config) {
    init_seed();
    int state_size = model->input_nb;
    vec state = (vec)malloc(sizeof(float)*state_size);
    int *mini_batch = (int *)malloc(sizeof(int)*config.batch_size);
    bool *batch_used = (bool *)malloc(sizeof(bool)*config.replay_buffer_capacity);
    dnn_t *target = dnn_copy(model);
    buffer_t *replay_buffer = buffer_init(config.replay_buffer_capacity, state_size);
    float epsilon = config.epsilon;
    for (int ep = 0; ep < config.episode_nb; ep++) {
        init_env(state);

        for (int step = 0; step < config.step_nb; step++) {
            int action = dnn_predict_epsilon(model, state, epsilon);

            buffer_init_transition(replay_buffer, state, state_size);
            
            float reward;
            bool terminate = update_env(state, action, &reward);

            buffer_update_transition(replay_buffer, state, action, reward, terminate, state_size);

            if (step%config.learn_step == 0 && replay_buffer->size >= config.batch_size) {
                generate_batch(mini_batch, batch_used, config.batch_size, replay_buffer->size);

                for (int i = 0; i < config.batch_size; i++) {
                    transition_t *transition = &replay_buffer->transitions[mini_batch[i]];
                    float y = transition->reward + (transition->terminate ? 0.f : (config.discount_factor*dnn_max_output(target, transition->next_state)));

                    dnn_gradient_descent_filtered(model, transition->state, y, transition->action, config.learning_rate);
                }
            }
            if (step%config.target_update_step == 0)
                dnn_copy_weights(target, model);
        }
        if (epsilon > config.epsilon_min)
            epsilon *= config.epsilon_decay;
    }
    dnn_destroy(target);
    buffer_destroy(replay_buffer);
    free(state);
    free(mini_batch);
    free(batch_used);
}