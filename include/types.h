#ifndef TYPES_H
#define TYPES_H

typedef float * vec;
typedef float * mat;

typedef struct {
    int in_nb;
    int out_nb;
    int activation_func;
    mat weights;
    vec activation;
    vec bias;
    vec delta;
} layer_t;

typedef struct {
    int input_nb;
    int output_nb;
    int layer_nb;
    layer_t **layers;
} dnn_t;

typedef struct {
    int episode_nb;
    int learn_step;
    int step_nb;
    float epsilon;
    float epsilon_decay;
    float epsilon_min;
    float discount_factor;
    float learning_rate;
    int batch_size;
    int replay_buffer_capacity;
    int target_update_step;
} config_t;

#endif