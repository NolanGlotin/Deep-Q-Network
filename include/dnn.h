#ifndef DNN_H
#define DNN_H

#include "types.h"

enum activation_functions {
    ACTIVATION_RELU = 0,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_LINEAR
};

// creation/destruction
dnn_t *dnn_create(int input_nb);
void dnn_destroy(dnn_t *dnn);
void dnn_add_layer(dnn_t *dnn, int out_nb, int activation_function);
dnn_t *dnn_copy(dnn_t *dnn);
void dnn_copy_weights(dnn_t *dest, const dnn_t *src);

// i/o
void dnn_save(dnn_t *dnn, char *file_name);
dnn_t *dnn_load(char *file_name);

// forward propagation
vec dnn_forward(dnn_t *dnn, vec inputs);
int dnn_predict(dnn_t *dnn, vec input);
int dnn_predict_epsilon(dnn_t *dnn, vec input, float epsilon);
float dnn_max_output(dnn_t *dnn, vec input);

// backward propagation
void dnn_gradient_descent_filtered(dnn_t *dnn, vec input, float target, int filter, float learning_rate);

#endif