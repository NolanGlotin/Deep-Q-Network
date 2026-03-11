#include "../include/dnn.h"
#include "../include/utils.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

float activation_function(int function_id, float x) {
    switch (function_id) {
        case ACTIVATION_SIGMOID:    return 1.f/(1.f + expf(-x));
        case ACTIVATION_RELU:       return maxf(x, 0.f);
        case ACTIVATION_TANH:       return tanhf(x);
        case ACTIVATION_LINEAR:     return x;
        default:                    return 0.f;
    }
}

void activation_function_vec(int function_id, vec x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = activation_function(function_id, x[i]);
}

float activation_function_derivative(int function_id, float a) {
    switch (function_id) {
        case ACTIVATION_SIGMOID:    return a*(1- a);
        case ACTIVATION_RELU:       return a > 0.f ? 1.f : 0.f;
        case ACTIVATION_TANH:       return 1.f - a*a;
        case ACTIVATION_LINEAR:     return 1.f;
        default:                    return 0.f;
    } 
}

void activation_function_derivative_vec(int function_id, vec x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = activation_function_derivative(function_id, x[i]);
}

layer_t *output_layer(dnn_t *dnn) {
    return dnn->layers[dnn->layer_nb - 1];
}

dnn_t *dnn_create(int input_nb) {
    dnn_t *dnn = (dnn_t *)malloc(sizeof(dnn_t));
    dnn->input_nb = input_nb;
    dnn->output_nb = input_nb;
    dnn->layer_nb = 0;
    dnn->layers = NULL;
    return dnn;
}

void dnn_add_layer(dnn_t *dnn, int out_nb, int activation_function) {
    layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
    assert(dnn != NULL);
    layer->out_nb = out_nb;
    layer->in_nb = dnn->layer_nb == 0 ? dnn->input_nb : output_layer(dnn)->out_nb;
    layer->activation_func = activation_function;
    layer->activation = (vec)malloc(sizeof(float)*out_nb);
    layer->bias = (vec)malloc(sizeof(float)*out_nb);
    layer->weights = (vec)malloc(sizeof(float)*out_nb*layer->in_nb);
    layer->delta = (vec)malloc(sizeof(float)*out_nb);
    for (int i = 0; i < out_nb; i++) {
        layer->activation[i] = rand_normal(0.1f);
        layer->bias[i] = rand_normal(0.1f);
        layer->delta[i] = 0.f;
        for (int j = 0; j < layer->in_nb; j++)
            layer->weights[i*layer->in_nb + j] = rand_normal(0.1f / sqrtf((float)layer->in_nb));
    }
    dnn->layer_nb += 1;
    dnn->layers = realloc(dnn->layers, sizeof(layer_t *)*dnn->layer_nb);
    dnn->layers[dnn->layer_nb - 1] = layer;
    dnn->output_nb = layer->out_nb;
}

void dnn_destroy(dnn_t *dnn) {
    for (int i = 0; i < dnn->layer_nb; i++) {
        free(dnn->layers[i]->weights);
        free(dnn->layers[i]->bias);
        free(dnn->layers[i]->activation);
        free(dnn->layers[i]->delta);
        free(dnn->layers[i]);
    }
    free(dnn->layers);
    free(dnn);
}

vec dnn_forward(dnn_t *dnn, vec input) {
    for (int i = 0; i < dnn->layer_nb; i++) {
        layer_t *layer = dnn->layers[i];
        mat_vec_affine(layer->activation, layer->weights, i == 0 ? input : dnn->layers[i - 1]->activation, layer->bias, layer->out_nb, layer->in_nb);
        activation_function_vec(layer->activation_func, layer->activation, layer->out_nb);
    }
    return output_layer(dnn)->activation;
}

void dnn_save(dnn_t *dnn, char *file_name) {
    FILE *file = fopen(file_name, "w");

    fprintf(file, "%d\n", dnn->layer_nb);
    for (int i = 0; i < dnn->layer_nb; i++) {
        layer_t *layer = dnn->layers[i];
        fprintf(file, "%d %d %d ", layer->in_nb, layer->out_nb, layer->activation_func);
        for (int j = 0; j < layer->in_nb*layer->out_nb; j++)
            fprintf(file, "%f ", layer->weights[j]);
        for (int j = 0; j < layer->out_nb; j++)
            fprintf(file, "%f ", layer->bias[j]);
        fprintf(file, "\n");
    }

    fclose(file);
}

void xfscanf(FILE *file, const char *format, void *x) {
    if (fscanf(file, format, x) != 1) {
        fprintf(stderr, "Failed to parse data.\n");
        exit(EXIT_FAILURE);
    }
}

dnn_t *dnn_load(char *file_name) {
    FILE *file = fopen(file_name, "r");

    dnn_t *dnn = dnn_create(0);
    int layer_nb;
    xfscanf(file, "%d", &layer_nb);

    for (int i = 0; i < layer_nb; i++) {
        int in, out, activation_func;
        xfscanf(file, "%d", &in);
        xfscanf(file, "%d", &out);
        xfscanf(file, "%d", &activation_func);
        if (i == 0)
            dnn->input_nb = in;
        dnn_add_layer(dnn, out, activation_func);
        for (int j = 0; j < in*out; j++)
            xfscanf(file, "%f", &dnn->layers[i]->weights[j]);
        for (int j = 0; j < out; j++)
            xfscanf(file, "%f", &dnn->layers[i]->bias[j]);
    }
    dnn->output_nb = output_layer(dnn)->out_nb;

    fclose(file);
    return dnn;
}

void dnn_copy_weights(dnn_t *dest, const dnn_t *src) {
    assert(src->layer_nb == dest->layer_nb);
    assert(src->input_nb == dest->input_nb);
    for (int i = 0; i < src->layer_nb; i++)
        assert(src->layers[i]->in_nb == dest->layers[i]->in_nb && src->layers[i]->out_nb == dest->layers[i]->out_nb);
    
    for (int i = 0; i < src->layer_nb; i++) {
        mat_copy(dest->layers[i]->weights, src->layers[i]->weights, src->layers[i]->out_nb, src->layers[i]->in_nb);
        vec_copy(dest->layers[i]->bias, src->layers[i]->bias, src->layers[i]->out_nb);
    }
}

dnn_t *dnn_copy(dnn_t *dnn) {
    dnn_t *copy = dnn_create(dnn->input_nb);
    for (int i = 0; i < dnn->layer_nb; i++)
        dnn_add_layer(copy, dnn->layers[i]->out_nb, dnn->layers[i]->activation_func);
    dnn_copy_weights(copy, dnn);
    return copy;
}

int dnn_predict(dnn_t *dnn, vec input) {
    return vec_argmax(dnn_forward(dnn, input), dnn->output_nb);
}

int dnn_predict_epsilon(dnn_t *dnn, vec input, float epsilon) {
    if (randf() > epsilon)
        return dnn_predict(dnn, input);
    else
        return randint(0, dnn->output_nb - 1);
}

float dnn_max_output(dnn_t *dnn, vec input) {
    return vec_max(dnn_forward(dnn, input), dnn->output_nb);
}

void dnn_gradient_descent_filtered(dnn_t *dnn, vec input, float target, int filter, float learning_rate) {
    assert(filter >= 0 && filter < dnn->output_nb);
    vec output = dnn_forward(dnn, input);
    layer_t *out_layer = output_layer(dnn);
    for (int i = 0; i < dnn->output_nb; i++)
        out_layer->delta[i] = (i == filter) ? (output[i] - target)*activation_function_derivative(out_layer->activation_func, output[i]) : 0.f;
    // vec_clip(out_layer->delta, -1.f, 1.f, dnn->output_nb);

    for (int i = dnn->layer_nb - 1; i >= 0; i--) {
        layer_t *layer = dnn->layers[i];

        // update weights and bias
        vec previous_activation = (i == 0) ? input : dnn->layers[i - 1]->activation;
        vec_add_scaled(layer->bias, layer->delta, -learning_rate, layer->out_nb);
        mat_add_outer(layer->weights, layer->delta, previous_activation, -learning_rate, layer->out_nb, layer->in_nb);

        // vec_clip(layer->bias, -1.0f, 1.0f, layer->out_nb);
        // vec_clip(layer->weights, -1.0f, 1.0f, layer->in_nb*layer->out_nb);

        // propagate delta
        if (i > 0) {
            layer_t *previous = dnn->layers[i - 1];
            mat_vec_mult_transpose(previous->delta, layer->weights, layer->delta, layer->out_nb, layer->in_nb);
                
            for (int j = 0; j < previous->out_nb; j++)
                previous->delta[j] *= activation_function_derivative(previous->activation_func, previous->activation[j]);
            // vec_clip(previous->delta, -1.f, 1.f, previous->out_nb);
        }  
    }
}