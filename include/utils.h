#ifndef UTILS_H
#define UTILS_H

#include "types.h"
#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <time.h>


#define PI 3.1415926535f

// standard functions
static inline float maxf(float x, float y) {
    return x > y ? x : y;
}

static inline float minf(float x, float y) {
    return x > y ? y : x;
}

static inline int maxint(int x, int y) {
    return x > y ? x : y;
}

static inline int minint(int x, int y) {
    return x > y ? y : x;
}

static inline float clipf(float x, float min, float max) {
    return x > max ? max : x < min ? min : x;
}


// vector functions

static inline int vec_argmax(const vec x, int n) {
    float max = x[0];
    int index = 0;
    for (int i = 0; i < n; i++)
        if (x[i] > max) {
            max = x[i];
            index = i;
        }
    return index;
}

static inline float vec_max(const vec x, int n) {
    float max = x[0];
    for (int i = 0; i < n; i++)
        if (x[i] > max)
            max = x[i];
    return max;
}

static inline void vec_clip(vec x, float min, float max, int n) {
    for (int i = 0; i < n; i++)
        x[i] = clipf(x[i], min, max);
}

// y <- y *. x
static inline void vec_mult(vec y, const vec x, int n) {
    for (int i = 0; i < n; i++)
        y[i] *= x[i];
}

// y <- y + alpha*x
static inline void vec_add_scaled(vec y, const vec x, float alpha, int n) {
    cblas_saxpy(n, alpha, x, 1, y, 1);
}

static inline void vec_copy(vec dest, const vec src, int n) {
    memcpy(dest, src, sizeof(float)*n);
}

// matrix functions

// z <- Ax + y
static inline void mat_vec_affine(vec z, const mat A, const vec x, const vec y, int rows, int cols) {
    vec_copy(z, y, rows);
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        rows, cols,
        1.f,
        A, cols,
        x, 1,
        1.f,
        z, 1
    );
}

// y <- ATx
static inline void mat_vec_mult_transpose(vec y, const mat A, const vec x, int rows, int cols) {
    cblas_sgemv(
        CblasRowMajor,
        CblasTrans,
        rows, cols,
        1.f,
        A, cols,
        x, 1,
        0.f,
        y, 1
    );
}

// A <- A + alpha*xyT
static inline void mat_add_outer(mat A, const vec x, const vec y, float alpha, int rows, int cols) {
    cblas_sger(
        CblasRowMajor,
        rows, cols,
        alpha,
        x, 1,
        y, 1,
        A, cols
    );
}

static inline void mat_copy(mat dest, const mat src, int rows, int cols) {
    memcpy(dest, src, sizeof(float)*rows*cols);
}

// random functions
static inline void init_seed(void) {
    srand(time(NULL));
}

static inline float rand_normal(float sigma) {
    float u1 = ((float)rand() + 1.f) / ((float)RAND_MAX + 1.f);
    float u2 = ((float)rand() + 1.f) / ((float)RAND_MAX + 1.f);
    return sqrtf(-2.f * logf(u1)) * cosf(2.f * PI * u2) * sigma;
}

static inline float randf(void) {
    return (float)rand()/(float)RAND_MAX;
}

static inline int randint(int min, int max) {
    return min + rand()%(max - min + 1);
}

#endif