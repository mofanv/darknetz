#ifndef BLAS_TA_H
#define BLAS_TA_H
#include "darknet_TA.h"

void fill_cpu_TA(int N, float ALPHA, float *X, int INCX);

void copy_cpu_TA(int N, float *X, int INCX, float *Y, int INCY);

void mean_cpu_TA(float *x, int batch, int filters, int spatial, float *mean);

void variance_cpu_TA(float *x, float *mean, int batch, int filters, int spatial, float *variance);

void scal_cpu_TA(int N, float ALPHA, float *X, int INCX);

void axpy_cpu_TA(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);

void normalize_cpu_TA(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void softmax_TA(float *input, int n, float temp, int stride, float *output);

void softmax_cpu_TA(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);

void softmax_x_ent_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);

void smooth_l1_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);

void l1_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);

void l2_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);
#endif
