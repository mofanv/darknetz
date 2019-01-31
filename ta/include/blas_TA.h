#ifndef BLAS_TA_H
#define BLAS_TA_H
#include "darknet.h"

float* fill_cpu_TA(int N, float ALPHA, float *X, int INCX);

float* copy_cpu_TA(int N, float *X, int INCX, float *Y, int INCY);

float* mean_cpu_TA(float *x, int batch, int filters, int spatial, float *mean);

float* variance_cpu_TA(float *x, float *mean, int batch, int filters, int spatial, float *variance);

float* scal_cpu_TA(int N, float ALPHA, float *X, int INCX);

float* axpy_cpu_TA(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);

float* normalize_cpu_TA(float *x, float *mean, float *variance, int batch, int filters, int spatial);

#endif
