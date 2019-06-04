#ifndef BATCHNORM_LAYER_TA_H
#define BATCHNORM_LAYER_TA_H

void forward_batchnorm_layer_TA(layer_TA l, network_TA net);

void backward_scale_cpu_TA(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);

void mean_delta_cpu_TA(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);

void variance_delta_cpu_TA(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);

void normalize_delta_cpu_TA(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void backward_batchnorm_layer_TA(layer_TA l, network_TA net);
#endif
