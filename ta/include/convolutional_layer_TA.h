#ifndef CONVOLUTIONAL_LAYER_TA_H
#define CONVOLUTIONAL_LAYER_TA_H

void add_bias_TA(float *output, float *biases, int batch, int n, int size);

void scale_bias_TA(float *output, float *scales, int batch, int n, int size);

void backward_bias_TA(float *bias_updates, float *delta, int batch, int n, int size);

#endif
