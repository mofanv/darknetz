#ifndef CONVOLUTIONAL_LAYER_TA_H
#define CONVOLUTIONAL_LAYER_TA_H

#include "darknet_TA.h"

typedef layer_TA convolutional_layer_TA;

void add_bias_TA(float *output, float *biases, int batch, int n, int size);

void scale_bias_TA(float *output, float *scales, int batch, int n, int size);

void backward_bias_TA(float *bias_updates, float *delta, int batch, int n, int size);

convolutional_layer_TA make_convolutional_layer_TA_new(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION_TA activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot);

void forward_convolutional_layer_TA_new(convolutional_layer_TA l, network_TA net);

void backward_convolutional_layer_TA_new(convolutional_layer_TA l, network_TA net);

void update_convolutional_layer_TA_new(convolutional_layer_TA l, update_args_TA a);

#endif
