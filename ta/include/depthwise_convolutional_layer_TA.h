#ifndef DEPTHWISE_CONVOLUTIONAL_LAYER_TA_H
#define DEPTHWISE_CONVOLUTIONAL_LAYER_TA_H

#include "activations_TA.h"
//#include "layer_TA.h"
#include "network_TA.h"
#include "darknet_TA.h"

typedef layer_TA depthwise_convolutional_layer_TA;

depthwise_convolutional_layer_TA make_depthwise_convolutional_layer_TA_new(int batch, int h, int w, int c, int size, int stride, int padding, ACTIVATION_TA activation, int batch_normalize);

void forward_depthwise_convolutional_layer_TA_new(const depthwise_convolutional_layer_TA layer, network_TA net);

void backward_depthwise_convolutional_layer_TA_new(depthwise_convolutional_layer_TA layer, network_TA net);

void update_depthwise_convolutional_layer_TA_new(depthwise_convolutional_layer_TA layer, update_args_TA a);

void resize_depthwise_convolutional_layer_TA(depthwise_convolutional_layer_TA *layer, int w, int h);

void denormalize_depthwise_convolutional_layer_TA(depthwise_convolutional_layer_TA l);

void add_bias_TA(float *output, float *biases, int batch, int n, int size);
void backward_bias_TA(float *bias_updates, float *delta, int batch, int n, int size);

int depthwise_convolutional_out_height_TA(depthwise_convolutional_layer_TA layer);
int depthwise_convolutional_out_width_TA(depthwise_convolutional_layer_TA layer);

#endif
