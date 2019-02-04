#ifndef SOFTMAX_LAYER_TA_H
#define SOFTMAX_LAYER_TA_H
#include "darknet_TA.h"

typedef layer_TA softmax_layer_TA;

void softmax_array_TA(float *input, int n, float temp, float *output);
softmax_layer_TA make_softmax_layer_TA_new(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss);
void forward_softmax_layer_TA(const softmax_layer_TA l, network_TA net);
void backward_softmax_layer_TA(const softmax_layer_TA l, network_TA net);

#endif
