#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "darknet_TA.h"

typedef layer_TA maxpool_layer_TA;

maxpool_layer_TA make_maxpool_layer_TA(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_maxpool_layer_TA(maxpool_layer_TA *l, int w, int h);
void forward_maxpool_layer_TA_new(const maxpool_layer_TA l, network_TA net);
void backward_maxpool_layer_TA_new(const maxpool_layer_TA l, network_TA net);

#endif
