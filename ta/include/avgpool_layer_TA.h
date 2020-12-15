#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "darknet_TA.h"

typedef layer_TA avgpool_layer_TA;

avgpool_layer_TA make_avgpool_layer_TA(int batch, int w, int h, int c);
void resize_avgpool_layer_TA(avgpool_layer_TA *l, int w, int h);
void forward_avgpool_layer_TA_new(const avgpool_layer_TA l, network_TA net);
void backward_avgpool_layer_TA_new(const avgpool_layer_TA l, network_TA net);

#endif
