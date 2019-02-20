#ifndef DROPOUT_LAYER_TA_H
#define DROPOUT_LAYER_TA_H

#include "darknet_TA.h"

typedef layer_TA dropout_layer_TA;

dropout_layer_TA make_dropout_layer_TA_new(int batch, int inputs, float probability, int w, int h, int c, int netnum);

void forward_dropout_layer_TA_new(dropout_layer_TA l, network_TA net);
void backward_dropout_layer_TA_new(dropout_layer_TA l, network_TA net);
void resize_dropout_layer_TA(dropout_layer_TA *l, int inputs);

#endif
