#ifndef COST_LAYER_TA_H
#define COST_LAYER_TA_H

#include "darknet_TA.h"

typedef layer_TA cost_layer_TA;

COST_TYPE_TA get_cost_type_TA(char *s);

char *get_cost_string_TA(COST_TYPE_TA a);

cost_layer_TA make_cost_layer_TA_new(int batch, int inputs, COST_TYPE_TA cost_type, float scale, float ratio, float noobject_scale, float thresh);

void forward_cost_layer_TA(cost_layer_TA l, network_TA net);

void backward_cost_layer_TA(const cost_layer_TA l, network_TA net);

#endif
