#ifndef NETWORK_TA_H
#define NETWORK_TA_H
#include "darknet.h"

extern layer lta;
extern int countlta;
extern int allnum;
extern float l_cost;
extern float *l_output;
extern float *l_delta;
extern float *n_delta;

float* add_bias_TA(float *output, float *biases, int batch, int n, int size);

float* scale_bias_TA(float *output, float *scales, int batch, int n, int size);

float* backward_bias_TA(float *bias_updates, float *delta, int batch, int n, int size);
void forward_connected_layer_TA(float *net_input, int net_train);

void backward_connected_layer_TA(float *net_input, float *net_delta, int net_train);

void update_connected_layer_TA(update_args a);

layer make_connected_layer_TA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);


#endif
