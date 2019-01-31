/* this version only supports run connected layers in the TA
 ---- IN THE TA ---- IN THE TA ---- IN THE TA ----
 */

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "utils_TA.h"

#include "darknet.h"
#include "network_TA.h"
#include "gemm_TA.h"
#include "blas_TA.h"
#include "activations_TA.h"
#include "batchnorm_layer_TA.h"

layer lta;

int countlta = 0;
int allnum = 0;

float l_cost = 0.0f;
float *l_output = {0};
float *l_delta = {0};
float *n_delta = {0};

float* add_bias_TA(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
    return output;
}

float* scale_bias_TA(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
    return output;
}

float* backward_bias_TA(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
    return bias_updates;
}

void forward_connected_layer_TA(float *net_input, int net_train)
{
    if(lta.delta){
        lta.delta = fill_cpu_TA(lta.outputs * lta.batch, 0, lta.delta, 1);
    }

    lta.output = fill_cpu_TA(lta.outputs*lta.batch, 0, lta.output, 1);
    int m = lta.batch;
    int k = lta.inputs;
    int n = lta.outputs;
    float *a = net_input;
    float *b = lta.weights;
    float *c = lta.output;

    lta.output = gemm_TA(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if(lta.batch_normalize){
        forward_batchnorm_layer_TA(lta, net_input, net_train);
    } else {
        lta.output = add_bias_TA(lta.output, lta.biases, lta.batch, lta.outputs, 1);
    }

    lta.output = activate_array_TA(lta.output, lta.outputs*lta.batch, lta.activation);

    //if(lta.cost){
    //    l_cost = lta.cost[0];
    //}
}

void backward_connected_layer_TA(float *net_input, float *net_delta, int net_train)
{
    lta.delta = gradient_array_TA(lta.output, lta.outputs*lta.batch, lta.activation, lta.delta);

    if(lta.batch_normalize){
        backward_batchnorm_layer_TA(lta, net_input, net_train);
    } else {
        lta.bias_updates = backward_bias_TA(lta.bias_updates, lta.delta, lta.batch, lta.outputs, 1);
    }

    int m = lta.outputs;
    int k = lta.batch;
    int n = lta.inputs;
    float *a = lta.delta;
    float *b = net_input;
    float *c = lta.weight_updates;

    lta.weight_updates = gemm_TA(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = lta.batch;
    k = lta.outputs;
    n = lta.inputs;

    a = lta.delta;
    b = lta.weights;
    c = net_delta;

    if(c) n_delta = gemm_TA(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void update_connected_layer_TA(update_args a)
{
    float learning_rate = a.learning_rate*lta.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    lta.biases = axpy_cpu_TA(lta.outputs, learning_rate/batch, lta.bias_updates, 1, lta.biases, 1);
    lta.bias_updates = scal_cpu_TA(lta.outputs, momentum, lta.bias_updates, 1);

    if(lta.batch_normalize){
        lta.scales = axpy_cpu_TA(lta.outputs, learning_rate/batch, lta.scale_updates, 1, lta.scales, 1);
        lta.scale_updates = scal_cpu_TA(lta.outputs, momentum, lta.scale_updates, 1);
    }

    lta.weight_updates = axpy_cpu_TA(lta.inputs*lta.outputs, -decay*batch, lta.weights, 1, lta.weight_updates, 1);
    lta.weights = axpy_cpu_TA(lta.inputs*lta.outputs, learning_rate/batch, lta.weight_updates, 1, lta.weights, 1);
    lta.weight_updates = scal_cpu_TA(lta.inputs*lta.outputs, momentum, lta.weight_updates, 1);
}


layer make_connected_layer_TA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));

    //float scale = 1./sqrt(inputs);
    float scale = ta_sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        //l.weight_updates[i] = 1.0f;
        l.weights[i] = scale * rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

    l.activation = activation;
    //fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);

    countlta++;
    return l;
}
