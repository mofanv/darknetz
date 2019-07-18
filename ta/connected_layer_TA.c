#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "utils_TA.h"
#include "darknet_TA.h"
#include "gemm_TA.h"
#include "math_TA.h"
#include "blas_TA.h"
#include "activations_TA.h"
#include "batchnorm_layer_TA.h"
#include "connected_layer_TA.h"
#include "convolutional_layer_TA.h"

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

void forward_connected_layer_TA_new(layer_TA l, network_TA net)
{
    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;

    gemm_TA(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if(l.batch_normalize){
        forward_batchnorm_layer_TA(l, net);
    } else {
        add_bias_TA(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_TA_new(layer_TA l, network_TA net)
{
    gradient_array_TA(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer_TA(l, net);
    } else {
        backward_bias_TA(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;

    gemm_TA(1,0,m,n,k,1,a,m,b,n,1,c,n);

    //diff_private_SGD(l.bias_updates, l.outputs);
    //diff_private_SGD(l.weight_updates, l.inputs*l.outputs);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if(c) gemm_TA(0,0,m,n,k,1,a,k,b,n,1,c,n);

}


void update_connected_layer_TA_new(layer_TA l, update_args_TA a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu_TA(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);

    scal_cpu_TA(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu_TA(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu_TA(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu_TA(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu_TA(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu_TA(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

layer_TA make_connected_layer_TA_new(int batch, int inputs, int outputs, ACTIVATION_TA activation, int batch_normalize, int adam)
{
    int i;
    layer_TA l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED_TA;

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

    l.forward_TA = forward_connected_layer_TA_new;
    l.backward_TA = backward_connected_layer_TA_new;
    l.update_TA = update_connected_layer_TA_new;

    //float scale = 1./sqrt(inputs);
    float scale = ta_sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        //l.weight_updates[i] = 1.0f;
        l.weights[i] = scale * rand_uniform_TA(-1, 1);
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
    //IMSG("connected_TA                         %4d  ->  %4d\n", inputs, outputs);

    return l;
}
