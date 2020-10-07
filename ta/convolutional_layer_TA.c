#include "convolutional_layer_TA.h"
#include "batchnorm_layer_TA.h"

#include "utils_TA.h"
#include "gemm_TA.h"
#include "math_TA.h"
#include "blas_TA.h"
#include "im2col_TA.h"
#include "col2im_TA.h"
#include "darknet_TA.h"
#include "activations_TA.h"

#include <stdio.h>
#include <time.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

void swap_binary_TA(convolutional_layer_TA *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;
}

void add_bias_TA(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}


void scale_bias_TA(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias_TA(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array_TA(delta+size*(i+b*n), size);
        }
    }
}

int convolutional_out_height_TA(convolutional_layer_TA l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width_TA(convolutional_layer_TA l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}


void binarize_weights_TA(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu_TA(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}


static size_t get_workspace_size(layer_TA l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}


convolutional_layer_TA make_convolutional_layer_TA_new(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION_TA activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot)
{
    int i;
    convolutional_layer_TA l = {0};
    l.type = CONVOLUTIONAL_TA;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = ta_sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) {
        l.weights[i] = scale*rand_normal_TA(0,1);
    }

    int out_w = convolutional_out_width_TA(l);
    int out_h = convolutional_out_height_TA(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward_TA = forward_convolutional_layer_TA_new;
    l.backward_TA = backward_convolutional_layer_TA_new;
    l.update_TA = update_convolutional_layer_TA_new;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    l.flipped = flipped;
    l.dot = dot;

//IMSG("conv_TA%4d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}


void forward_convolutional_layer_TA_new(convolutional_layer_TA l, network_TA net)
{
    int i, j;

    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights_TA(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary_TA(&l);
        binarize_cpu_TA(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }



    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu_TA(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_TA(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }



    if(l.batch_normalize){
        forward_batchnorm_layer_TA(l, net);
    } else {
        add_bias_TA(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary_TA(&l);

}

void backward_convolutional_layer_TA_new(convolutional_layer_TA l, network_TA net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array_TA(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer_TA(l, net);
    } else {
        backward_bias_TA(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu_TA(im, l.c/l.groups, l.h, l.w,
                           l.size, l.stride, l.pad, b);
            }

            gemm_TA(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_TA(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu_TA(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer_TA_new(convolutional_layer_TA l, update_args_TA a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu_TA(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu_TA(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu_TA(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu_TA(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu_TA(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu_TA(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu_TA(l.nweights, momentum, l.weight_updates, 1);
}
