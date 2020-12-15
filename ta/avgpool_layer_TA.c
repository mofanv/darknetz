#include "avgpool_layer_TA.h"
#include "math_TA.h"

#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

avgpool_layer_TA make_avgpool_layer_TA(int batch, int w, int h, int c)
{
    avgpool_layer_TA l = {0};
    l.type = AVGPOOL_TA;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward_TA = forward_avgpool_layer_TA_new;
    l.backward_TA = backward_avgpool_layer_TA_new;

    return l;
}

void resize_avgpool_layer_TA(avgpool_layer_TA *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer_TA_new(const avgpool_layer_TA l, network_TA net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}

void backward_avgpool_layer_TA_new(const avgpool_layer_TA l, network_TA net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}
