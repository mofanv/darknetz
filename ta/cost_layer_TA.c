#include "cost_layer_TA.h"

#include "utils_TA.h"
#include "blas_TA.h"
#include "math_TA.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

COST_TYPE_TA get_cost_type_TA(char *s)
{
    if (strcmp(s, "seg")==0) return SEG_TA;
    if (strcmp(s, "sse")==0) return SSE_TA;
    if (strcmp(s, "masked")==0) return MASKED_TA;
    if (strcmp(s, "smooth")==0) return SMOOTH_TA;
    if (strcmp(s, "L1")==0) return L1_TA;
    if (strcmp(s, "wgan")==0) return WGAN_TA;
    IMSG("Couldn't find cost type %s, going with SSE\n", s);
    return SSE_TA;
}


char *get_cost_string_TA(COST_TYPE_TA a)
{
    switch(a){
        case SEG_TA:
            return "seg";
        case SSE_TA:
            return "sse";
        case MASKED_TA:
            return "masked";
        case SMOOTH_TA:
            return "smooth";
        case L1_TA:
            return "L1";
        case WGAN_TA:
            return "wgan";
    }
    return "sse";
}


cost_layer_TA make_cost_layer_TA_new(int batch, int inputs, COST_TYPE_TA cost_type, float scale, float ratio, float noobject_scale, float thresh)
{
    //IMSG("cost_TA                                        %4d\n",  inputs);
    cost_layer_TA l = {0};
    l.type = COST_TA;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.scale = scale;
    l.ratio = ratio;
    l.noobject_scale = noobject_scale;
    l.thresh = thresh;

    l.forward_TA = forward_cost_layer_TA;
    l.backward_TA = backward_cost_layer_TA;

    return l;
}



void forward_cost_layer_TA(cost_layer_TA l, network_TA net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED_TA){
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM_TA) net.input[i] = SECRET_NUM_TA;
        }
    }
    if(l.cost_type == SMOOTH_TA){
        smooth_l1_cpu_TA(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1_TA){
        l1_cpu_TA(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
        l2_cpu_TA(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    l.cost[0] = sum_array_TA(l.output, l.batch*l.inputs);

}



void backward_cost_layer_TA(const cost_layer_TA l, network_TA net)
{
    axpy_cpu_TA(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}
