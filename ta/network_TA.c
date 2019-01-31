#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "darknet_TA.h"
#include "blas_TA.h"
#include "network_TA.h"

#include "darknet_TA.h"

network_TA netta;
int roundnum = 0;
float err_sum = 0;
float avg_loss = -1;

void make_network_TA()
{
    netta.n = 3;
    netta.seen = calloc(1, sizeof(size_t));
    netta.layers = calloc(netta.n, sizeof(layer_TA));
    netta.t    = calloc(1, sizeof(int));
    netta.cost = calloc(1, sizeof(float));
    netta.batch = 100;
    netta.subdivisions = 1;
    netta.burn_in = 0;
    netta.learning_rate = 0.01;
    netta.power = 4;
    netta.policy = POLY_TA;
    netta.scale = 1;
    netta.max_batches = 500;
    netta.momentum = 0.9;
    netta.decay = 0.00005;
    netta.adam = 0;
    netta.B1 = 0.9;
    netta.B2 = 0.999;
    netta.eps = 0.0000001;

    //netta.truth = net->truth; ////// ing network.c train_network
}


void forward_network_TA()
{
    roundnum++;
    int i;
    for(i = 0; i < netta.n; ++i){
        netta.index = i;
        layer_TA l = netta.layers[i];

        if(l.delta){
            fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
        }

        l.forward_TA(l, netta);
        netta.input = l.output;

        if(l.truth) {
            netta.truth = l.output;
        }

    }

    calc_network_cost_TA();
}


void update_network_TA(update_args_TA a)
{
    int i;
    for(i = 0; i < netta.n; ++i){
        layer_TA l = netta.layers[i];
        if(l.update_TA){
            l.update_TA(l, a);
        }
    }
}


void calc_network_cost_TA()
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < netta.n; ++i){
        if(netta.layers[i].cost){
            sum += netta.layers[i].cost[0];
            ++count;
        }
    }
    *netta.cost = sum/count;
    err_sum += *netta.cost;
}


void calc_network_loss_TA(int n, int batch)
{
    float loss = (float)err_sum/(n*batch);

    if(avg_loss == -1) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;

    printf("%f, %f avg in TA\n",loss, avg_loss);
    err_sum = 0;
    free(net_truth);
}


void backward_network_TA(float *ca_net_input, float *ca_net_delta)
{
    int i;

    for(i = netta.n-1; i >= 0; --i){
        layer_TA l = netta.layers[i];

        if(l.stopbackward) break;
        if(i == 0){
            netta.input = ca_net_input;
            netta.delta = ca_net_delta;
        }else{
            layer_TA prev = netta.layers[i-1];
            netta.input = prev.output;
            netta.delta = prev.delta;
        }

        netta.index = i;
        l.backward_TA(l, netta);
    }

    //backward_network_back_TA_params(netta.input, netta.delta, netta.layers[0].inputs, netta.batch);
}
