#ifndef NETWORK_TA_H
#define NETWORK_TA_H

extern network_TA netta;
extern int roundnum;

void make_network_TA();

void calc_network_cost_TA();

void calc_network_loss_TA(int n, int batch);

void forward_network_TA();

void backward_network_TA(float *ca_net_input, float *ca_net_delta);

void update_network_TA(update_args_TA a);
#endif
