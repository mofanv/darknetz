#ifndef MAIN_CA_H
#define MAIN_CA_H

#include <err.h>
#include <stdio.h>
#include <string.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

#define MAKE_CONNECTED_CMD 0
#define MAKE_SOFTMAX_CMD 1
#define MAKE_COST_CMD 2
#define FORWARD_CMD 3
#define BACKWARD_CMD 4
#define BACKWARD_ADD_CMD 5
#define UPDATE_CMD 6
#define NET_TRUTH_CMD 7
#define CALC_LOSS_CMD 8

#define TA_DARKNETP_UUID \
	{ 0x7fc5c039, 0x0542, 0x4ee1, \
		{ 0x80, 0xaf, 0xb4, 0xea, 0xb2, 0xf1, 0x99, 0x8d} }

extern TEEC_Context ctx;
extern TEEC_Session sess;

extern float *net_input;
extern float *net_delta;


void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void forward_connected_layer_CA(float *net_input, int net_inputs, int net_batch, int net_train);

void backward_connected_layer_CA_addidion();

void backward_connected_layer_CA(float *net_input, int l_inputs, int batch, float *net_delta, int net_train);

void update_connected_layer_CA(update_args a);

void make_softmax_layer_CA(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss);

void make_cost_layer_CA(int batch, int inputs, COST_TYPE cost_type, float scale, float ratio, float noobject_scale, float thresh);

void net_truth_CA(float *net_truth, int net_truths, int net_batch);

void calc_network_loss_CA(int n, int batch);

#endif
