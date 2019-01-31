#ifndef MAIN_CA_H
#define MAIN_CA_H

#include <err.h>
#include <stdio.h>
#include <string.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

#define MAKE_CMD 0
#define FORWARD_CMD 1
#define BACKWARD_CMD 2
#define BACKWARD_ADD_CMD 3
#define UPDATE_CMD 4

#define TA_DARKNETP_UUID \
	{ 0x7fc5c039, 0x0542, 0x4ee1, \
		{ 0x80, 0xaf, 0xb4, 0xea, 0xb2, 0xf1, 0x99, 0x8d} }

extern TEEC_Context ctx;
extern TEEC_Session sess;

extern float *lta_output;
extern float *lta_delta;
extern float *n_delta;


void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void forward_connected_layer_CA(float *net_input, int net_inputs, int net_train);

void backward_connected_layer_CA_addidion();

void backward_connected_layer_CA(float *net_input, int net_inputs, float *net_delta, int net_deltas, int net_train);

void update_connected_layer_CA(update_args a);

#endif
