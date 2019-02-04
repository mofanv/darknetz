#ifndef TA_DARKNETP_H
#define TA_DARKNETP_H

#include "darknet_TA.h"
#include "network_TA.h"

extern layer_TA lta;
extern layer_TA lta_sm;
extern layer_TA lta_c;
extern float *netta_truth;

/*
 * This UUID is generated with uuidgen
 * the ITU-T UUID generator at http://www.itu.int/ITU-T/asn1/uuid.html
 */
#define TA_DARKNETP_UUID \
	{ 0x7fc5c039, 0x0542, 0x4ee1, \
		{ 0x80, 0xaf, 0xb4, 0xea, 0xb2, 0xf1, 0x99, 0x8d} }

/* The function IDs implemented in this TA */
#define MAKE_CONNECTED_CMD 0
#define MAKE_SOFTMAX_CMD 1
#define MAKE_COST_CMD 2
#define FORWARD_CMD 3
#define BACKWARD_CMD 4
#define BACKWARD_ADD_CMD 5
#define UPDATE_CMD 6
#define NET_TRUTH_CMD 7
#define CALC_LOSS_CMD 8


#endif /*TA_DARKNETP_H*/
