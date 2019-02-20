#ifndef TA_DARKNETP_H
#define TA_DARKNETP_H

#include "darknet_TA.h"
#include "network_TA.h"

extern float *netta_truth;

/*
 * This UUID is generated with uuidgen
 * the ITU-T UUID generator at http://www.itu.int/ITU-T/asn1/uuid.html
 */
#define TA_DARKNETP_UUID \
	{ 0x7fc5c039, 0x0542, 0x4ee1, \
		{ 0x80, 0xaf, 0xb4, 0xea, 0xb2, 0xf1, 0x99, 0x8d} }

/* The function IDs implemented in this TA */
#define MAKE_NETWORK_CMD 1
#define WORKSPACE_NETWORK_CMD 2
#define MAKE_CONV_CMD 3
#define MAKE_MAX_CMD 4
#define MAKE_DROP_CMD 5
#define MAKE_CONNECTED_CMD 6
#define MAKE_SOFTMAX_CMD 7
#define MAKE_COST_CMD 8
#define FORWARD_CMD 9
#define BACKWARD_CMD 10
#define BACKWARD_ADD_CMD 11
#define UPDATE_CMD 12
#define NET_TRUTH_CMD 13
#define CALC_LOSS_CMD 14


#endif /*TA_DARKNETP_H*/
