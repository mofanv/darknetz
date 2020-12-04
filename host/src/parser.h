#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

extern int partition_point1;
extern int partition_point2;
extern int frozen_bool;
extern int sepa_save_bool;
extern int count_global;
extern int global_dp;

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#endif
