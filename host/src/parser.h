#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

extern int partition_point;
extern int count_global;

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#endif
