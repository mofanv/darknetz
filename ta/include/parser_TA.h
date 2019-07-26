#ifndef PAR_TA_H
#define PAR_TA_H
#include "darknet_TA.h"

void load_weights_TA(float *vec, int length, int layer_i, char type, int transpose);

void save_weights_TA(float *weights_encrypted, int length, int layer_i, char type);
#endif
