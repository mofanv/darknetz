#ifndef UTILS_H_TA
#define UTILS_H_TA
#include <stdio.h>
#include "darknet.h"
#include "list.h"

float sum_array(float *a, int n);
float rand_uniform(float min, float max);
float rand_normal_TA(float mu, float sigma);

#endif
