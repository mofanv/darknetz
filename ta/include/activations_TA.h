#ifndef ACTIVATIONS_TA_H
#define ACTIVATIONS_TA_H
#include "darknet_TA.h"
#include "math_TA.h"
#include <stdio.h>

char *get_activation_string_TA(ACTIVATION_TA a);

ACTIVATION_TA get_activation_TA(char *s);

float activate_TA(float x, ACTIVATION_TA a);

float*  activate_array_TA(float *x, const int n, const ACTIVATION_TA a);

float gradient_TA(float x, ACTIVATION_TA a);

float * gradient_array_TA(const float *x, const int n, const ACTIVATION_TA a, float *delta);


static inline float stair_activate_TA(float x)
{
    int n = ta_floor(x);
    if (n%2 == 0) return ta_floor(x/2.);
    else return (x - n) + ta_floor(x/2.);
}
static inline float hardtan_activate_TA(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate_TA(float x){return x;}
static inline float logistic_activate_TA(float x){return 1./(1. + ta_exp(-x));}
static inline float loggy_activate_TA(float x){return 2./(1. + ta_exp(-x)) - 1;}
static inline float relu_activate_TA(float x){return x*(x>0);}
static inline float elu_activate_TA(float x){return (x >= 0)*x + (x < 0)*(ta_exp(x)-1);}
static inline float selu_activate_TA(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(ta_exp(x)-1);}
static inline float relie_activate_TA(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate_TA(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate_TA(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate_TA(float x){return (ta_exp(2*x)-1)/(ta_exp(2*x)+1);}
static inline float plse_activate_TA(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate_TA(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}


static inline float lhtan_gradient_TA(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient_TA(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient_TA(float x){return 1;}
static inline float logistic_gradient_TA(float x){return (1-x)*x;}
static inline float loggy_gradient_TA(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient_TA(float x)
{
    if (ta_floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient_TA(float x){return (x>0);}
static inline float elu_gradient_TA(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float selu_gradient_TA(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline float relie_gradient_TA(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient_TA(float x){return (x>0)+.1;}
static inline float leaky_gradient_TA(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient_TA(float x){return 1-x*x;}
static inline float plse_gradient_TA(float x){return (x < 0 || x > 1) ? .01 : .125;}

#endif
