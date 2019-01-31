#include "activations_TA.h"

#include "math_ta.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char *get_activation_string_TA(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation_TA(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    //fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}


float activate_TA(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_TA(x);
        case LOGISTIC:
            return logistic_activate_TA(x);
        case LOGGY:
            return loggy_activate_TA(x);
        case RELU:
            return relu_activate_TA(x);
        case ELU:
            return elu_activate_TA(x);
        case SELU:
            return selu_activate_TA(x);
        case RELIE:
            return relie_activate_TA(x);
        case RAMP:
            return ramp_activate_TA(x);
        case LEAKY:
            return leaky_activate_TA(x);
        case TANH:
            return tanh_activate_TA(x);
        case PLSE:
            return plse_activate_TA(x);
        case STAIR:
            return stair_activate_TA(x);
        case HARDTAN:
            return hardtan_activate_TA(x);
        case LHTAN:
            return lhtan_activate_TA(x);
    }
    return 0;
}

float*  activate_array_TA(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate_TA(x[i], a);
    }
    return x;
}


float gradient_TA(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_TA(x);
        case LOGISTIC:
            return logistic_gradient_TA(x);
        case LOGGY:
            return loggy_gradient_TA(x);
        case RELU:
            return relu_gradient_TA(x);
        case ELU:
            return elu_gradient_TA(x);
        case SELU:
            return selu_gradient_TA(x);
        case RELIE:
            return relie_gradient_TA(x);
        case RAMP:
            return ramp_gradient_TA(x);
        case LEAKY:
            return leaky_gradient_TA(x);
        case TANH:
            return tanh_gradient_TA(x);
        case PLSE:
            return plse_gradient_TA(x);
        case STAIR:
            return stair_gradient_TA(x);
        case HARDTAN:
            return hardtan_gradient_TA(x);
        case LHTAN:
            return lhtan_gradient_TA(x);
    }
    return 0;
}

float * gradient_array_TA(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient_TA(x[i], a);
    }
    return delta;
}
