#include "activations_TA.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string_TA(ACTIVATION_TA a)
{
    switch(a){
        case LOGISTIC_TA:
            return "logistic";
        case LOGGY_TA:
            return "loggy";
        case RELU_TA:
            return "relu";
        case ELU_TA:
            return "elu";
        case SELU_TA:
            return "selu";
        case RELIE_TA:
            return "relie";
        case RAMP_TA:
            return "ramp";
        case LINEAR_TA:
            return "linear";
        case TANH_TA:
            return "tanh";
        case PLSE_TA:
            return "plse";
        case LEAKY_TA:
            return "leaky";
        case STAIR_TA:
            return "stair";
        case HARDTAN_TA:
            return "hardtan";
        case LHTAN_TA:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION_TA get_activation_TA(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC_TA;
    if (strcmp(s, "loggy")==0) return LOGGY_TA;
    if (strcmp(s, "relu")==0) return RELU_TA;
    if (strcmp(s, "elu")==0) return ELU_TA;
    if (strcmp(s, "selu")==0) return SELU_TA;
    if (strcmp(s, "relie")==0) return RELIE_TA;
    if (strcmp(s, "plse")==0) return PLSE_TA;
    if (strcmp(s, "hardtan")==0) return HARDTAN_TA;
    if (strcmp(s, "lhtan")==0) return LHTAN_TA;
    if (strcmp(s, "linear")==0) return LINEAR_TA;
    if (strcmp(s, "ramp")==0) return RAMP_TA;
    if (strcmp(s, "leaky")==0) return LEAKY_TA;
    if (strcmp(s, "tanh")==0) return TANH_TA;
    if (strcmp(s, "stair")==0) return STAIR_TA;
    //fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU_TA;
}



float activate_TA(float x, ACTIVATION_TA a)
{
    switch(a){
        case LINEAR_TA:
            return linear_activate_TA(x);
        case LOGISTIC_TA:
            return logistic_activate_TA(x);
        case LOGGY_TA:
            return loggy_activate_TA(x);
        case RELU_TA:
            return relu_activate_TA(x);
        case ELU_TA:
            return elu_activate_TA(x);
        case SELU_TA:
            return selu_activate_TA(x);
        case RELIE_TA:
            return relie_activate_TA(x);
        case RAMP_TA:
            return ramp_activate_TA(x);
        case LEAKY_TA:
            return leaky_activate_TA(x);
        case TANH_TA:
            return tanh_activate_TA(x);
        case PLSE_TA:
            return plse_activate_TA(x);
        case STAIR_TA:
            return stair_activate_TA(x);
        case HARDTAN_TA:
            return hardtan_activate_TA(x);
        case LHTAN_TA:
            return lhtan_activate_TA(x);
    }
    return 0;
}

float*  activate_array_TA(float *x, const int n, const ACTIVATION_TA a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate_TA(x[i], a);
    }
    return x;
}


float gradient_TA(float x, ACTIVATION_TA a)
{
    switch(a){
        case LINEAR_TA:
            return linear_gradient_TA(x);
        case LOGISTIC_TA:
            return logistic_gradient_TA(x);
        case LOGGY_TA:
            return loggy_gradient_TA(x);
        case RELU_TA:
            return relu_gradient_TA(x);
        case ELU_TA:
            return elu_gradient_TA(x);
        case SELU_TA:
            return selu_gradient_TA(x);
        case RELIE_TA:
            return relie_gradient_TA(x);
        case RAMP_TA:
            return ramp_gradient_TA(x);
        case LEAKY_TA:
            return leaky_gradient_TA(x);
        case TANH_TA:
            return tanh_gradient_TA(x);
        case PLSE_TA:
            return plse_gradient_TA(x);
        case STAIR_TA:
            return stair_gradient_TA(x);
        case HARDTAN_TA:
            return hardtan_gradient_TA(x);
        case LHTAN_TA:
            return lhtan_gradient_TA(x);
    }
    return 0;
}

float * gradient_array_TA(const float *x, const int n, const ACTIVATION_TA a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient_TA(x[i], a);
    }
    return delta;
}
