#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <darknetp_ta.h>
#include "activations_TA.h"
#include "network_TA.h"
#include "darknet.h"
#include "diffprivate_TA.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOOKUP_SIZE 4096


TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("has been called");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("has been called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    /* Unused parameters */
    (void)&params;
    (void)&sess_ctx;

    IMSG("I'm Vincent, from secure world!\n");
    return TEE_SUCCESS;
}


void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    (void)&sess_ctx; /* Unused parameter */
    IMSG("Goodbye!\n");
}


static TEE_Result make_connected_layer(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *passarg;
    passarg = params[0].memref.buffer;
    int batch = passarg[0];
    int inputs = passarg[1];
    int outputs = passarg[2];
    int batch_normalize = passarg[3];
    int adam = passarg[4];

    char *acti;
    acti = params[1].memref.buffer;
    ACTIVATION activation = get_activation_TA(acti);

    lta = make_connected_layer_TA(batch, inputs, outputs, activation, batch_normalize, adam);

/*
    params[2].memref.buffer = lta.output;
    params[2].memref.size = sizeof(float) * lta.batch * lta.outputs;
    params[3].memref.buffer = lta.delta;
    params[3].memref.size = sizeof(float) * lta.batch * lta.outputs;
*/

    return TEE_SUCCESS;
}

static TEE_Result forward_connected_layer(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    int net_train = params[1].value.a;

    forward_connected_layer_TA(net_input, net_train);

    float * buffer1 = params[2].memref.buffer;
    float * buffer2 = params[3].memref.buffer;

    for(int z=0; z<lta.batch*lta.outputs; z++){
        buffer1[z] = lta.output[z];
    }

    for(int z=0; z<lta.batch*lta.outputs; z++){
        buffer2[z] = lta.delta[z];
    }

    return TEE_SUCCESS;
}

static TEE_Result backward_connected_layer_addition(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE);

    //float *ltaoutput_diff = diff_private(lta.output, lta.outputs*lta.batch, 4.0f, 4.0f);
    //float *ltadelta_diff = diff_private(lta.delta, lta.outputs*lta.batch, 4.0f, 4.0f);
    //IMSG("diff");

    float * buffer1 = params[0].memref.buffer;
    float * buffer2 = params[1].memref.buffer;
    float * buffer3 = params[2].memref.buffer;

    for(int z=0; z<lta.batch*lta.outputs; z++){
        buffer1[z] = lta.output[z];
    }

    for(int z=0; z<lta.batch*lta.outputs; z++){
        buffer2[z] = lta.delta[z];
    }

    IMSG("lta_inputs: %d \n", lta.inputs);

    for(int z=0; z<lta.inputs; z++){
        buffer3[z] = n_delta[z];
    }

    //free(ltaoutput_diff);
    //free(ltadelta_diff);
    return TEE_SUCCESS;
}



static TEE_Result backward_connected_layer(uint32_t param_types,
                                           TEE_Param params[4])
{


    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    float *net_delta = params[1].memref.buffer;
    int net_train = params[2].value.a;

    backward_connected_layer_TA(net_input, net_delta, net_train);

    return TEE_SUCCESS;
}

static TEE_Result update_connected_layer(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *passint = params[0].memref.buffer;
    float *passflo = params[1].memref.buffer;

     update_args argup;
     argup.batch = passint[0];
     argup.learning_rate = passflo[0];
     argup.momentum = passflo[1];
     argup.decay = passflo[2];
     argup.adam = passint[1];
     argup.B1 = passflo[3];
     argup.B2 = passflo[4];
     argup.eps = passflo[5];
     argup.t = passint[2];

    update_connected_layer_TA(argup);

    return TEE_SUCCESS;
}


TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */

    switch (cmd_id) {
        case MAKE_CMD:

        return make_connected_layer(param_types, params);

        case FORWARD_CMD:
        return forward_connected_layer(param_types, params);

        case BACKWARD_CMD:
        return backward_connected_layer(param_types, params);

        case BACKWARD_ADD_CMD:
        return backward_connected_layer_addition(param_types, params);

        case UPDATE_CMD:
        return update_connected_layer(param_types, params);

        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
