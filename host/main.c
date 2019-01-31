#include <err.h>
#include <stdio.h>
#include <string.h>

#include "darknet.h"
#include "activations.h"
#include "main.h"

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* TEE resources */
TEEC_Context ctx;
TEEC_Session sess;

float *lta_output;
float *lta_delta;
float *n_delta;

void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    lta_output = malloc(sizeof(float) * 103000);
    lta_delta = malloc(sizeof(float) * 103000);
    n_delta = malloc(sizeof(float) * 103000);

    int passarg[5];
    passarg[0] = batch;
    passarg[1] = inputs;
    passarg[2] = outputs;
    passarg[3] = batch_normalize;
    passarg[4] = adam;

    char *actv = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passarg;
    op.params[0].tmpref.size = sizeof(passarg);

    op.params[1].tmpref.buffer = actv;
    op.params[1].tmpref.size = strlen(actv);

/*
    op.params[2].tmpref.buffer = lta_output;
    op.params[2].tmpref.size = sizeof(lta_output);

    op.params[3].tmpref.buffer = lta_delta;
    op.params[3].tmpref.size = sizeof(lta_delta);
*/
    res = TEEC_InvokeCommand(&sess, MAKE_CMD,
                             &op, &origin);


    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE) failed 0x%x origin 0x%x",
         res, origin);
}

void forward_connected_layer_CA(float *net_input, int net_inputs, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_OUTPUT);
    op.params[0].tmpref.buffer = net_input;
    op.params[0].tmpref.size = sizeof(float) * net_inputs;
    op.params[1].value.a = net_train;

    op.params[2].tmpref.buffer = lta_output;
    op.params[2].tmpref.size = sizeof(float) * 103000;
    op.params[3].tmpref.buffer = lta_delta;
    op.params[3].tmpref.size = sizeof(float) * 103000;

    res = TEEC_InvokeCommand(&sess, FORWARD_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);

}

void backward_connected_layer_CA_addidion()
{

  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;


  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_OUTPUT,
                                   TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE);

   op.params[0].tmpref.buffer = lta_output;
   op.params[0].tmpref.size = sizeof(float) * 103000;
   op.params[1].tmpref.buffer = lta_delta;
   op.params[1].tmpref.size = sizeof(float) * 103000;
   op.params[2].tmpref.buffer = n_delta;
   op.params[2].tmpref.size = sizeof(float) * 103000;

/*
  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, TEEC_NONE,TEEC_NONE, TEEC_NONE);

  op.params[0].value.a = 1;
*/
   res = TEEC_InvokeCommand(&sess, BACKWARD_ADD_CMD,
                            &op, &origin);

   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(backward_add) failed 0x%x origin 0x%x",
        res, origin);
}



void backward_connected_layer_CA(float *net_input, int net_inputs, float *net_delta, int net_deltas, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_VALUE_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = net_input; // as lta.output
    op.params[0].tmpref.size = sizeof(float) * net_inputs * 100;
    op.params[1].tmpref.buffer = net_delta; // as n_delta
    op.params[1].tmpref.size = sizeof(float) * net_deltas * 100;
    op.params[2].value.a = net_train;

    res = TEEC_InvokeCommand(&sess, BACKWARD_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(backward) failed 0x%x origin 0x%x",
         res, origin);
}



void update_connected_layer_CA(update_args a)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[3];
    passint[0] = a.batch;
    passint[1] = a.adam;
    passint[2] = a.t;

    float passflo[6];
    passflo[0] = a.learning_rate;
    passflo[1] = a.momentum;
    passflo[2] = a.decay;
    passflo[3] = a.B1;
    passflo[4] = a.B2;
    passflo[5] = a.eps;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    res = TEEC_InvokeCommand(&sess, UPDATE_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(update) failed 0x%x origin 0x%x",
         res, origin);

}


void prepare_tee_session()
{
    TEEC_UUID uuid = TA_DARKNETP_UUID;
    uint32_t origin;
    TEEC_Result res;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

    /* Open a session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
         res, origin);
}

void terminate_tee_session()
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}



int main(int argc, char **argv)
{

    printf("Prepare session with the TA\n");
    prepare_tee_session();

    printf("Begin darknet\n");
    darknet_main(argc, argv);

    terminate_tee_session();
    return 0;
}
