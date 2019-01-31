#include <err.h>
#include <stdio.h>
#include <string.h>

#include "darknet.h"
#include "activations.h"
#include "cost_layer.h"

#include "main.h"

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* TEE resources */
TEEC_Context ctx;
TEEC_Session sess;

float *net_input;
float *net_delta;

void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

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

    res = TEEC_InvokeCommand(&sess, MAKE_CONNECTED_CMD,
                             &op, &origin);


    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE) failed 0x%x origin 0x%x",
         res, origin);
}

void forward_connected_layer_CA(float *net_input, int l_inputs, int net_batch, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

     float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
     for(int z=0; z<l_inputs*net_batch; z++){
         params0[z] = net_input[z];
     }
     int params1 = net_train;

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * l_inputs*net_batch;
    op.params[1].value.a = params1;

    res = TEEC_InvokeCommand(&sess, FORWARD_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}

void backward_connected_layer_CA_addidion()
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

  net_input = malloc(sizeof(float) * 102400);
  net_delta = malloc(sizeof(float) * 102400);

  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_OUTPUT,
                                   TEEC_NONE, TEEC_NONE);

   op.params[0].tmpref.buffer = net_input;
   op.params[0].tmpref.size = sizeof(float) * 102400;
   op.params[1].tmpref.buffer = net_delta;
   op.params[1].tmpref.size = sizeof(float) * 102400;

   res = TEEC_InvokeCommand(&sess, BACKWARD_ADD_CMD,
                            &op, &origin);
   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(backward_add) failed 0x%x origin 0x%x",
        res, origin);
}



void backward_connected_layer_CA(float *net_input, int l_inputs, int net_batch, float *net_delta, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;


    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_VALUE_INPUT, TEEC_NONE);

     float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
     float *params1 = malloc(sizeof(float)*l_inputs*net_batch);

     for(int z=0; z<l_inputs*net_batch; z++){
         params0[z] = net_input[z];
         params1[z] = net_delta[z];
     }

    op.params[0].tmpref.buffer = params0; // as lta.output
    op.params[0].tmpref.size = sizeof(float)*l_inputs*net_batch;
    op.params[1].tmpref.buffer = params1; // as n_delta
    op.params[1].tmpref.size = sizeof(float)*l_inputs*net_batch;
    op.params[2].value.a = net_train;

    res = TEEC_InvokeCommand(&sess, BACKWARD_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(backward) failed 0x%x origin 0x%x",
         res, origin);

   free(params0);
   free(params1);
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

void make_softmax_layer_CA(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[8];
    float passflo = temperature;
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = groups;
    passint[3] = w;
    passint[4] = h;
    passint[5] = c;
    passint[6] = spatial;
    passint[7] = noloss;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_VALUE_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].value.a = passflo;

    res = TEEC_InvokeCommand(&sess, MAKE_SOFTMAX_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(update) failed 0x%x origin 0x%x",
         res, origin);
}

void make_cost_layer_CA(int batch, int inputs, COST_TYPE cost_type, float scale, float ratio, float noobject_scale, float thresh)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[2];
    float passflo[4];
    char *passcost;

    passint[0] = batch;
    passint[1] = inputs;
    passflo[0] = scale;
    passflo[1] = ratio;
    passflo[2] = noobject_scale;
    passflo[3] = thresh;

    passcost = get_cost_string(cost_type);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    op.params[2].tmpref.buffer = passcost;
    op.params[2].tmpref.size = strlen(passcost);

    res = TEEC_InvokeCommand(&sess, MAKE_COST_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(update) failed 0x%x origin 0x%x",
         res, origin);
}


void net_truth_CA(float *net_truth, int net_truths, int net_batch)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    // allocate memory for transmitting truth
    float *params0 = malloc(sizeof(float) * net_truths * net_batch);

    for(int z=0; z<net_truths*net_batch; z++){
        params0[z] = net_truth[z];
    }

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float)*net_truths*net_batch;

    res = TEEC_InvokeCommand(&sess, NET_TRUTH_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(update) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}

void calc_network_loss_CA(int n, int batch)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int params0[2];
    params0[0] = n;
    params0[1] = batch;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(params0);

    res = TEEC_InvokeCommand(&sess, CALC_LOSS_CMD,
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
