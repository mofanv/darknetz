#ifndef DARKNET_API_TA
#define DARKNET_API_TA
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#define SECRET_NUM_TA -1234
extern int gpu_index;

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree_TA;

typedef enum{
    LOGISTIC_TA, RELU_TA, RELIE_TA, LINEAR_TA, RAMP_TA, TANH_TA, PLSE_TA, LEAKY_TA, ELU_TA, LOGGY_TA, STAIR_TA, HARDTAN_TA, LHTAN_TA, SELU_TA
} ACTIVATION_TA;

typedef enum{
    CONVOLUTIONAL_TA,
    DECONVOLUTIONAL_TA,
    CONNECTED_TA,
    MAXPOOL_TA,
    SOFTMAX_TA,
    DETECTION_TA,
    DROPOUT_TA,
    CROP_TA,
    ROUTE_TA,
    COST_TA,
    NORMALIZATION_TA,
    AVGPOOL_TA,
    LOCAL_TA,
    SHORTCUT_TA,
    ACTIVE_TA,
    RNN_TA,
    GRU_TA,
    LSTM_TA,
    CRNN_TA,
    BATCHNORM_TA,
    NETWORK_TA,
    XNOR_TA,
    REGION_TA,
    YOLO_TA,
    ISEG_TA,
    REORG_TA,
    UPSAMPLE_TA,
    LOGXENT_TA,
    L2NORM_TA,
    BLANK_TA
} LAYER_TYPE_TA;

typedef enum{
    SSE_TA, MASKED_TA, L1_TA, SEG_TA, SMOOTH_TA, WGAN_TA
} COST_TYPE_TA;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args_TA;

struct network_TA;
typedef struct network_TA network_TA;

struct layer_TA;
typedef struct layer_TA layer_TA;

struct layer_TA{
    LAYER_TYPE_TA type;
    ACTIVATION_TA activation;
    COST_TYPE_TA cost_type;
    void (*forward_TA)   (struct layer_TA, struct network_TA);
    void (*backward_TA)  (struct layer_TA, struct network_TA);
    void (*update_TA)    (struct layer_TA, update_args_TA);

    int netnum;

    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu;

    float * binary_input;

    struct layer_TA *input_layer;
    struct layer_TA *self_layer;
    struct layer_TA *output_layer;

    struct layer_TA *reset_layer;
    struct layer_TA *update_layer;
    struct layer_TA *state_layer;

    struct layer_TA *input_gate_layer;
    struct layer_TA *state_gate_layer;
    struct layer_TA *input_save_layer;
    struct layer_TA *state_save_layer;
    struct layer_TA *input_state_layer;
    struct layer_TA *state_state_layer;

    struct layer_TA *input_z_layer;
    struct layer_TA *state_z_layer;

    struct layer_TA *input_r_layer;
    struct layer_TA *state_r_layer;

    struct layer_TA *input_h_layer;
    struct layer_TA *state_h_layer;

    struct layer_TA *wz;
    struct layer_TA *uz;
    struct layer_TA *wr;
    struct layer_TA *ur;
    struct layer_TA *wh;
    struct layer_TA *uh;
    struct layer_TA *uo;
    struct layer_TA *wo;
    struct layer_TA *uf;
    struct layer_TA *wf;
    struct layer_TA *ui;
    struct layer_TA *wi;
    struct layer_TA *ug;
    struct layer_TA *wg;

    tree_TA *softmax_tree;

    size_t workspace_size;
};

void free_layer_TA(layer_TA);

typedef enum {
    CONSTANT_TA, STEP_TA, EXP_TA, POLY_TA, STEPS_TA, SIG_TA, RANDOM_TA
} learning_rate_policy_TA;

typedef struct network_TA{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer_TA *layers;
    float *output;
    learning_rate_policy_TA policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree_TA *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

    size_t workspace_size;

} network_TA;

#endif
