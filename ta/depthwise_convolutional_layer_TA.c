#include "depthwise_convolutional_layer_TA.h"
#include "batchnorm_layer_TA.h"

#include "utils_TA.h"
#include "gemm_TA.h"
#include "math_TA.h"
#include "blas_TA.h"
#include "im2col_TA.h"
#include "col2im_TA.h"
#include "darknet_TA.h"

#include <stdio.h>
#include <time.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

int depthwise_convolutional_out_height_TA(depthwise_convolutional_layer_TA l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int depthwise_convolutional_out_width_TA(depthwise_convolutional_layer_TA l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}


static size_t get_workspace_size(layer_TA l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}


depthwise_convolutional_layer_TA make_depthwise_convolutional_layer_TA_new(int batch, int h, int w, int c,int size, int stride, int padding, ACTIVATION_TA activation, int batch_normalize)
{
    int i;
	depthwise_convolutional_layer_TA l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL_TA;

    l.h = h;
    l.w = w;
    l.n = c;
	l.c = c;

    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(l.n*size*size, sizeof(float));
    l.weight_updates = calloc(l.n*size*size, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));

    l.nweights = l.n*size*size;
    l.nbiases = l.n;

    // float scale = 1./sqrt(size*size*c);
    float scale = ta_sqrt(2./(size*size*c));
    //scale = .02;
   //for(i = 0; i < c*size*size; ++i) l.weights[i] = 0.01*i;
    for(i = 0; i < l.n*l.size*l.size; ++i) l.weights[i] = scale*rand_normal_TA(0,1);
    int out_w = depthwise_convolutional_out_width_TA(l);
    int out_h = depthwise_convolutional_out_height_TA(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward_TA = forward_depthwise_convolutional_layer_TA_new;
    l.backward_TA = backward_depthwise_convolutional_layer_TA_new;
    l.update_TA = update_depthwise_convolutional_layer_TA_new;


    if(batch_normalize){
        l.scales = calloc(c, sizeof(float));
        l.scale_updates = calloc(c, sizeof(float));
        for(i = 0; i < c; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(c, sizeof(float));
        l.variance = calloc(c, sizeof(float));

        l.mean_delta = calloc(c, sizeof(float));
        l.variance_delta = calloc(c, sizeof(float));

        l.rolling_mean = calloc(c, sizeof(float));
        l.rolling_variance = calloc(c, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    return l;
}

void resize_depthwise_convolutional_layer_TA(depthwise_convolutional_layer_TA *l, int w, int h)
{
	l->w = w;
	l->h = h;
	int out_w = depthwise_convolutional_out_width_TA(*l);
	int out_h = depthwise_convolutional_out_height_TA(*l);

	l->out_w = out_w;
	l->out_h = out_h;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));
	if (l->batch_normalize) {
		l->x = realloc(l->x, l->batch*l->outputs * sizeof(float));
		l->x_norm = realloc(l->x_norm, l->batch*l->outputs * sizeof(float));
	}

	l->workspace_size = get_workspace_size(*l);
}


/*void test_depthwise_convolutional_layer()
{
#include "softmax_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"

    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
	float truth[] = { 0,0,1 };
	float delta[75] = {0 };


	int num_layer = 4;
	network net = make_network(num_layer);
	net.h=5;
	net.w=5;
	net.c=3;
	net.batch = 1;

	net.input = data;

	net.truth = truth;
	net.train = 1;



	depthwise_convolutional_layer depthwise_conv1 = make_depthwise_convolutional_layer(net.batch, net.h, net.w, net.c, 3, 1, 0, RELU, 0);
	avgpool_layer global_avgpool1 = make_avgpool_layer(net.batch, depthwise_conv1.out_w, depthwise_conv1.out_h, depthwise_conv1.n);
	softmax_layer softmax_1 = make_softmax_layer(net.batch, depthwise_conv1.n, 1);
	softmax_1.temperature = 1;
	cost_layer cost_1 = make_cost_layer(net.batch, depthwise_conv1.n, SSE, 1);


	net.layers[0] = depthwise_conv1;
	net.layers[1] = global_avgpool1;
	net.layers[2] = softmax_1;
	net.layers[3] = cost_1;
	net.workspace = calloc(1, 75);



	for (int i = 0; i < net.n; ++i) {
		net.index = i;
		layer l = net.layers[i];
		if (l.delta) {
			fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
		}
		l.forward(l, net);
		net.input = l.output;
		if (l.truth) {
			net.truth = l.output;
		}
	}
	calc_network_cost(net);

	fprintf(stderr, "**********************cost:%f ***************", *net.cost);




	fprintf(stderr, "**********************backward *************** \n");


	network orig = net;
	for (int i = net.n - 1; i >= 0; --i) {
		layer l = net.layers[i];
		if (i == 0) {
			//net = orig;
			net.input = data;
			net.delta = delta;
		}
		else {
			layer prev = net.layers[i - 1];
			net.input = prev.output;
			net.delta = prev.delta;
		}
		net.index = i;
		l.backward(l, net);
	}




    //forward_depthwise_convolutional_layer(l,net);
}*/



void add_bias_depthwise_TA(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias_depthwise_TA(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias_depthwise_TA(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array_TA(delta+size*(i+b*n), size);
        }
    }
}

void forward_depthwise_convolutional_layer_TA_new(depthwise_convolutional_layer_TA l, network_TA net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;

    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);

    int k = l.size*l.size;
    int n = out_h*out_w;

    for(int b = 0; b < l.batch; ++b){
		for (int c=0;c<l.c;c++)
		{
			float *aoffset = l.weights+c*l.size*l.size;
			float *boffset = net.workspace;
			float *coffset = l.output+c*l.out_h*l.out_w+b*l.n*l.out_h*l.out_w;
			float *intput_offset = net.input + c*l.h*l.w+ b*l.c*l.h*l.w;
			im2col_cpu_TA(intput_offset, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm_TA(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);

			//for (int i = 0; i < l.size*l.size; i++)
			//{
				//fprintf(stderr, "w %f \t", aoffset[i]);
			//}

		}
    }

/*
	for (int i = 0; i < l.batch*l.c*l.out_h*l.out_w; i++)
	{
		fprintf(stderr, "%f \t", l.output[i]);
	}
*/

    if(l.batch_normalize){
        forward_batchnorm_layer_TA(l, net);
    } else {
        add_bias_TA(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

	int m = l.n;
    activate_array_TA(l.output, m*n*l.batch, l.activation);
/*
	for (int i = 0; i < l.batch*l.c*l.out_h*l.out_w; i++)
	{
		fprintf(stderr, "%f \t", l.output[i]);
	}*/

}

void backward_depthwise_convolutional_layer_TA_new(depthwise_convolutional_layer_TA l, network_TA net)
{
    int i;
    int m = l.n;
    int n = l.size*l.size;
    int k = l.out_w*l.out_h;

    gradient_array_TA(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer_TA(l, net);
    } else {
        backward_bias_TA(l.bias_updates, l.delta, l.batch, l.n, k);
    }


	for (int b = 0; b < l.batch; ++b) {
		for (int c = 0; c<l.c; c++)
		{

			float *aoffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
			float *boffset = net.workspace;
			float *coffset = l.weight_updates + c*l.size*l.size;


			float *im = net.input + c*l.h*l.w + b*l.c*l.h*l.w;


			im2col_cpu_TA(im, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm_TA(0, 1, 1, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);

			if (net.delta) {
				aoffset = l.weights+ c*l.size*l.size;
				boffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
				coffset = net.workspace;

				gemm_TA(1, 0, n, k, 1, 1, aoffset, n, boffset, k, 0, coffset, k);

				col2im_cpu_TA(net.workspace, 1, l.h, l.w, l.size, l.stride, l.pad, net.delta + c*l.h*l.w + b*l.n*l.h*l.w);
			}


		}
	}


/*
	for (int i = 0; i < l.c*l.size*l.size; i++)
	{
		fprintf(stderr, "weight_updates:%f \t", l.weight_updates[i]);
	}
*/



}

void update_depthwise_convolutional_layer_TA_new(depthwise_convolutional_layer_TA l, update_args_TA a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c;
    axpy_cpu_TA(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu_TA(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu_TA(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu_TA(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu_TA(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu_TA(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu_TA(size, momentum, l.weight_updates, 1);
}
void denormalize_depthwise_convolutional_layer_TA(depthwise_convolutional_layer_TA l)
{
	int i, j;
	for (i = 0; i < l.n; ++i) {
		float scale = l.scales[i] / ta_sqrt(l.rolling_variance[i] + .00001);
		for (j = 0; j < l.size*l.size; ++j) {
			l.weights[i*l.size*l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}
