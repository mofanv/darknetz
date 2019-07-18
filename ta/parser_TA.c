#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "darknetp_ta.h"
#include "network_TA.h"
#include <parser_TA.h>
#include <blas_TA.h>



void transpose_matrix_TA(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}


void load_weights_TA(float *vec, int length, int layer_i, char type, int transpose)
{
    layer_TA l = netta.layers[layer_i];

    if(type == 'b'){
        copy_cpu_TA(length, vec, 1, l.biases, 1);
        
        for(int z=0; z<length; z++){
            char char0[20];
            ftoa(vec[z], char0, 5);
            IMSG("z=%d, b=%s || ", z, char0);
            
            char char1[20];
            ftoa(l.biases[z], char1, 5);
            IMSG("z=%d, b=%s || ", z, char1);
        }
    }
    else if(type == 'w'){
        copy_cpu_TA(length, vec, 1, l.weights, 1);
        
        for(int z=0; z<length; z++){
            char char0[20];
            ftoa(vec[z], char0, 5);
            IMSG("z=%d, w=%s || ", z, char0);
            
            char char1[20];
            ftoa(l.weights[z], char1, 5);
            IMSG("z=%d, w=%s || ", z, char1);
        }
    }
    else if(type == 's'){
        copy_cpu_TA(length, vec, 1, l.scales, 1);
    }
    else if(type == 'm'){
        copy_cpu_TA(length, vec, 1, l.rolling_mean, 1);
    }
    else if(type == 'v'){
        copy_cpu_TA(length, vec, 1, l.rolling_variance, 1);
    }
    
    
    if(l.type == CONVOLUTIONAL_TA || l.type == DECONVOLUTIONAL_TA){
        if(l.flipped && type == 'w'){
            transpose_matrix_TA(l.weights, l.c*l.size*l.size, l.n);
        }
    }
    else if(l.type == CONNECTED_TA){
        if(transpose && type == 'w'){
            transpose_matrix_TA(l.weights, l.inputs, l.outputs);
        }
    }
}
