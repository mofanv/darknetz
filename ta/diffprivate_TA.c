#include "math_TA.h"
#include "utils_TA.h"
#include "diffprivate_TA.h"

float *diff_private(float *input, int len_input, float bound, float epsilon)
{
    float sum = 0;
    for(int i=0; i<len_input; i++){
        sum += input[i] * input[i];
    }
    float squm = ta_sqrt(sum)/bound;
    
    if(squm < 1){
        squm = 1;
    }
    
    float sigma = (2*bound*bound)/ (epsilon*epsilon) * ta_log(10, 5/(4*epsilon));
    
    float *output = malloc(len_input * sizeof(float));
    for(int i=0; i<len_input; i++){
        output[i] = input[i]/squm + rand_normal_TA(0.0f, sigma);
    }
    
    return output;
}
