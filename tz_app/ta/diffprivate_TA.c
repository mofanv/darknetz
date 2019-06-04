#include "math_TA.h"
#include "utils_TA.h"
#include "diffprivate_TA.h"

void *diff_private_SGD(float *input, int len_input)
{
    //initial parameters
    float bound = 4;
    float epsilon = 4;
    float delta = 0.00001;
    
    float sum = 0;
    for(int i=0; i<len_input; i++){
        sum += input[i] * input[i];
    }
    float squm = ta_sqrt(sum)/bound;
    
    if(squm < 1){
        squm = 1;
    }
    
    float sigma = ta_sqrt((2*bound*bound)/ (epsilon*epsilon) * ta_log(10, 1.25/delta));
    
    for(int i=0; i<len_input; i++){
        input[i] = input[i]/squm + rand_normal_TA(0.0f, sigma);
        //printf("sigma=%f,delta=%f,ta_log(10, 1.25/delta)=%f, rand_normal=%f\n",sigma,delta,ta_log(10, 1.25/delta),rand_normal_TA(0.0f, sigma));
    }
}
