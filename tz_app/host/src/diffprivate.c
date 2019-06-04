#include "utils.h"
#include "math.h"
#include "diffprivate.h"

void diff_private_func(float *input, int len_input)
{
    //initial parameters
    float bound = 4;
    float epsilon = 4;
    float delta = 0.00001;
    
    float sum = 0;
    for(int i=0; i<len_input; i++){
        sum += pow(input[i],2);
    }
    float squm = sqrt(sum)/bound;
    
    if(squm < 1){
        squm = 1;
    }
    
    float sigma = sqrt((2*pow(bound,2))/(pow(epsilon,2)) * log(1.25/delta));
    
    for(int i=0; i<len_input; i++){
        input[i] = input[i]/squm + rand_normal_ms(0.0f, sigma);
        //printf("diffinput=%f\n",input[i]);
        //printf("sigma=%f,delta=%f,ta_log(10, 1.25/delta)=%f, rand_normal=%f\n",sigma,delta,log(1.25/delta),rand_normal_TA(0.0f, sigma));
    }
}
