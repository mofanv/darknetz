#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils_TA.h"
#include "math_TA.h"

float sum_array_TA(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    
    return sum;
}

float rand_uniform_TA(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }    
    return ((float)ta_rand() * (max - min)) + min;
}


float rand_normal_TA(float mu, float sigma){
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (float) X2);
    }
    
    do
    {
        U1 = -1 + (float) ta_rand () * 2;
        U2 = -1 + (float) ta_rand () * 2;
        W = ta_pow (U1, 2) + ta_pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = ta_sqrt ((-2 * ta_ln (W)) / W); //ta_log
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (float) X1);
}
