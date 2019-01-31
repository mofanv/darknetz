#include "gemm_TA.h"
#include "utils_TA.h"
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <stdlib.h>
#include <stdio.h>
#include "math_ta.h"

float* gemm_TA(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc)
{
    C = gemm_cpu_TA( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
    return C;
}

float* gemm_nn_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            //float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
    return C;
}

float* gemm_nt_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{

    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            //float sum = 0;
            for(k = 0; k < K; ++k){
                //IMSG("A %d in %d; B %d in %d\n", i*lda+k, (M-1)*lda + K, j*ldb+k,(N-1)*ldb + K);
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
    return C;
}

float* gemm_tn_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            //float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
    return C;
}

float* gemm_tt_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            //float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
    return C;
}


float* gemm_cpu_TA(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        C = gemm_nn_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        C = gemm_tn_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        C = gemm_nt_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        C = gemm_tt_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    return C;
}
