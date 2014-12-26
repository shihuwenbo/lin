#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lin.h"
#include "utils.h"

int main() {

    // timing variables
    clock_t begin;
    clock_t end;

    // timing information
    float tmm1 = 0.0;
    float tmm2 = 0.0;
    float tmm3 = 0.0;
    float tgmm1 = 0.0;

    // error measure between h and d
    float err1 = 0.0;

    // initialize A and B
    float *A = NULL;
    size_t nr_A = 3000;
    size_t nc_A = 1000;
    rand_mat(&A, nr_A, nc_A);

    float *B = NULL;
    size_t nr_B = 1000;
    size_t nc_B = 4000;
    rand_mat(&B, nr_B, nc_B);

    // copy A and B to device
    float *gA;
    cu_safe_falloc(&gA, nr_A*nc_A);
    memcpy_htod(gA, A, nr_A*nc_A);

    float *gB;
    cu_safe_falloc(&gB, nr_B*nc_B);
    memcpy_htod(gB, B, nr_B*nc_B);

    // test mm1
    size_t nr_C1 = nr_A;
    size_t nc_C1 = nc_B;
    float *C1 = (float*) safe_calloc(nr_C1*nc_C1, sizeof(float));
    begin = clock();
    mm1(A, nr_A, nc_A, B, nr_B, nc_B, C1, nr_C1, nc_C1);
    end = clock();
    tmm1 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);
   
    // test mm2
    size_t nr_C2 = nr_A;
    size_t nc_C2 = nc_B;
    float *C2 = (float*) safe_calloc(nr_C2*nc_C2, sizeof(float));
    begin = clock();
    mm2(A, nr_A, nc_A, B, nr_B, nc_B, C2, nr_C2, nc_C2);
    end = clock();
    tmm2 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);

    // test mm2
    size_t nr_C3 = nr_A;
    size_t nc_C3 = nc_B;
    float *C3 = (float*) safe_calloc(nr_C3*nc_C3, sizeof(float));
    begin = clock();
    mm3(A, nr_A, nc_A, B, nr_B, nc_B, C3, nr_C3, nc_C3);
    end = clock();
    tmm3 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);

    // test gmm1
    size_t gnr_C1 = nr_A;
    size_t gnc_C1 = nc_B;
    float *gC1;
    cu_safe_falloc(&gC1, gnr_C1*gnc_C1);
    begin = clock();
    gmm1(gA, nr_A, nc_A, gB, nr_B, nc_B, gC1, gnr_C1, gnc_C1);
    end = clock();
    tgmm1 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);

    // validate gmm1
    float *vC1 = (float*) safe_calloc(gnr_C1*gnc_C1, sizeof(float));
    memcpy_dtoh(vC1, gC1, gnr_C1*gnc_C1);
    for(size_t i=0; i<gnr_C1*gnc_C1; i++) {
        float diff = vC1[i]-C1[i];
        err1 += diff*diff;
    }

    // print result
    printf("tmm1: %fns\n", tmm1/1e3);
    printf("tmm2: %fns\n", tmm2/1e3);
    printf("tmm3: %fns\n", tmm3/1e3);
    printf("tgmm1: %fns\n", tgmm1/1e3);
    printf("err1: %f\n", err1);
}
