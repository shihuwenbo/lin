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
    float tmm4 = 0.0;
    float tgmm1 = 0.0;
    float tgmm2 = 0.0;

    // error measure between h and d
    float err_gmm1 = 0.0;
    float err_gmm2 = 0.0;
    float err_mm4 = 0.0;

    // initialize A and B
    float *A = NULL;
    size_t nr_A = 1024;
    size_t nc_A = 1024;
    rand_mat(&A, nr_A, nc_A);

    float *B = NULL;
    size_t nr_B = 1024;
    size_t nc_B = 1024;
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

    // test mm3
    size_t nr_C3 = nr_A;
    size_t nc_C3 = nc_B;
    float *C3 = (float*) safe_calloc(nr_C3*nc_C3, sizeof(float));
    begin = clock();
    mm3(A, nr_A, nc_A, B, nr_B, nc_B, C3, nr_C3, nc_C3);
    end = clock();
    tmm3 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);

    // test mm4
    size_t nr_C4 = nr_A;
    size_t nc_C4 = nc_B;
    float *C4 = (float*) safe_calloc(nr_C4*nc_C4, sizeof(float));
    begin = clock();
    mm4(A, nr_A, nc_A, B, nr_B, nc_B, C4, nr_C4, nc_C4);
    end = clock();
    tmm4 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);
    for(size_t i=0; i<nr_C4*nc_C4; i++) {
        float diff = C4[i]-C1[i];
        err_mm4 += diff*diff;
    }

    // test gmm1
    size_t gnr_C1 = nr_A;
    size_t gnc_C1 = nc_B;
    float *gC1;
    cu_safe_falloc(&gC1, gnr_C1*gnc_C1);
    begin = clock();
    gmm1(gA, nr_A, nc_A, gB, nr_B, nc_B, gC1, gnr_C1, gnc_C1);
    end = clock();
    tgmm1 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);
    float *vC1 = (float*) safe_calloc(gnr_C1*gnc_C1, sizeof(float));
    memcpy_dtoh(vC1, gC1, gnr_C1*gnc_C1);
    for(size_t i=0; i<gnr_C1*gnc_C1; i++) {
        float diff = vC1[i]-C1[i];
        err_gmm1 += diff*diff;
    }

    // test gmm2
    size_t gnr_C2 = nr_A;
    size_t gnc_C2 = nc_B;
    float *gC2;
    cu_safe_falloc(&gC2, gnr_C2*gnc_C2);
    begin = clock();
    gmm2(gA, nr_A, nc_A, gB, nr_B, nc_B, gC2, gnr_C2, gnc_C2);
    end = clock();
    tgmm2 = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);
    float *vC2 = (float*) safe_calloc(gnr_C2*gnc_C2, sizeof(float));
    memcpy_dtoh(vC2, gC2, gnr_C2*gnc_C2);
    for(size_t i=0; i<gnr_C2*gnc_C2; i++) {
        float diff = vC2[i]-C1[i];
        err_gmm2 += diff*diff;
    }
    
    // print result
    printf("tmm1: %fs\n", tmm1/1e9);
    printf("tmm2: %fs\n", tmm2/1e9);
    printf("tmm3: %fs\n", tmm3/1e9);
    printf("tgmm1: %fs\n", tgmm1/1e9);
    printf("tgmm2: %fs\n", tgmm2/1e9);
    printf("err_gmm1: %f\n", err_gmm1);
    printf("err_gmm2: %f\n", err_gmm2);
    printf("err_mm4: %f\n", err_mm4);
}
