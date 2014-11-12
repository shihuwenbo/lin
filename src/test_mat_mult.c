#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mat_mult.h"
#include "utils.h"

int main() {

    /** initialize A **/
    printf("init A\n");
    float *A = NULL;
    size_t nr_A = 5000;
    size_t nc_A = 5000;
    rand_mat(&A, nr_A, nc_A);
    /*
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nc_A; j++) {
            A[i*nc_A+j] = i*nc_A+j;
        }
    }
    print_mat(A, nr_A, nc_A);
    */

    /** initialize B **/
    printf("init B\n");
    float *B = NULL;
    size_t nr_B = 5000;
    size_t nc_B = 5000;
    rand_mat(&B, nr_B, nc_B);
    /*
    for(size_t i=0; i<nr_B; i++) {
        for(size_t j=0; j<nc_B; j++) {
            B[i*nc_B+j] = i*nc_B+j;
        }
    }
    print_mat(B, nr_B, nc_B);
    */

    /** initialize C **/
    printf("init C\n");
    float *C = NULL;
    size_t nr_C = 0;
    size_t nc_C = 0;
    
    // timing variables
    clock_t begin;
    clock_t end;
    float difftime_ms;

    /** test mat_mult_v1 **/
    float *C_true;
    printf("test mat_mult_v1\n");
    begin = clock();
    mat_mult_v1(A, nr_A, nc_A, B, nr_B, nc_B, &C_true, &nr_C, &nc_C);
    end = clock();
    difftime_ms = (float)(end-begin);
    difftime_ms /= (float)(CLOCKS_PER_SEC/1000.0); 
    printf("took %f ms\n", difftime_ms);
    //print_mat(C, nr_C, nc_C);
    /** test mat_mult_v1 **/

    /** test mat_mult_v2 **/
    printf("test mat_mult_v2\n");
    begin = clock();
    mat_mult_v2(A, nr_A, nc_A, B, nr_B, nc_B, &C, &nr_C, &nc_C);
    end = clock();
    difftime_ms = (float)(end-begin);
    difftime_ms /= (float)(CLOCKS_PER_SEC/1000.0); 
    printf("took %f ms\n", difftime_ms);
    //print_mat(C, nr_C, nc_C);
    free(C);
    /** test mat_mult_v2 **/

    /** test mat_mult_v3 **/
    printf("test mat_mult_v3\n");
    begin = clock();
    mat_mult_v3(A, nr_A, nc_A, B, nr_B, nc_B, &C, &nr_C, &nc_C);
    end = clock();
    difftime_ms = (float)(end-begin);
    difftime_ms /= (float)(CLOCKS_PER_SEC/1000.0); 
    printf("took %f ms\n", difftime_ms);
    //print_mat(C, nr_C, nc_C);
    free(C);
    /** test mat_mult_v3 **/

    /** test mat_mult_v2_gpu **/
    float *gA;
    cu_safe_falloc(&gA, nr_A*nc_A);
    memcpy_htod(gA, A, nr_A*nc_A);

    float *gB;
    cu_safe_falloc(&gB, nr_B*nc_B);
    memcpy_htod(gB, B, nr_B*nc_B);

    float *gC;
    printf("test mat_mult_gpu_v1\n");
    begin = clock();
    mat_mult_gpu_v1(gA, nr_A, nc_A, gB, nr_B, nc_B, &gC, &nr_C, &nc_C);
    end = clock();
    difftime_ms = (float)(end-begin);
    difftime_ms /= (float)(CLOCKS_PER_SEC/1000.0); 
    printf("took %f ms\n", difftime_ms);
    
    float *C_validate = (float*) safe_calloc(nr_C*nc_C, sizeof(float));
    memcpy_dtoh(C_validate, gC, nr_C*nc_C);
    float sum_diff = 0.0;
    for(size_t i=0; i<nr_C*nc_C; i++) {
        sum_diff += fabs(C_validate[i]-C_true[i]);
    }
    sum_diff = sum_diff/((float)nr_C*nc_C);
    printf("diff: %f\n", sum_diff);
    
    // free memory
    cu_free(gA);
    cu_free(gB);
    cu_free(gC);

    return 0;
}
