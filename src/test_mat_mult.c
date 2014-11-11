#include <stdio.h>
#include <stdlib.h>
#include "mat_mult.h"
#include "utils.h"

int main() {

    /** initialize A **/
    printf("init A\n");
    float *A = NULL;
    size_t nr_A = 4000;
    size_t nc_A = 3000;
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
    size_t nr_B = 3000;
    size_t nc_B = 4000;
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
    printf("test mat_mult_v1\n");
    begin = clock();
    mat_mult_v1(A, nr_A, nc_A, B, nr_B, nc_B, &C, &nr_C, &nc_C);
    end = clock();
    difftime_ms = (float)(end-begin);
    difftime_ms /= (float)(CLOCKS_PER_SEC/1000.0); 
    printf("took %f ms\n", difftime_ms);
    //print_mat(C, nr_C, nc_C);
    free(C);
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

    return 0;
}
