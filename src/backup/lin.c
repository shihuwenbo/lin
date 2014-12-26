#include "lin.h"

// multiply matrix A with matrix B, store result in matrix C
void mat_mult_v1(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float **C, size_t *nr_C, size_t *nc_C) {

    // get dimension of C
    *nr_C = nr_A;
    *nc_C = nc_B;

    // allocate memory
    float *C_tmp = (float*) safe_calloc(nr_A*nc_B, sizeof(float));

    // make the computation
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nc_B; j++) {
            C_tmp[i*nc_B+j] = 0.0;
            for(size_t k=0; k<nc_A; k++) {
                C_tmp[i*nc_B+j] += A[i*nc_A+k]*B[k*nc_B+j];
            }
            
        }
    }

    C[0] = C_tmp;
}

// multiply matrix A with matrix B, store result in matrix C
void mat_mult_v2(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float **C, size_t *nr_C, size_t *nc_C) {

    // get dimension of C
    *nr_C = nr_A;
    *nc_C = nc_B;

    // allocate memory
    float *C_tmp = (float*) safe_calloc(nr_A*nc_B, sizeof(float));

    // make the computation
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nc_B; j++) {
            C_tmp[i*nc_B+j] = 0.0;
        }
        for(size_t k=0; k<nc_A; k++) {
            for(size_t j=0; j<nc_B; j++) {
                C_tmp[i*nc_B+j] += A[i*nc_A+k]*B[k*nc_B+j];
            }
        }
    }

    C[0] = C_tmp;
}

// multiply matrix A with matrix B, store result in matrix C
void mat_mult_v3(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float **C, size_t *nr_C, size_t *nc_C) {

    // get dimension of C
    *nr_C = nr_A;
    *nc_C = nc_B;

    // allocate memory
    float *C_tmp = (float*) safe_calloc(nr_A*nc_B, sizeof(float));

    // temporary storage for B's column
    float *B_tmp = (float*) safe_calloc(nr_B, sizeof(float));

    // make the computation
    for(size_t j=0; j<nc_B; j++) {
        for(size_t k=0; k<nr_B; k++) {
            B_tmp[k] = B[k*nc_B+j];
        }
        for(size_t i=0; i<nr_A; i++) {
            C_tmp[i*nc_B+j] = 0.0;
            for(size_t k=0; k<nc_A; k++) {
                C_tmp[i*nc_B+j] += A[i*nc_A+k]*B_tmp[k];
            }
        }
    }

    // clean up memory
    free(B_tmp);

    C[0] = C_tmp;
}
