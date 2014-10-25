#include "lin.h"

// safely allocate n_elem of elements, each with size sizeof_elem
void* safe_calloc(size_t n_elem, size_t sizeof_elem) {
    void* ptr = calloc(n_elem, sizeof_elem);
    if(!ptr) {
        size_t nbytes = n_elem*sizeof_elem;
        fprintf(stderr, "Error: Could not allocate ");
        fprintf(stderr, "%u bytes of memory!\n", (unsigned int)nbytes);
        exit(1);
    }
    return ptr;
}

// initialize a random matrix
void rand_mat(float **A, size_t nr_A, size_t nc_A) {
   
    // make randome seed
    srand(time(NULL));

    // allocate memory
    float *A_tmp = (float*) safe_calloc(nr_A*nc_A, sizeof(float));
    
    // initialize to random
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nc_A; j++) {
            float r = (float)rand()/(float)RAND_MAX;
            A_tmp[i*nc_A+j] = r;
        }
    }

    A[0] = A_tmp;
}

// initialize a identity matrix
void eye(float **A, size_t nr_A) {
    
    // allocate memory
    float *A_tmp = (float*) safe_calloc(nr_A*nr_A, sizeof(float));
    
    // initialize to random
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nr_A; j++) {
            if( i == j) {
                A_tmp[i*nr_A+j] = 1.0;
            }
            else {
                A_tmp[i*nr_A+j] = 0.0;
            }
        }
    }

    A[0] = A_tmp;
}

// print out a matrix
void print_mat(float *A, size_t nr_A, size_t nc_A) {
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nc_A; j++) {
            printf("%f ", A[i*nc_A+j]);
        }
        printf("\n");
    }
}

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
