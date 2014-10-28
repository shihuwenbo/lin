#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// safely allocate n_elem of elements, each with size sizeof_elem
void* safe_calloc(size_t n_elem, size_t sizeof_elem);

// initialize a random matrix
void rand_mat(float **A, size_t nr_A, size_t nc_A);

// initialize a identity matrix
void eye(float **A, size_t nr_A);

// print out a matrix
void print_mat(float *A, size_t nr_A, size_t nc_A);

// multiply matrix A with matrix B, store result in matrix C
void mat_mult_v1(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float **C, size_t *nr_C, size_t *nc_C);

// multiply matrix A with matrix B, store result in matrix C
void mat_mult_v2(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float **C, size_t *nr_C, size_t *nc_C);

// multiply matrix A with matrix B, store result in matrix C
void mat_mult_v3(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float **C, size_t *nr_C, size_t *nc_C);
