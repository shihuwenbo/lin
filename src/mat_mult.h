#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

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
