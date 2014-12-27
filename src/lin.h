#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

#ifndef LIN
#define LIN

// multiply matrix A with matrix B, store result in matrix C
void mm1(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C);

// multiply matrix A with matrix B, store result in matrix C
void mm2(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C);

// multiply matrix A with matrix B, store result in matrix C
void mm3(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C);

// multiply matrix A with matrix B, store result in matrix C
void mm4(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C);

// multiply matrix gA with matrix gB, store result in matrix gC
void gmm1(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C);

#endif
