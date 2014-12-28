#include <stdio.h>
#include <cuda.h>
#include "utils.h"

// kernel function for gmm1
__global__ void gmm1_kernel(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C);

// kernel function for gmm2
__global__ void gmm2_kernel(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C,
        size_t ntcells_A, size_t ntcells_B, size_t ntcells_C,
        size_t tilesize_r, size_t tilesize_c, size_t tilesize_m,
        size_t ntiles_c, size_t ntiles_c_A, size_t ntiles_c_B,
        size_t ncells_A, size_t ncells_B, size_t ncells_C);
