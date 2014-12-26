#include <stdio.h>
#include <cuda.h>
#include "utils.h"

size_t block_size = 64;
size_t max_grid_x = 65535;

// kernel function for gmm1
__global__ void gmm1_kernel(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C);
