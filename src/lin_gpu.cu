#include "lin_gpu.h"

// first version of matrix multiplication
// use nr_C*nc_C threads to compute A*B
extern "C" void gmm1(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C) {

    // compute how many threads and blocks are needed
    size_t num_cell = nr_A*nc_B;
    size_t num_block = (num_cell-1)/block_size+1;

    // compute grid dimension
    size_t num_grid_y = (num_block-1)/max_grid_x+1;
    size_t num_grid_x = num_block < max_grid_x ? num_block : max_grid_x;

    // launch kernel
    dim3 dimBlock(block_size, 1);
    dim3 dimGrid(num_grid_x, num_grid_y);
    gmm1_kernel<<<dimGrid,dimBlock>>>(gA, nr_A, nc_A,
            gB, nr_B, nc_B, gC, nr_C, nc_C);

    // check kernel result
    cudaThreadSynchronize();
    cudaError_t crc = cudaGetLastError();
    if(crc) {
        printf("emptyKernel error=%d:%s\n", crc, cudaGetErrorString(crc));
        exit(1);
    }
}

// kernel function for mat_mult_gpu_v1
__global__ void gmm1_kernel(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C) {

    // get absoluate idx of thread
    size_t j = threadIdx.x+blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);

    // check if j is within range
    if(j < nr_C*nc_C) {
        
        // obtain row and column of the cell thread j to compute
        size_t r = j / nc_C;
        size_t c = j % nc_C;

        // compute the inner product of r-th row of A and c-th column of B
        float val = 0.0;
        for(size_t i=0; i<nc_A; i++) {
            val += gA[r*nc_A+i]*gB[i*nc_B+c];
        }

        // save results
        gC[j] = val;
    }
}
