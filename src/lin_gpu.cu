#include "lin_gpu.h"

size_t block_size = 128;
size_t max_grid_x = 65535;

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

// use compute A*B using tiling
extern "C" void gmm2(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C) {

    // define tile size
    size_t tilesize_r = 32;
    size_t tilesize_c = 32;
    size_t tilesize_m = 32;

    // check for small matrix conditions
    tilesize_r = tilesize_r>nr_C ? nr_C : tilesize_r;
    tilesize_c = tilesize_c>nc_C ? nc_C : tilesize_c;
    tilesize_m = tilesize_m>nc_A ? nc_A : tilesize_m;
    
    // compute the number of tiles
    size_t ntiles_r = (nr_C-1)/tilesize_r+1;
    size_t ntiles_c = (nc_C-1)/tilesize_c+1;
    size_t ntiles = ntiles_r*ntiles_c;

    // compute number of col/row tiles for A/B
    size_t ntiles_c_A = (nc_A-1)/tilesize_m+1;
    size_t ntiles_c_B = (nc_B-1)/tilesize_c+1;

    // number of cells in a tile
    size_t ntcells_C = tilesize_r*tilesize_c;
    size_t ntcells_A = tilesize_r*tilesize_m;
    size_t ntcells_B = tilesize_m*tilesize_c;

    // number of cells
    size_t ncells_C = nr_C*nc_C;
    size_t ncells_A = nr_A*nc_A;
    size_t ncells_B = nr_B*nc_B;

    // allocate shared memory
    size_t nshmem = ntcells_C+ntcells_A+ntcells_B;
    nshmem *= sizeof(float);

    // compute how many threads and blocks are needed
    size_t num_block = ntiles;

    // compute grid dimension
    size_t num_grid_y = (num_block-1)/max_grid_x+1;
    size_t num_grid_x = num_block<max_grid_x ? num_block : max_grid_x;

    // launch kernel
    dim3 dimBlock(block_size, 1);
    dim3 dimGrid(num_grid_x, num_grid_y);
    gmm2_kernel<<<dimGrid,dimBlock,nshmem>>>(
        gA, nr_A, nc_A,
        gB, nr_B, nc_B, gC, nr_C, nc_C,
        ntcells_A, ntcells_B, ntcells_C,
        tilesize_r, tilesize_c, tilesize_m,
        ntiles_c, ntiles_c_A, ntiles_c_B,
        ncells_A, ncells_B, ncells_C
    );

    // check kernel result
    cudaThreadSynchronize();
    cudaError_t crc = cudaGetLastError();
    if(crc) {
        printf("emptyKernel error=%d:%s\n", crc, cudaGetErrorString(crc));
        exit(1);
    }
}

// kernel function for mat_mult_gpu_v1
__global__ void gmm2_kernel(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float *gC, size_t nr_C, size_t nc_C,
        size_t ntcells_A, size_t ntcells_B, size_t ntcells_C,
        size_t tilesize_r, size_t tilesize_c, size_t tilesize_m,
        size_t ntiles_c, size_t ntiles_c_A, size_t ntiles_c_B,
        size_t ncells_A, size_t ncells_B, size_t ncells_C) {

    // get index for tile
    size_t i = blockIdx.x+gridDim.x*blockIdx.y;

    // get block and thread information
    size_t blksz = blockDim.x;
    size_t tidx = threadIdx.x;

    // get shared memory
    extern __shared__ float shmem[];
    float *ptra = &shmem[0];
    float *ptrb = &shmem[ntcells_A];
    float *ptrc = &shmem[ntcells_A+ntcells_B];

    // initialize shared memory
    #pragma unroll
    for(size_t ii=tidx; ii<ntcells_C; ii+=blksz) {
        ptrc[ii] = 0.0;
    }

    // get tile r and tile c
    size_t ir = i/ntiles_c;
    size_t ic = i%ntiles_c;
        
    // check boundary condition
    size_t ntcbm1 = ntiles_c_B-1;
    size_t ncolb = ic==ntcbm1 ? (nc_B-ntcbm1*tilesize_c) : tilesize_c;
    size_t ncolc = ncolb;

    // iterate through corresponding tiles for A and B
    for(size_t j=0; j<ntiles_c_A; j++) {

        // check boundary condition
        size_t ntcam1 = ntiles_c_A-1;
        size_t ncola = j==ntcam1 ? (nc_A-ntcam1*tilesize_m) : tilesize_m;
        
        // load tiles for A into shared memory
        for(size_t ii=tidx; ii<ntcells_A; ii+=blksz) {

            // within cell idx
            size_t tr = ii/ncola;
            size_t tc = ii%ncola;
           
            // absolute A index
            size_t aidx = (ir*tilesize_r+tr)*nc_A;
            aidx += j*tilesize_m+tc;
            
            // load to shared memory
            if(aidx < ncells_A) {
                ptra[tr*ncola+tc] = gA[aidx];
            }
        }

        // load tiles for B into shared memory
        for(size_t ii=tidx; ii<ntcells_B; ii+=blksz) {
            
            // within tile idx
            size_t tr = ii/ncolb;
            size_t tc = ii%ncolb;
            
            // absolute B index
            size_t bidx = (j*tilesize_m+tr)*nc_B;
            bidx += ic*tilesize_c+tc;

            // load to shared memory
            if(bidx < ncells_B) {
                ptrb[tr*ncolb+tc] = gB[bidx];
            }
        }
        __syncthreads();

        // make mat mult in shared memory
        for(size_t ii=tidx; ii<tilesize_r*ncolc; ii+=blksz) {
        
            // within tile idex
            size_t tr = ii/ncolc;
            size_t tc = ii%ncolc;

            // vector dot product
            #pragma unroll
            for(size_t jj=0; jj<ncola; jj++){
                float prod = ptra[tr*ncola+jj]*ptrb[jj*ncolb+tc];
                ptrc[tr*ncolc+tc] += prod;
            }
        }
    }

    // write from shared memory to global memory
    #pragma unroll
    for(size_t ii=tidx; ii<tilesize_r*ncolc; ii+=blksz) {
        
        // within tile idx
        size_t tr = ii/ncolc;
        size_t tc = ii%ncolc;
            
        // absolute C index
        size_t cidx = (ir*tilesize_r+tr)*nc_C;
        cidx += ic*tilesize_c+tc;
       
        // write to global memory
        if(cidx < ncells_C) {
            gC[cidx] = ptrc[tr*ncolc+tc];
        }
    }
}
