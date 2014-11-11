#include "mat_mult_gpu.h"

extern "C" void mat_mult_gpu(float *gA, size_t nr_A, size_t nc_A,
        float *gB, size_t nr_B, size_t nc_B,
        float **gC, size_t *nr_C, size_t *nc_C) {
    printf("here\n");
   int ngx, ngy;
   ngx  = nblock_size < 32768 ? nblock_size : 32768;
   ngy = (ngrid_size - 1)/ngx + 1;
   dim3 dimBlock(nblock_size,1);
   dim3 dimGrid(ngx,ngy);
   emptyKernel<<<dimGrid,dimBlock>>>();
   
   cudaThreadSynchronize();
   
   cudaError_t crc = cudaGetLastError();
   if(crc) {
      printf("emptyKernel error=%d:%s\n", crc, cudaGetErrorString(crc));
      exit(1);
   }
}
