#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

// allocate memory on gpu
extern "C" void cu_safe_malloc(float **g_f, size_t n_elem,
        size_t sizeof_elem) {
    void *gptr;
    cudaError_t crc = cudaMalloc(&gptr, n_elem*sizeof_elem);
    if(crc) {
        printf("cudaMalloc Error=%d:%s\n", crc, cudaGetErrorString(crc));
        exit(1);
    }
    *g_f = (float*) gptr;
}

// free memory on gpu
extern "C" void cu_free(void *g_d) {
   cudaError_t crc = cudaFree(g_d);
   if (crc) {
      printf("cudaFree Error=%d:%s\n", crc, cudaGetErrorString(crc));
      exit(1);
   }
}
