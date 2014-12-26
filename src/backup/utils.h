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

// allocate float memory on gpu
void cu_safe_falloc(float **g_f, size_t n_elem);

// free memory on gpu
void cu_free(void *g_d);

// copy from cpu space f to gpu space g_f
void memcpy_htod(float *g_f, float *f, size_t n_elem);

// copy from gpu space g_f to cpu space f
void memcpy_dtoh(float *f, float *g_f, size_t n_elem);
