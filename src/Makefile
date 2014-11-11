CC=gcc
COPTS=-O3 -Wall -std=c99

NVCC=nvcc
NVLIBS=-L/usr/local/cuda/lib64 -lcuda -lcudart
NVINCLUDES=-I/usr/local/cuda/include
NVOPTS=-O3 -arch sm_20

BINS=test_mat_mult

all: $(BINS)

# main
test_mat_mult: test_mat_mult.o mat_mult.o mat_mult_gpu.o utils.o utils_gpu.o
	$(CC) test_mat_mult.o mat_mult.o mat_mult_gpu.o utils.o utils_gpu.o \
	    -o test_mat_mult $(COPTS) $(NVLIBS)

# main.o
test_mat_mult.o: test_mat_mult.c
	$(CC) -c test_mat_mult.c $(COPTS)

# mat_mult.o
mat_mult.o: mat_mult.c mat_mult.h
	$(CC) -c mat_mult.c $(COPTS)

# mat_mult_gpu.o
mat_mult_gpu.o: mat_mult_gpu.cu mat_mult_gpu.h
	$(NVCC) -c mat_mult_gpu.cu $(NVINCLUDES) $(NVOPTS)
	
# utils.o
utils.o: utils.c utils.h
	$(CC) -c utils.c $(COPTS)

# utils_gpu.o
utils_gpu.o: utils_gpu.cu utils.h
	$(NVCC) -c utils_gpu.cu $(NVINCLUDES) $(NVOPTS)

clobber:
	rm -f *.o

clean:
	rm -f *.o $(BINS)
