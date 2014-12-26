CC=gcc
COPTS=-O3 -Wall -std=c99

NVCC=nvcc
NVLIBS=-L/u/local/cuda/current/lib64 -lcuda -lcudart
NVINCLUDES=-I/usr/local/cuda/include
NVOPTS=-O3 -arch sm_20

BINS=test

all: $(BINS)

# main
test: test.o lin.o lin_gpu.o utils.o utils_gpu.o
	$(CC) test.o lin.o lin_gpu.o utils.o utils_gpu.o \
	    -o test $(COPTS) $(NVLIBS)

# main.o
test.o: test.c
	$(CC) -c test.c $(COPTS)

# lin.o
lin.o: lin.c lin.h
	$(CC) -c lin.c $(COPTS)

# lin_gpu.o
lin_gpu.o: lin_gpu.cu lin_gpu.h
	$(NVCC) -c lin_gpu.cu $(NVINCLUDES) $(NVOPTS)
	
# utils.o
utils.o: utils.c utils.h
	$(CC) -c utils.c $(COPTS)

# utils_gpu.o
utils_gpu.o: utils_gpu.cu utils.h
	$(NVCC) -c utils_gpu.cu $(NVINCLUDES) $(NVOPTS)

clean:
	rm -f *.o $(BINS)
