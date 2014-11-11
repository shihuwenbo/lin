CC=gcc
COPTS=-O3 -Wall -std=c99

BINS=test_mat_mult

all: $(BINS)

# main
test_mat_mult: test_mat_mult.o mat_mult.o utils.o
	$(CC) test_mat_mult.o mat_mult.o utils.o -o test_mat_mult $(COPTS)

# main.o
test_mat_mult.o: test_mat_mult.c
	$(CC) -c test_mat_mult.c $(COPTS)

# mat_mult.o
mat_mult.o: mat_mult.c mat_mult.h
	$(CC) -c mat_mult.c $(COPTS)

# utils.o
utils.o: utils.c utils.h
	$(CC) -c utils.c $(COPTS)

clobber:
	rm -f *.o

clean:
	rm -f *.o $(BINS)
