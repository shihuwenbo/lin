CC=gcc
COPTS=-O3 -Wall -std=c99

BINS=main

all: $(BINS)

# main
main: main.o lin.o
	$(CC) main.o lin.o -o main $(COPTS)

# main.o
main.o: main.c
	$(CC) -c main.c $(COPTS)

# lin.o
lin.o: lin.c lin.h
	$(CC) -c lin.c $(COPTS)

clobber:
	rm -f *.o

clean:
	rm -f *.o $(BINS)
