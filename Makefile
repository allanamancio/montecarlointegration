main:
	nvcc -lm -Xcompiler="-pthread" mcintegration.cu -o main

program: mpi.o cuda.o
	mpicc -lcudart -L/usr/local/cuda-9.1/lib64 mpi.o cuda.o -o program -lm -lstdc++

cuda.o: cuda.cu
	nvcc -c cuda.cu

mpi.o: mpi.c
	mpicc -c mpi.c

clean:
	$(RM) main mainsh *.o *~