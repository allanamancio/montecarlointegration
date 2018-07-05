main:
	nvcc -lm -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi -Xcompiler="-pthread" mcintegration.cu -o main

clean:
	$(RM) main mainsh *.o *~