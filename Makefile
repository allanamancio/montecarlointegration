main:
	mpicc mcintegration.c -std=c11 -lm -pthread -o main

clean:
	$(RM) main mainsh *.o *~