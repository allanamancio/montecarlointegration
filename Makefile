main:
	nvcc -lm -Xcompiler="-pthread" mcintegration.cu -o main

clean:
	$(RM) main mainsh *.o *~