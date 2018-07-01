#include <stdio.h>
#include <stdlib.h>

long long N; int k, M; //Arguments for integration

void *integration(void *argument) {
	
}

int main(int argc, char **argv) {
	//Checking quantity of arguments
	if (argc != 4) {
		fprintf(stderr, "ERROR: Invalid number of arguments.\n");
		exit(1);
	}
	N = atoll(argv[1]); k = atoi(argv[2]); M = atoi(argv[3]); //Arguments

	//MPI
	MPI_Status status;

	int num_cpus, this_cpu;
	MP_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_cpus);
	MPI_Comm_rank(MPI_COMM_WORLD, &this_cpu);

	if (this_cpu == 0) {

	}
	else {

	}

	MPI_Finalize();
}