#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>

void mycal(long long N, int k, int M);

int main(int argc, char *argv[]){
	int N,k,M;
	N = atoll(argv[1]); k = atoi(argv[2]); M = atoi(argv[3]);

	int myid, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	mycal(N, k, M);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}