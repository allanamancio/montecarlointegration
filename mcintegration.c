#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

long long N; int k, M; //Arguments for integration
double result_1, result_2; //Value of integration

void *emalloc(size_t size) {
	void *memory = malloc(size);

	if (!memory) {
		fprintf(stderr, "ERROR: Failed to malloc.\n");
		exit(1);
	}

	return memory;
}

double x_random() {
	//Generate a random number in the interval (0, 1.5]
	return ( ((double) (rand() + 1)) / ( ((long long) RAND_MAX) + 1) ) * 1.5;
}

double f(int M_arg, int k_arg, double x_arg) {
	//Calculate the math function: sin([2M + 1]*pi*x)*cos(2*pi*k*x)/sin(pi*x)
	return (sin((2*M_arg + 1)*M_PI*x_arg)*cos(2*M_PI*k_arg*x_arg))/sin(M_PI*x_arg);
}

void *integration(void *argument) {
	double x;
	double _f_;
	double _f2_;

	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		x = x_random(); //Random number in (0, 1.5]
		double y = f(M, k, x);
		_f_ = y;
		_f2_ = y*y;
	}

	_f_ = _f_/N;
	_f2_ = _f2_/N;

	result_1 = 1.5 * (_f_ + sqrt((_f2_ - _f_*_f_)/N));
	result_2 = 1.5 * (_f_ - sqrt((_f2_ - _f_*_f_)/N));

	return NULL;
}

int main(int argc, char **argv) {
	//Checking quantity of arguments
	if (argc != 4) {
		fprintf(stderr, "ERROR: Invalid number of arguments.\n");
		exit(1);
	}
	N = atoll(argv[1]); k = atoi(argv[2]); M = atoi(argv[3]); //Arguments

	//Integration result algebrically
	int result;
	if (abs(k) <= abs(M) && M >= 0) result = 1;
	if (abs(k) <= abs(M) && M >= 0) result = 1;
	else result = 0;

	//Monte Carlos Integration with Distributed Computing Techniques
	MPI_Status status;

	int num_cpus, this_cpu;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_cpus);
	MPI_Comm_rank(MPI_COMM_WORLD, &this_cpu);

	//1. Load Balancer with the Minimum Time
	if (this_cpu == 0) {

	}
	else {

	}

	MPI_Finalize();

	//2. One GPU and one CPU thread

	//3. T CPU threads
	if (this_cpu == 0) {
		
	}
	else {
		pthread_t *id = emalloc(num_cpus*sizeof(pthread_t));
		for (int i = 0; i < num_cpus; i++) {
			if (pthread_create(&id[i], NULL, integration, NULL)) {
				fprintf(stderr, "ERROR: Thread not created.\n");
				exit(1);
			}
		}

		for (int i = 0; i < num_cpus; i++) {
			if (pthread_join(id[i], NULL)) {
				fprintf(stderr, "ERROR: Thread not joined.\n");
				exit(1);
			}
		}
	}

	//4. One CPU thread
	clock_t start, end;
	double time;

	start = clock();
	integration(NULL); //Work
	end = clock();

	//Print
	time = ((double) (end - start))/CLOCKS_PER_SEC;
	printf("Tempo sequencial em segundos: %lf\n", time);
	printf("Erro no calculo com a soma: %lf\n", fabs(result_1 - result));
	printf("Erro no calculo com a subtracao: %lf\n", fabs(result_2 - result));
}