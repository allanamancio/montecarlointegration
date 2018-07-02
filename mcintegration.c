#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

//Constant
#define PI 3.14159265358979323846 //Pi

long long N; int k, M; //Arguments for integration
double _f_, _f2_; //Intermediary value of integration
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
	//Generate a random number in the interval (0, 0.5]
	return ( ((double) (rand() + 1)) / ( ((long long) RAND_MAX) + 1) ) * 0.5;
}

double f(int M_arg, int k_arg, double x_arg) {
	//Calculate the math function: 
	
	// sin([2M + 1]*pi*x) * cos(2*pi*k*x)
	// ----------------------------------
	//            sin(pi*x)

	return (sin((2*M_arg + 1)*PI*x_arg)*cos(2*PI*k_arg*x_arg))/sin(PI*x_arg);
}

void *thread_integration(void *num_cpus_arg) {
	int cpus = *((int *) num_cpus_arg);

	double x;
	_f_ = 0;
	_f2_ = 0;

	srand(time(NULL));
	for (int i = 0; i < N/cpus; i++) {
		x = x_random(); //Random number in (0, 0.5]
		double y = f(M, k, x);
		_f_ = _f_ + y;
		_f2_ = _f2_ + y*y;
	}

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
	if (abs(k) <= abs(M) && M < 0) result = -1;
	else result = 0;

	//Setting time measurement
	clock_t start, end;
	double time;

	// -----------------------------------------------------------------------------------------------------------------

	//Monte Carlos Integration with Distributed Computing Techniques
	MPI_Status status;

	int num_processes, this_process;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

	/*1. LOAD BALANCER WITH THE MINIMUM TIME*/

	/*2. ONE GPU AND ONE CPU THREAD*/

	/*3. T CPU THREADS*/
	int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
	pthread_t *id; if (num_cpus > 1) id = emalloc((num_cpus - 1)*sizeof(pthread_t));

	start = clock(); //Start of work
	for (int i = 0; i < num_cpus - 1; i++) { //T-1 threads
		if (pthread_create(&id[i], NULL, thread_integration, (void *) &num_cpus)) {
			fprintf(stderr, "ERROR: Thread not created.\n");
			exit(1);
		}
	}

	thread_integration((void *) &num_cpus); //Main thread

	for (int i = 0; i < num_cpus - 1; i++) {
		if (pthread_join(id[i], NULL)) {
			fprintf(stderr, "ERROR: Thread not joined.\n");
			exit(1);
		}
	}

	//Integration value
	_f_ = _f_/N;
	_f2_ = _f2_/N;

	result_1 = (_f_ + sqrt((_f2_ - _f_*_f_)/N));
	result_2 = (_f_ - sqrt((_f2_ - _f_*_f_)/N));
	end = clock(); //End of work

	//Print
	time = ((double) (end - start))/CLOCKS_PER_SEC;
	printf("Tempo na CPU com %d threads em segundos: %lf\n", num_cpus, time);
	printf("Erro no calculo com a soma: %lf\n", fabs(result_1 - result));
	printf("Erro no calculo com a subtracao: %lf\n\n", fabs(result_2 - result));

	/*4. ONE CPU THREAD*/
	num_cpus = 1;
	
	start = clock(); //Start of work
	thread_integration((void *) &num_cpus);

	_f_ = _f_/N;
	_f2_ = _f2_/N;

	result_1 = (_f_ + sqrt((_f2_ - _f_*_f_)/N));
	result_2 = (_f_ - sqrt((_f2_ - _f_*_f_)/N));
	end = clock(); //End of work

	//Print
	time = ((double) (end - start))/CLOCKS_PER_SEC;
	printf("Tempo sequencial em segundos: %lf\n", time);
	printf("Erro no calculo com a soma: %lf\n", fabs(result_1 - result));
	printf("Erro no calculo com a subtracao: %lf\n", fabs(result_2 - result));

	// -----------------------------------------------------------------------------------------------------------------

	//Finishing
	if (num_cpus > 1) free(id);
	MPI_Finalize();
}