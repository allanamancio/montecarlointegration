#include <stdio.h>
#include <stdlib.h>
// #include <mpi.h> //*MPI
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
// #include <cuda_profiler_api.h>
// #include <curand_kernel.h>

//Constant
#define PI 3.14159265358979323846 //Pi

//Global variables
long long N; int k, M; //Arguments for integration
double _f_, _f2_; //Intermediary value of integration

//Semaphore
pthread_mutex_t sum_fs_;

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

/*__global__ void cuda_integration(double *_fcuda_, double *_f2cuda_, int _M, int _k, long long _N, unsigned long seed) {
	curandState state;
	curand_init(seed, 0, 0, &state);

	double x, y;
	*_fcuda_ = 0;
	*_f2cuda_ = 0;

	for (int i = 0; i < _N; i++) {
		x = ( ((double) (curand(&state) + 1)) / ( ((long long) UINT_MAX) + 1) ) * 0.5;
		y = (sin((2 * _M + 1) * PI * x) * cos(2 * PI * _k * x)) / sin(PI * x);
		*_fcuda_ += y;
		*_f2cuda_ += y * y;
	}
}*/

void *thread_integration(void *num_cpus_arg) {
	int cpus = *((int *) num_cpus_arg);

	double x, y;
	double _fpart_ = 0; //Partial _f_
	double _f2part_ = 0; //Partial _f2_

	for (int i = 0; i < N/cpus; i++) {
		x = x_random(); //Random number in (0, 0.5]
		y = f(M, k, x);
		_fpart_ += y;
		_f2part_ += y*y;
	}

	pthread_mutex_lock(&sum_fs_); //Lock
	_f_ += _fpart_;
	_f2_ += _f2part_;
	pthread_mutex_unlock(&sum_fs_); //Unlock

	return NULL;
}

int main(int argc, char **argv) {
	//Checking quantity of arguments
	if (argc != 4) {
		fprintf(stderr, "ERROR: Invalid number of arguments.\n");
		exit(1);
	}
	N = atoll(argv[1]); k = atoi(argv[2]); M = atoi(argv[3]); //Arguments

	//Variables
	double result_1, result_2; //Value of integration

	//Integration result algebrically
	double result;
	if (abs(k) <= abs(M) && M >= 0) result = 1;
	else if (abs(k) <= abs(M) && M < 0) result = -1;
	else result = 0;

	//Setting time measurement
	clock_t start, end;
	double execution_time;

	//Semaphore
	if (pthread_mutex_init(&sum_fs_, NULL)) {
        fprintf(stderr, "ERROR: Mutex not initialized\n");
        exit(1);
	}

	srand(time(NULL)); //Seed of random

	// -----------------------------------------------------------------------------------------------------------------

	//Monte Carlos Integration with Distributed Computing Techniques
	// MPI_Status status; //*MPI

	// int num_processes, this_process; //*MPI
	// MPI_Init(&argc, &argv); //*MPI
	// MPI_Comm_size(MPI_COMM_WORLD, &num_processes); //*MPI
	// MPI_Comm_rank(MPI_COMM_WORLD, &this_process); //*MPI

	/*1. LOAD BALANCER WITH THE MINIMUM TIME*/




	/*2. ONE GPU AND ONE CPU THREAD*/
	/*cudaSetDevice(0);
	cudaDeviceReset();

	_f_ += 0; //Initialization
	_f2_ += 0; //Initialization

	double *_fcuda_ = NULL;
	double *_f2cuda_ = NULL;

	//Alloc _f_ (_fcuda_) and _f2_ (_f2cuda_) on device
	cudaMalloc((void **) &_fcuda_, sizeof(double));
	cudaMalloc((void **) &_f2cuda_, sizeof(double));

	start = clock(); //Start of work
	//Copy _f_ and _f2_ from host to device
	cudaMemcpy(_fcuda_, &_f_, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(_f2cuda_, &_f2_, sizeof(double), cudaMemcpyHostToDevice);

	cuda_integration<<<1, 1>>>(_fcuda_, _f2cuda_, M, k, N, time(NULL));
	
	//Rescue _f_ and _f2_ from device to host
	cudaMemcpy(&_f_, _fcuda_, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&_f2_, _f2cuda_, sizeof(double), cudaMemcpyDeviceToHost);

	//Integration value
	_f_ = _f_/N;
	_f2_ = _f2_/N;

	result_1 = (_f_ + sqrt((_f2_ - _f_*_f_)/N));
	result_2 = (_f_ - sqrt((_f2_ - _f_*_f_)/N));
	end = clock(); //End of work

	//Print time and error
	execution_time = ((double) (end - start))/CLOCKS_PER_SEC;
	printf("result %lf, result1 %lf, result2 %lf\n", result, result_1, result_2); //DEBUG
	printf("Tempo na GPU com uma thread na CPU em segundos: %lf\n", execution_time);
	printf("Erro no calculo com a soma: %lf\n", fabs(result_1 - result));
	printf("Erro no calculo com a subtracao: %lf\n\n", fabs(result_2 - result));*/



	/*3. T CPU THREADS*/
	int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
	pthread_t *id; if (num_cpus > 1) id = (pthread_t *) emalloc((num_cpus - 1)*sizeof(pthread_t));

	_f_ += 0; //Initialization
	_f2_ += 0; //Initialization

	start = clock(); //Start of work
	if (N < num_cpus) { //It's not worth to use T > 1 threads
		num_cpus = 1;
		thread_integration((void *) &num_cpus);
	}
	else {
		for (int i = 0; i < num_cpus - 1; i++) { //T-1 threads
			if (pthread_create(&id[i], NULL, thread_integration, (void *) &num_cpus)) {
				fprintf(stderr, "ERROR: Thread not created.\n");
				exit(1);
			}
		}

		//Main thread v
		double x, y;
		double _fpart_ = 0; //Partial _f_
		double _f2part_ = 0; //Partial _f_

		for (int i = 0; i < N/num_cpus + (N - N/num_cpus*num_cpus); i++) {
			x = x_random(); //Random number in (0, 0.5]
			y = f(M, k, x);	
			_fpart_ += y;
			_f2part_ + y*y;
		}

		pthread_mutex_lock(&sum_fs_); //Lock
		_f_ += _fpart_;
		_f2_ += _f2part_;
		pthread_mutex_unlock(&sum_fs_); //Unlock
		//Main thread ^


		for (int i = 0; i < num_cpus - 1; i++) { //Waiting for the other threads
			if (pthread_join(id[i], NULL)) {
				fprintf(stderr, "ERROR: Thread not joined.\n");
				exit(1);
			}
		}
	}

	//Integration value
	_f_ = _f_/N;
	_f2_ = _f2_/N;

	result_1 = (_f_ + sqrt((_f2_ - _f_*_f_)/N));
	result_2 = (_f_ - sqrt((_f2_ - _f_*_f_)/N));
	end = clock(); //End of work

	//Print time and error
	execution_time = ((double) (end - start))/CLOCKS_PER_SEC;
	printf("result %lf, result1 %lf, result2 %lf\n", result, result_1, result_2); //DEBUG
	printf("Tempo na CPU com %d threads em segundos: %lf\n", num_cpus, execution_time);
	printf("Erro no calculo com a soma: %lf\n", fabs(result_1 - result));
	printf("Erro no calculo com a subtracao: %lf\n\n", fabs(result_2 - result));



	/*4. ONE CPU THREAD*/
	num_cpus = 1;
	
	_f_ += 0; //Initialization
	_f2_ += 0; //Initialization

	start = clock(); //Start of work
	thread_integration((void *) &num_cpus);

	//Integration value
	_f_ = _f_/N;
	_f2_ = _f2_/N;

	result_1 = (_f_ + sqrt((_f2_ - _f_*_f_)/N));
	result_2 = (_f_ - sqrt((_f2_ - _f_*_f_)/N));
	end = clock(); //End of work

	//Print time and error
	execution_time = ((double) (end - start))/CLOCKS_PER_SEC;
	printf("result %lf, result1 %lf, result2 %lf\n", result, result_1, result_2); //DEBUG
	printf("Tempo sequencial em segundos: %lf\n", execution_time);
	printf("Erro no calculo com a soma: %lf\n", fabs(result_1 - result));
	printf("Erro no calculo com a subtracao: %lf\n", fabs(result_2 - result));

	// -----------------------------------------------------------------------------------------------------------------

	//Finishing
	if (num_cpus > 1) free(id);
	pthread_mutex_destroy(&sum_fs_);
	// MPI_Finalize(); //*MPI
}