#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> //*MPI
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <sys/time.h>

//Constant
#define PI 3.14159265358979323846 //Pi

//Global variables
long long N; int k, M; //Arguments for integration
double _f_, _f2_; //Intermediary value of integration

//Semaphore
pthread_mutex_t sum_fs_;

extern "C" double x_random() {
	//Generate a random number in the interval (0, 0.5]
	return ( ((double) (rand() + 1)) / ( ((long long) RAND_MAX) + 1) ) * 0.5;
}

extern "C" double f(int M_arg, int k_arg, double x_arg) {
	//Calculate the math function: 
	
	// sin([2M + 1]*pi*x) * cos(2*pi*k*x)
	// ----------------------------------
	//            sin(pi*x)

	return (sin((2*M_arg + 1)*PI*x_arg)*cos(2*PI*k_arg*x_arg))/sin(PI*x_arg);
}

extern "C" __global__ void cuda_integration(double *_fcuda_, double *_f2cuda_, int _M, int _k, long long _N, unsigned long seed) {
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
}

extern "C" void *thread_integration(void *num_cpus_arg) {
	int cpus = *((int *) num_cpus_arg);

	//Setting time measurement
	// clock_t start, end;
	// double execution_time;

	// start = clock(); //Start of work

	double x, y;
	double _fpart_ = 0; //Partial _f_
	double _f2part_ = 0; //Partial _f2_

	for (int i = 0; i < N/cpus; i++) {
		x = x_random(); //Random number in (0, 0.5]
		y = f(M, k, x);
		_fpart_ += y;
		_f2part_ += y*y;
	}

	// end = clock(); //End of work

	// execution_time = ((double) (end - start))/CLOCKS_PER_SEC;
	// printf("Tempo na thread_integration: %lf\n", execution_time);

	pthread_mutex_lock(&sum_fs_); //Lock
	_f_ += _fpart_;
	_f2_ += _f2part_;
	pthread_mutex_unlock(&sum_fs_); //Unlock

	return NULL;
}

extern "C" int mycal(long long NN, int kk, int MM) {
	N = NN;
	k = kk;
	M = MM;

	printf("%d %d %d\n", N, k, M);
	double result_1, result_2; //Value of integration
	int num_cpus; //Number of CPUs

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

	cudaSetDevice(0);
	cudaDeviceReset();
	/*1. LOAD BALANCER WITH THE MINIMUM TIME*/
	start = clock();

	num_cpus = 2;
	pthread_t cpu_id;
	double *_fcuda_ = NULL;
	double *_f2cuda_ = NULL;

	//Alloc _f_ (_fcuda_) and _f2_ (_f2cuda_) on device
	cudaMalloc((void **) &_fcuda_, sizeof(double));
	cudaMalloc((void **) &_f2cuda_, sizeof(double));


	// *_f_ = 0; //Initialization
	// *_f2_ = 0; //Initialization


	if (pthread_create(&cpu_id, NULL, thread_integration, (void *) &num_cpus)) {
		fprintf(stderr, "ERROR: Thread not created.\n");
		exit(1);
	}

	// if (this_process == 0) {
		double _fpart_; //Partial _f_
		double _f2part_; //Partial _f2_

		//Copy _f_ and _f2_ from host to device
		cudaMemcpy(_fcuda_, &_fpart_, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(_f2cuda_, &_f2part_, sizeof(double), cudaMemcpyHostToDevice);

		cuda_integration<<<1, 1>>>(_fcuda_, _f2cuda_, M, k, N/2 + (N - N/2*2), time(NULL));
		
		//Rescue _f_ and _f2_ from device to host
		cudaMemcpy(&_fpart_, _fcuda_, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&_f2part_, _f2cuda_, sizeof(double), cudaMemcpyDeviceToHost);

		pthread_mutex_lock(&sum_fs_); //Lock
		_f_ += _fpart_;
		_f2_ += _f2part_;
		pthread_mutex_unlock(&sum_fs_); //Unlock
	// }

	if (pthread_join(cpu_id, NULL)) {
		fprintf(stderr, "ERROR: Thread not joined.\n");
		exit(1);
	}
	//Integration value
	_f_ = _f_/N;
	_f2_ = _f2_/N;

	result_1 = (_f_ + sqrt((_f2_ - _f_*_f_)/N));
	result_2 = (_f_ - sqrt((_f2_ - _f_*_f_)/N));

	// MPI_Finalize(); //*MPI

	end = clock();
	execution_time = ((double) (end - start))/CLOCKS_PER_SEC;
	printf("Tempo com balanceamento de carga em segundos: %lf\n", execution_time);
	printf("Erro no calculo com a soma: %lf\n", fabs(result_1 - result));
	printf("Erro no calculo com a subtracao: %lf\n\n", fabs(result_2 - result));

	return 0;
}