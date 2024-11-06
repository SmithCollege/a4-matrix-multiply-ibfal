#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
 
#define TILE_WIDTH 5

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}
 
void gpu_blas( double* M,  double* N,  double* P, int width){
	const double a = 1;
	const double b = 0;
	const double *A = &a;
	const double *B = &b;

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,width,width, width, A,M, width,N,width, B, P, width);	

	cublasDestroy(handle);
}
 
 int main(){
     int width = 100000;
     double *x,*y,*z, *dx, *dy, *dz;
 
     dx = (double *)malloc(sizeof(double) * width * width);
     dy = (double *)malloc(sizeof(double) * width * width);
     dz = (double *)malloc(sizeof(double) * width * width);
 
     cudaMallocManaged(&x, sizeof(double) * width * width);
     cudaMallocManaged(&y, sizeof(double) * width * width);
     cudaMallocManaged(&z, sizeof(double) * width * width);
 
     for (int i = 0; i < width; i++) {
       for (int j = 0; j < width; j++) {
         dx[i * width + j] = 1.0; // x[i][j]
         dy[i * width + j] = 1.0;
       }
      }
 
    cudaMemcpy(x, dx, sizeof(double) * width * width, cudaMemcpyHostToDevice);
    cudaMemcpy(y,dy, sizeof(double) * width * width, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

	double t0 = get_clock();
    gpu_blas(x, y, z, width);
	double t1 = get_clock();
	
    cudaMemcpy(dz,z, sizeof(double) * width * width, cudaMemcpyDeviceToHost);
          
	printf("\n");
	printf("Time: %f ns\n", (1000000000.0*(t1-t0)));
    printf("\n");
    
    for (int i = 0; i < width; i++) {       
    	for (int j = 0; j < width; j++) {
        	if (z[i * width + j] != width) {
            	 printf("Error at z[%d][%d]: %f\n", i, j, z[i * width + j]);
              }
            }
          }
    return 0;
}
