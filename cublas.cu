#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <cublas_v2.h>
 
#define TILE_WIDTH 2
 
 __global__ void gpu_blas( double* M,  double* N,  double* P, int width){
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
     int width = TILE_WIDTH*2;
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


    dim3 dimGrid(ceil((1.0*width)/TILE_WIDTH), ceil((1.0*width)/TILE_WIDTH),1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    cudaDeviceSynchronize();

    gpu_blas<<<dimGrid, dimBlock>>>(x, y, z, width);

        cudaMemcpy(dz,z, sizeof(double) * width * width, cudaMemcpyDeviceToHost);

          for (int i = 0; i < width; i++) {
                printf("\n");
            for (int j = 0; j < width; j++) {
              printf("%f ", z[i * width + j]);
            }
          }

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
