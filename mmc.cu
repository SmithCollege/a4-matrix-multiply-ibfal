#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 10

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ 
void MatrixMul(float* M, float* N, float* P, int width) {
	int r = blockIdx.y*blockDim.y+threadIdx.y;
	int c = blockIdx.x*blockDim.x+threadIdx.x;

	if ((r<width) && (c<width)){
		float pval = 0.0;
		for (int k=0; k<width; ++k){
			pval += M[r*width+k]*N[k*width+c];	
			}
		P[r*width+c]=pval;//using nvprof i can see that the sum is correct but what its printing out and storing in z is not. 
	}
}

int main() {
  int width = 90;
  float *x,*y,*z, *dx, *dy, *dz;	

  dx = (float *)malloc(sizeof(float) * width * width);
  dy = (float *)malloc(sizeof(float) * width * width);
  dz = (float *)malloc(sizeof(float) * width * width);
  
  cudaMallocManaged(&x, sizeof(float) * width * width);
  cudaMallocManaged(&y, sizeof(float) * width * width);
  cudaMallocManaged(&z, sizeof(float) * width * width);
  
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      dx[i * width + j] = 1.0; // x[i][j]
      dy[i * width + j] = 1.0;
    }
  }

  cudaMemcpy(x, dx, sizeof(float) * width * width, cudaMemcpyHostToDevice);
  cudaMemcpy(y,dy, sizeof(float) * width * width, cudaMemcpyHostToDevice);
	

  dim3 dimGrid(ceil((1.0*width)/TILE_WIDTH), ceil((1.0*width)/TILE_WIDTH),1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  cudaDeviceSynchronize();
  
  double t0 = get_clock();
  MatrixMul<<<dimGrid, dimBlock>>>(x, y, z, width);
  double t1 = get_clock();
  
  cudaMemcpy(dz,z, sizeof(float) * width * width, cudaMemcpyDeviceToHost);
  
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
