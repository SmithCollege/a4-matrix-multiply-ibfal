#include <iostream>
#include <math.h>
#include <sys/time.h>

#define TILE_WIDTH 10

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void MatrixMul(float* M, float* N, float* P, int Width){
	__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
// Identify the row and column of the P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
// Loop over the M and N tiles required to compute the P element
// The code assumes that the Width is a multiple of TILE_WIDTH!
	for (int m = 0; m < Width/TILE_WIDTH; ++m) {
// Collaborative loading of M and N tiles into shared memory
		subTileM[ty][tx] = M[Row*Width + m*TILE_WIDTH+tx];
		subTileN[ty][tx] = N[(m*TILE_WIDTH+ty)*Width+Col];
		__syncthreads();

	 	for (int k = 0; k < TILE_WIDTH; ++k){
			Pvalue += subTileM[ty][k] * subTileN[k][tx];
		}
		__syncthreads();
	}
	P[Row*Width+Col]=Pvalue;
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
