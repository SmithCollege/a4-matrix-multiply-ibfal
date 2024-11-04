#include <iostream>
#include <math.h>

#define TILE_WIDTH 2

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
  int width = TILE_WIDTH*2;
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

  MatrixMul<<<dimGrid, dimBlock>>>(x, y, z, width);
  
  cudaMemcpy(dz,z, sizeof(float) * width * width, cudaMemcpyDeviceToHost);
  
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
