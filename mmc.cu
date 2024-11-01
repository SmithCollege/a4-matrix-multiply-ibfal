#include <iostream>
#include <math.h>

#define TILE_WIDTH 2

__global__ 
void MatrixMul(float* M, float* N, float* P, int width) {
	int r = blockIdx.y*blockDim.y+threadIdx.y;
	int c = blockIdx.x*blockDim.x+threadIdx.x;

	if ((r<width) &&(c<width)){
		float pval =0;
		for (int k=0; k<width; ++k){
			pval += M[r*width+k]*N[k*width+c];
			}
		P[r*width+c]=pval;
	}
}

int main() {
  int width = 4;
  float *x,*y,*z;	
  
  cudaMallocManaged(&x, sizeof(float) * width * width);
  cudaMallocManaged(&y, sizeof(float) * width * width);
  cudaMallocManaged(&z, sizeof(float) * width * width);
  
  for (int i = 0; i < width; i++) {
  printf("i= %d ", i);
    for (int j = 0; j < width; j++) {
    printf("j= %d ", j);	
      x[i * width + j] = 1; // x[i][j]
      y[i * width + j] = 1;
      printf("values = %d  %d ", x[i * width + j], y[i * width + j]);
    }
  }

  for (int i = 0; i < width; i++) {
  	printf("\n");
    for (int j = 0; j < width; j++) {
      printf("%d ", x[i * width + j]); 
    }
  }

  for (int i = 0; i < width; i++) {
  	printf("\n");
    for (int j = 0; j < width; j++) {
      printf("%d ", y[i * width + j]); 
    }
  }

  dim3 dimGrid(ceil((1.0*width)/TILE_WIDTH), ceil((1.0*width)/TILE_WIDTH),1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


  MatrixMul<<<dimGrid, dimBlock>>>(x, y, z, width);

  for (int i = 0; i < width; i++) {
  	printf("\n");
    for (int j = 0; j < width; j++) {
      printf("%d ", z[i * width + j]); 
    }
  }


  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      if (z[i * width + j] != width) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * width + j]);
      }
    }
  }


  return 0;
}
