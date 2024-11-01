#include <iostream>
#include <math.h>
#include <sys/time.h>


__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width){
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
	P[Row*width+Col]=Pvalue;
	}

