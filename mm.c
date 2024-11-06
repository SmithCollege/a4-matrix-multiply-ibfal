#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


void MatrixMulonHost(float*M, float*N,float*P, int width){
	for(int i=0;i<width; ++i)
		for(int j=0;j<width;++j){
			float sum = 0;
			for(int k = 0; k<width; ++k){
				float a = M[i*width+k];
				float b = N[k*width+j];
				sum += a*b;
			}
			P[i*width+j]= sum;
		}
}

int main() {
  int size = 100;

  float* x = malloc(sizeof(float) * size * size);
  float* y = malloc(sizeof(float) * size * size);
  float* z = malloc(sizeof(float) * size * size);//row of a and col of b

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
     x[i * size + j] = 1.0; // x[i][j]
     y[i * size + j] = 1.0;
    }
  }
  
  double t0 = get_clock();
  MatrixMulonHost(x, y, z, size);
  double t1 = get_clock();

  printf("\n");
  printf("Time: %f ns\n", (1000000000.0*(t1-t0)));
  printf("\n");

//error check
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (z[i * size + j] != size) {
        printf("Error at z[%f][%f]: %f\n", i, j, z[i * size + j]);
      }
    }
  }


  return 0;
}
