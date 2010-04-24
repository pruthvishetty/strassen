/* 
Compile options
	gcc -lrt -lm -fompenmp
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>

//matrix dimensions
#define DIM_N 1000
#define DIM_K 1000
#define DIM_M 1000

//other stuff
double sum, snorm;
double *d;

//matrices
double A[DIM_M][DIM_k], B[DIM_K][DIM_N], C[DIM_M][DIM_N];

int main (int argc, char *argv[]){
  long stime, ntime;
  struct timespec start, finish;
  
  d = (double *) malloc(sizeof(double) * DIM_K);
  sum = 0.0; snorm = 0.0;
  for (i=0; i< DIM_K; i++){
    d[i] = drand48();
    snorm += d[i]*d[i];
  }
  for(i=0; i<DIM_K; i++){
    d[i] = d[i]/sqrt(snorm);
    sum += d[i]*d[i];
  }
  for (i=0; i<DIM_M; i++)
    for (j=0; j<DIM_K; j++)
      A[i][j] = -2*d[i]*d[j];
  
  for (i=0; i<DIM_M; i++)
    A[i][i] = 1.0 + A[i][i];

  for (i=0; i<DIM_K; i++)
    for (j=0; j<DIM_N; j++)
      B[i][j] = -2*d[i]*d[j];
  
  for (i=0; i<DIM_K; i++)
    B[i][i] = 1.0 + A[i][i];

  //start timer
}
