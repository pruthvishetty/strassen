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
  clock_gettime(CLOCK_REALTIME,&start);
  
  #pragma omp parallel shared(A,B,C,chunk) private(a,b,c)
  {
     #pragma omp for schedule(dynamic,chunk) nowait
      for (b=0; b<DIM_K; b++){
          for (a=0; a<DIM_I; a++){
              for (c=0; c<DIM_J; c++){
                 C[a][b] += A[a][c] * B[c][b];
              }
          }
      }
  }

  //stop timer
  clock_gettime(CLOCK_REALTIME,&finish);

  //calculate time taken
  ntime = finish.tv_nsec - start.tv_nsec;
  stime = (long) finish.tv_sec - (long) start.tv_sec;
  total_time = (double) stime + ((double) ntime)/1e9;
  printf("Time taken = %lf \n", (double) stime + ((double) ntime)/1e9);
}
/****************************************************************************
* Note that all following functions only deal with square matrices 
* where N is divisible by 2.
*****************************************************************************/
void addMatrices(double **x, double **y, double **z, int size){
//performs a matrix addition operation, z=x+y
}
void subMatrices(double **x, double **y, double **z, int size){
//performs a matrix subtraction operation, z=x-y
}
void splitMatrix(double **a, double**a11, **a12, **a21, **a22, int size){
//takes a matrix a adn splits it into its 4 quadrants.
}
void catMatrix(double **a, double**a11, **a12, **a21, **a22, int size){
//does the inverse of the splitMatrix function
}
void multMatrix(double **x, double **y, double **z, int size){
//multiplys two matrices: z=x*y
}
