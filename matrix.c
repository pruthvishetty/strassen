/* Compile options
   gcc -lrt -lm -fopenmp 
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
#define DIM_I 1000
#define DIM_J 1200
#define DIM_K 1000
#define STEP 10
//other stuff
double sum, snorm;
double *d;
//matrices
double A[DIM_I][DIM_J], B[DIM_J][DIM_K], C[DIM_I][DIM_K];

int main (int argc, char *argv[])
{
   int n_threads, t_id, chunk = STEP, i, j, k, a, b, c;
   //measuring time
   long stime, ntime;
   struct timespec start,finish;
   int total_time;

   //create the matrices in question
   d = (double*) malloc(sizeof(double)*DIM_J);
   sum = 0.0; snorm = 0.0;
   for (i = 0; i < DIM_J; i++){
      d[i] = drand48();
      snorm += d[i]*d[i];
   }
   for (i = 0; i < DIM_J; i++){
      d[i] = d[i]/sqrt(snorm);
      sum += d[i]*d[i];
   }

   for (i = 0; i < DIM_I; i++)
      for (j = 0; j < DIM_J; j++)
      A[i][j] =  -2*d[i]*d[j];

   for (i = 0; i < DIM_I; i++)
      A[i][i] = 1.0 + A[i][i];

   for (j = 0; j < DIM_J; j++)
      for (k = 0; k < DIM_K; k++)
      B[j][k] = -2*d[j]*d[k];

   for (j = 0; j < DIM_J; j++)
      B[j][j] = 1.0 + B[j][j];

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
  printf("Time taken = %ds \n", total_time);
}
