#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[])
{
int n_threads, t_id;
#pragma omp parallel private(n_threads, t_id)
  {
  /* Get thread number */
  t_id = omp_get_thread_num();
  printf("Hello World from thread = %d\n", t_id);
  /* master thread only */
  if (t_id == 0)
    {
    n_threads = omp_get_num_threads();
    printf("Number of threads = %d\n", n_threads);
    }
  } /* All threads join master thread and disband */
}

