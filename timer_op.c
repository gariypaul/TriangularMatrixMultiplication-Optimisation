
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "timer.h"

// addition of external function interfaces to be used in test

extern void COMPUTE_OP_TEST(int m0, int n0, float *A_dist, float *B_dist,
                            float *C_dist);

extern void DISTRIBUTE_ALLOCATION_TEST(int m0, int n0, float **A_dist,
                                       float **B_dist, float **C_dist);

extern void DISTRIBUTE_DATA_TEST(int m0, int n0, float *A_seq, float *B_seq,
                                 float *C_seq, float *A_dist, float *B_dist,
                                 float *C_dist);

extern void COLLECTION_TEST(int m0, int n0, float *C_seq, float *C_dist);

extern void FREE_MEMORY_TEST(float *A_dist, float *B_dist, float *C_dist);

// fill created memory buffer with random values
void fill_buffer_with_random_values(float *buffer, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    buffer[i] = (float)rand() / (float)(RAND_MAX);
  }
}

// fill a memory buffer with a specified value
void fill_buffer_with_specified_value(float *buffer, int num_elements,
                                      float value) {
  for (int i = 0; i < num_elements; i++) {
    buffer[i] = value;
  }
}

long pick_min_in_list(int num_trials, long *results) {
  long current_min = LONG_MAX;

  for (int i = 0; i < num_trials; ++i)
    if (results[i] < current_min) current_min = results[i];

  return current_min;
}

long pick_max_in_list(int num_trials, long *results) {
  long current_max = LONG_MIN;

  for (int i = 0; i < num_trials; ++i)
    if (results[i] > current_max) current_max = results[i];

  return current_max;
}

// function to flush cache and have clean cache state for test iterations
void flush_cache() {
  // Allocate memory equal to L3 cache size
  int size = 1024 * 1024 * 8;

  // allocate memory for buffer that fills the cache
  int *buffer = (int *)malloc(size * sizeof(int));
  if (buffer == NULL) {
    printf("Buffer(Cache flush): Memory allocation failed\n");
    exit(1);
  }
  // fill the buffer
  int i, result = 0;
  // volatile prevents compiler from optimizing out the loop
  volatile int sink;
  for (i = 0; i < size; i++) {
    result += buffer[i];
  }
  // force write to memory
  sink = result;
  // free the buffer
  free(buffer);
}

void time_function_call(int num_trials, int num_runs, long *results, int m0,
                        int n0, float *A_dist, float *B_dist, float *C_dist) {
  int rid;
  int num_ranks;
  int tag;
  MPI_Status status;
  int root_id = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  // initialize timer counters
  TIMER_INIT_COUNTERS(start, stop);

  // barrier to synchronize all ranks
  MPI_Barrier(MPI_COMM_WORLD);
  // warm up timers
  TIMER_WARMUP(start, stop);

  // flush cache before trials
  flush_cache();

  // run the function call for the specified number of trials
  for (int trial = 0; trial < num_trials; trial++) {
    // start timer
    TIMER_GET_CLOCK(start);

    // run for number of runs
    for (int run = 0; run < num_runs; run++) {
      // call the function to be timed
      COMPUTE_OP_TEST(m0, n0, A_dist, B_dist, C_dist);
    }

    TIMER_GET_CLOCK(stop);

    // get the difference in time and append to results
    TIMER_GET_DIFF(start, stop, results[trial]);

    long max_time;
    // reduce the max time among all ranks to root
    MPI_Reduce(&results[trial], &max_time, 1, MPI_LONG, MPI_MAX, root_id,
               MPI_COMM_WORLD);

    // root rank stores the max time in results
    if (rid == root_id) {
      results[trial] = max_time;
    }
  }
}
int scale_steps(int step, int dim) {
  if (dim < 0) {
    return -1 * dim;
  } else {
    return dim * step;
  }
}
int main(int argc, char *argv[]) {
  int rid;
  int num_ranks;
  int tag = 0;
  int root_id = 0;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  FILE *csv_file;

  int num_trials = 10;
  int num_runs = 1;

  // Parameters for the test
  int min_size;
  int max_size;
  int step_size;

  int input_m0;
  int input_n0;

  if (argc == 1) {
    min_size = 16;
    max_size = 1024;
    step_size = 16;

    input_m0 = 3;
    input_n0 = 3;

    csv_file = stdout;
  } else if (argc == 5 + 1 || argc == 6 + 1) {
    min_size = atoi(argv[1]);
    max_size = atoi(argv[2]);
    step_size = atoi(argv[3]);

    input_m0 = atoi(argv[4]);
    input_n0 = atoi(argv[5]);

    // print to stdout if no file is specified
    csv_file = stdout;

    // if file specified then use the file
    if (argc == 6 + 1) {
      csv_file = fopen(argv[6], "w");
    } else {
      csv_file = NULL;
    }
  } else {
    printf(
        "Usage: %s [min_size] [max_size] [step_size] [m0] [n0] [output_file]\n",
        argv[0]);
    exit(1);
  }
  // use the root id to print the header on CSV file
  if (rid == root_id) {
    fprintf(csv_file, "num_ranks,m0,n0,gflops\n");
  }

  for (int size = min_size; size <= max_size; size += step_size) {
    // scale the input sizes as per the step size
    int m0 = scale_steps(size, input_m0);
    int n0 = scale_steps(size, input_n0);

    // buffer sizes
    int A_seq_size = m0 * m0;
    int B_seq_size = m0 * n0;
    int C_seq_size = m0 * n0;

    // allocate memory for sequential buffers
    float *A_seq = (float *)malloc(A_seq_size * sizeof(float));
    float *B_seq = (float *)malloc(B_seq_size * sizeof(float));
    float *C_seq = (float *)malloc(C_seq_size * sizeof(float));

    // check if memory allocation was successful
    if (A_seq == NULL || B_seq == NULL || C_seq == NULL) {
      printf("Test: Sequential Memory buffer allocation failed\n");
      exit(1);
    }

    // fill buffers with random values using root_id
    if (rid == root_id) {
      fill_buffer_with_random_values(A_seq, A_seq_size);
      fill_buffer_with_random_values(B_seq, B_seq_size);
      fill_buffer_with_specified_value(C_seq, C_seq_size, 0.0);
    }

    // allocate pointers for distributed buffers
    float *A_dist_test;
    float *B_dist_test;
    float *C_dist_test;

    // // distribute memory allocation
    DISTRIBUTE_ALLOCATION_TEST(m0, n0, &A_dist_test, &B_dist_test,
                               &C_dist_test);

    if (A_dist_test == NULL || B_dist_test == NULL || C_dist_test == NULL) {
      printf("Test: Distributed Memory buffer allocation failed\n");
      exit(1);
    }
    // // distribute data
    DISTRIBUTE_DATA_TEST(m0, n0, A_seq, B_seq, C_seq, A_dist_test, B_dist_test,
                         C_dist_test);

    // allocate memory for results
    long *results = (long *)malloc(num_trials * sizeof(long));
    if (results == NULL) {
      printf("Test:Results Memory allocation failed\n");
      exit(1);
    }

    // perform test
    time_function_call(num_trials, num_runs, results, m0, n0, A_dist_test,
                       B_dist_test, C_dist_test);

    // pick min in results
    long min_time = pick_min_in_list(num_trials, results);

    // get floating operation per second
    long num_flops =
        m0 * m0 * n0 * 2;  // multiply by two to factor in addition operation

    // get throughput in GFLOPS
    float throughput = (float)num_flops / (float)min_time;

    // free results memory and set pointer to NULL to avoid dangling pointers
    free(results);
    results = NULL;

    // collect the distributed data and write to sequential buffer
    COLLECTION_TEST(m0, n0, C_seq, C_dist_test);

    // free buffers
    FREE_MEMORY_TEST(A_dist_test, B_dist_test, C_dist_test);
    A_dist_test = NULL;
    B_dist_test = NULL;
    C_dist_test = NULL;

    // print the results to the csv file
    if (rid == root_id) {
      fprintf(csv_file, "%d, %d, %d,%2.2f\n", num_ranks, m0, n0, throughput);
    }

    // free the sequential buffers and set pointers to NULL to avoid dangling
    // pointers
    free(A_seq);
    free(B_seq);
    free(C_seq);
    // A_seq = NULL;
    // B_seq = NULL;
    // C_seq = NULL;
  }

  // close the file if it was opened
  if (rid == root_id && csv_file != NULL) {
    fclose(csv_file);
  }

  MPI_Finalize();
}
