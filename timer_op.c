
#include <limits.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "timer.h"

// addition of external function interfaces to be used in test
extern void COMPUTE_OP_REF(int m0, int n0, float *A_dist, float *B_dist, float *C_dist);

extern void COMPUTE_OP_TEST(int m0, int n0, float *A_dist, float *B_dist, float *C_dist);

extern void DISTRIBUTE_ALLOCATION_REF(int m0, int n0, float **A_dist, float **B_dist, float **C_dist);

extern void DISTRIBUTE_ALLOCATION_TEST(int m0, int n0, float **A_dist, float **B_dist, float **C_dist);

extern void DISTRIBUTE_DATA_REF(int m0, int n0, float *A_seq, float *B_seq, float *C_seq, float *A_dist, float *B_dist, float *C_dist);

extern void DISTRIBUTE_DATA_TEST(int m0, int n0, float *A_seq, float *B_seq, float *C_seq, float *A_dist, float *B_dist, float *C_dist);

extern void COLLECTION_REF(int m0, int n0, float *C_seq, float *C_dist);

extern void COLLECTION_TEST(int m0, int n0, float *C_seq, float *C_dist);

extern void FREE_MEMORY_REF(float *A_dist, float *B_dist, float *C_dist);

extern void FREE_MEMORY_TEST(float *A_dist, float *B_dist, float *C_dist);

// fill created memory buffer with random values
void fill_buffer_with_random_values(float *buffer, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
    {
        buffer[i] = (float)rand() / (float)(RAND_MAX);
    }
}

// fill a memory buffer with a specified value
void fill_buffer_with_specified_value(float *buffer, int num_elements, float value)
{
    for (int i = 0; i < num_elements; i++)
    {
        buffer[i] = value;
    }
}

long pick_min_in_list(int num_trials, long *results)
{
    long current_min = LONG_MAX;

    for (int i = 0; i < num_trials; ++i)
        if (results[i] < current_min)
            current_min = results[i];

    return current_min;
}

long pick_max_in_list(int num_trials, long *results)
{
    long current_max = LONG_MIN;

    for (int i = 0; i < num_trials; ++i)
        if (results[i] > current_max)
            current_max = results[i];

    return current_max;
}

// function to flush cache and have clean cache state for test iterations
void flush_cache()
{
    // Allocate memory equal to L3 cache size
    int size = 1024 * 1024 * 8;

    // allocate memory for buffer that fills the cache
    int *buffer = (int *)malloc(size * sizeof(int));
    // fill the buffer
    int i, result = 0;
    // volatile prevents compiler from optimizing out the loop
    volatile int sink;
    for (i = 0; i < size; i++)
    {
        result += buffer[i];
    }
    // force write to memory
    sink = result;
    // free the buffer
    free(buffer);
}

void time_function_call(int num_trials, int num_runs, long *results, int m0, int n0, float *A_dist, float *B_dist, float *C_dist)
{
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
    for (int trial = 0; trial < num_trials; trial++)
    {
        // start timer
        TIMER_GET_CLOCK(start);

        // run for number of runs
        for (int run = 0; run < num_runs; run++)
        {
            // call the function to be timed
            COMPUTE_OP_TEST(m0, n0, A_dist, B_dist, C_dist);
        }

        TIMER_GET_CLOCK(stop);

        // get the difference in time and append to results
        TIMER_GET_DIFF(start, stop, results[trial]);

        long max_time;
        // reduce the max time among all ranks to root
        MPI_Reduce(&results[trial], &max_time, 1, MPI_LONG, MPI_MAX, root_id, MPI_COMM_WORLD);

        // root rank stores the max time in results
        if (rid == root_id)
        {
            results[trial] = max_time;
        }
    }
}