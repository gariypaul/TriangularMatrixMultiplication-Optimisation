#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#ifndef COMPUTE_OP
#define COMPUTE_OP baseline_compute
#endif

#ifndef DISTRIBUTE_ALLOCATION
#define DISTRIBUTE_ALLOCATION baseline_distribute
#endif

#ifndef DISTRIBUTE_DATA
#define DISTRIBUTE_DATA baseline_distribute_data
#endif

#ifndef COLLECTION
#define COLLECTION baseline_collect
#endif

#ifndef FREE_MEMORY
#define FREE_MEMORY baseline_free
#endif

#define BLOCK_SIZE 16

#define min(a, b) (((a) < (b)) ? (a) : (b))
/*
This operation focuses on Lower Triangular Matrix Multiplication
The operation is C = A * B
where A is a Lower Triangular Matrix
      B is a Matrix
      C is the result of the operation
A is a m0 x n0 matrix
B is a m0 x n0 matrix
C is a m0 x n0 matrix


*/

void COMPUTE_OP(int m0, int n0, float *A, float *B, float *C)
{
    int root_id = 0;
    int num_ranks;
    int rid;
    MPI_Status status;
    int tag = 0;
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // query the rank of the current process
    rid = MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    if (rid == root_id)
    {
        // Matrices Row and Column Strides
        // Matrices are stored in Row Major Order
        int rs_A = m0;
        int CS_A = 1;

        int rs_B = m0;
        int CS_B = 1;

        int rs_C = m0;
        int CS_C = 1;

        int block_size_dist = BLOCK_SIZE * (m0 / BLOCK_SIZE);
        /*
            TODO:
            1: Add loop unrolling
            2: Add Blocking
            3: Add Parallelization
        */

        // Lower Triangular Matrix Multiplication algorithm

        for (int i = 0; i < block_size_dist; i += BLOCK_SIZE)
        {
            for (int j = 0; j < block_size_dist; j += BLOCK_SIZE)
            {
                for (int r = 0; r < m0; r++)
                {
                    for (int ji = j; ji < min(j + BLOCK_SIZE, block_size_dist); ji++)
                    {
                        float sum = 0.0f; // Initialize sum for each output element
                        for (int ii = i; ii < min(i + BLOCK_SIZE, r + 1); ii++)
                        {
                            sum += A[r * rs_A + ii] * B[ii * rs_B + ji];
                        }
                        C[r * rs_C + ji] += sum; // Accumulate result into C
                    }
                }
            }
        }
    }
}

void DISTRIBUTE_ALLOCATION(int m0, int n0, float **A_dist, float **B_dist, float **C_dist)
{

    int root_id = 0;
    int num_ranks;
    int rid;
    MPI_Status status;
    int tag = 0;
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // query the rank of the current process
    rid = MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    if (rid == root_id)
    {
        // Allocate memory for the matrices
        *A_dist = (float *)malloc(m0 * m0 * sizeof(float));
        *B_dist = (float *)malloc(m0 * n0 * sizeof(float));
        *C_dist = (float *)malloc(m0 * n0 * sizeof(float));
        // Check if memory allocation was successful
        if (*A_dist == NULL || *B_dist == NULL || *C_dist == NULL)
        {
            printf("Memory allocation failed\n");
            exit(1);
        }
    }
}

void DISTRIBUTE_DATA(int m0, int n0, float *A_seq, float *B_seq, float *C_seq, float *A_dist, float *B_dist, float *C_dist)
{
    int root_id = 0;
    int num_ranks;
    int rid;
    MPI_Status status;
    int tag = 0;
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // query the rank of the current process
    rid = MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    // Matrices Row and Column Strides
    int rs_A = m0;
    int CS_A = 1;

    int rs_B = m0;
    int CS_B = 1;

    int rs_C = m0;
    int CS_C = 1;

    if (rid == root_id)
    {
        // Copy the data from the sequential matrices to the distributed matrices
        for (int i0 = 0; i0 < m0; i0++)
        {
            for (int j0 = 0; j0 < n0; j0++)
            {
                A_dist[i0 * rs_A + j0 * CS_A] = A_seq[i0 * rs_A + j0 * CS_A];
            }
        }

        for (int i0 = 0; i0 < m0; i0++)
        {
            for (int j0 = 0; j0 < n0; j0++)
            {
                B_dist[i0 * rs_B + j0 * CS_B] = B_seq[i0 * rs_B + j0 * CS_B];
            }
        }

        for (int i0 = 0; i0 < m0; i0++)
        {
            for (int j0 = 0; j0 < n0; j0++)
            {
                C_dist[i0 * rs_C + j0 * CS_C] = C_seq[i0 * rs_C + j0 * CS_C];
            }
        }
    }
}

void COLLECTION(int m0, int n0, float *C_seq, float *C_dist)
{
    int root_id = 0;
    int num_ranks;
    int rid;
    MPI_Status status;
    int tag = 0;
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // query the rank of the current process
    rid = MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    if (rid == root_id)
    {
        // Copy the data from the distributed matrix to the sequential matrix
        for (int i0 = 0; i0 < m0; i0++)
        {
            for (int j0 = 0; j0 < n0; j0++)
            {
                C_seq[i0 * m0 + j0] = C_dist[i0 * m0 + j0];
            }
        }
    }
}

void FREE_MEMORY(float *A_dist, float *B_dist, float *C_dist)
{
    int root_id = 0;
    int num_ranks;
    int rid;
    MPI_Status status;
    int tag = 0;
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // query the rank of the current process
    rid = MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    if (rid == root_id)
    {
        // Free the memory allocated for the matrices
        free(A_dist);
        free(B_dist);
        free(C_dist);
    }
}