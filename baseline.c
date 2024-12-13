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
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_COMM_SIZE(MPI_COMM_WORLD);
    // query the rank of the current process
    rid = MPI_COMM_RANK(MPI_COMM_WORLD);

    if (rid == 0)
    {
        // Matrices Row and Column Strides
        // Matrices are stored in Row Major Order
        int rs_A = m0;
        int CS_A = 1;

        int rs_B = m0;
        int CS_B = 1;

        int rs_C = m0;
        int CS_C = 1;

        // Lower Triangular Matrix Multiplication algorithm
        for (int i0 = 0; i0 < m0; i0++)
        {
            for (int j0 = 0; j0 < n0; j0++)
            {
                float result = 0.0f;
                for (int k0 = 0; k0 < m0; k0++)
                {
                    if (i0 >= k0)
                    {
                        result += A[i0 * rs_A + k0 * CS_A] * B[k0 * rs_B + j0 * CS_B];
                    }
                }
                C[i0 * rs_C + j0 * CS_C] = result;
            }
        }
    }
}

void DISTRIBUTE_ALLOCATION(int m0, int n0, float **A_dist, float **B_dist, float **C_dist)
{
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_COMM_SIZE(MPI_COMM_WORLD);
    // query the rank of the current process
    rid = MPI_COMM_RANK(MPI_COMM_WORLD);

    if(rid == 0)
    {
        // Allocate memory for the matrices
        *A_dist = (float *)malloc(m0 * n0 * sizeof(float));
        *B_dist = (float *)malloc(m0 * n0 * sizeof(float));
        *C_dist = (float *)malloc(m0 * n0 * sizeof(float));
    }

}

void DISTRIBUTE_DATA(int m0, int n0, float *A_seq, float *B_seq, float *C_seq, float *A_dist, float *B_dist, float *C_dist)
{
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_COMM_SIZE(MPI_COMM_WORLD);
    // query the rank of the current process
    rid = MPI_COMM_RANK(MPI_COMM_WORLD);

    if(rid == 0)
    {
        // Copy the data from the sequential matrices to the distributed matrices
        for (int i0 = 0; i0 < m0; i0++)
        {
            for (int j0 = 0; j0 < n0; j0++)
            {
                A_dist[i0 * m0 + j0] = A_seq[i0 * m0 + j0];
                B_dist[i0 * m0 + j0] = B_seq[i0 * m0 + j0];
                C_dist[i0 * m0 + j0] = C_seq[i0 * m0 + j0];
            }
        }
    }
}

void COLLECTION(int m0, int n0, float *C_seq, float *C_dist)
{
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_COMM_SIZE(MPI_COMM_WORLD);
    // query the rank of the current process
    rid = MPI_COMM_RANK(MPI_COMM_WORLD);

    if(rid == 0)
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
    // query the number of ranks from MPI using the default communicator
    num_ranks = MPI_COMM_SIZE(MPI_COMM_WORLD);
    // query the rank of the current process
    rid = MPI_COMM_RANK(MPI_COMM_WORLD);

    if(rid == 0)
    {
        // Free the memory allocated for the matrices
        free(A_dist);
        free(B_dist);
        free(C_dist);
    }
}