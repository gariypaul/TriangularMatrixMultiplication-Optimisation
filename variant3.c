#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

void COMPUTE_OP(int m0, int n0, float *A, float *B, float *C) {
    int num_ranks, rid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    // Calculate work distribution with load balancing
    int rows_per_rank = m0 / num_ranks;
    int extra_rows = m0 % num_ranks;
    int start_row = rid * rows_per_rank + (rid < extra_rows ? rid : extra_rows);
    int end_row = start_row + rows_per_rank + (rid < extra_rows ? 1 : 0);

    // Local computation buffer
    int local_rows = end_row - start_row;
    float *local_C = (float *)calloc(local_rows * n0, sizeof(float));

    // Blocked computation with correct triangular bounds
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n0; j++) {
            float sum = 0.0f;
            // Only iterate up to current row i
            for (int k = 0; k <= i; k++) {
                sum += A[i * m0 + k] * B[k * n0 + j];
            }
            local_C[(i - start_row) * n0 + j] = sum;
        }
    }

    // Prepare for flexible gathering
    int *recv_counts = NULL;
    int *displs = NULL;
    
    if (rid == 0) {
        recv_counts = (int *)malloc(num_ranks * sizeof(int));
        displs = (int *)malloc(num_ranks * sizeof(int));
        
        int curr_displ = 0;
        for (int r = 0; r < num_ranks; r++) {
            int r_rows = (m0 / num_ranks) + (r < (m0 % num_ranks) ? 1 : 0);
            recv_counts[r] = r_rows * n0;
            displs[r] = curr_displ;
            curr_displ += recv_counts[r];
        }
    }

    // Gather results using MPI_Gatherv
    MPI_Gatherv(local_C, local_rows * n0, MPI_FLOAT,
                C, recv_counts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    free(local_C);
    if (rid == 0) {
        free(recv_counts);
        free(displs);
    }
}

void DISTRIBUTE_ALLOCATION(int m0, int n0, float **A_dist, float **B_dist,
                           float **C_dist) {
  int rid;
  MPI_Comm_rank(MPI_COMM_WORLD, &rid);

  // Allocate memory on all ranks
  *A_dist = (float *)malloc(m0 * m0 * sizeof(float));
  *B_dist = (float *)malloc(m0 * n0 * sizeof(float));
  *C_dist = (float *)malloc(m0 * n0 * sizeof(float));

  if (*A_dist == NULL || *B_dist == NULL || *C_dist == NULL) {
    printf("Rank %d: Memory allocation failed\n", rid);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void DISTRIBUTE_DATA(int m0, int n0, float *A_seq, float *B_seq, float *C_seq,
                     float *A_dist, float *B_dist, float *C_dist) {
  int rid;
  MPI_Comm_rank(MPI_COMM_WORLD, &rid);

  // Root copies data to buffers
  if (rid == 0) {
    // Copy lower triangular part of A
    for (int i = 0; i < m0; i++) {
      for (int j = 0; j <= i; j++) {
        A_dist[i * m0 + j] = A_seq[i * m0 + j];
      }
    }
    // Full matrices for B and C
    for (int i = 0; i < m0 * n0; i++) {
      B_dist[i] = B_seq[i];
      C_dist[i] = C_seq[i];
    }
  }

  // Broadcast data to all ranks
  MPI_Bcast(A_dist, m0 * m0, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_dist, m0 * n0, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(C_dist, m0 * n0, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void COLLECTION(int m0, int n0, float *C_seq, float *C_dist) {
  int rid;
  MPI_Comm_rank(MPI_COMM_WORLD, &rid);

  // Root collects final results (already handled in COMPUTE_OP's MPI_Gather)
  if (rid == 0) {
    for (int i = 0; i < m0 * n0; i++) {
      C_seq[i] = C_dist[i];
    }
  }
}

void FREE_MEMORY(float *A_dist, float *B_dist, float *C_dist) {
  free(A_dist);
  free(B_dist);
  free(C_dist);
}