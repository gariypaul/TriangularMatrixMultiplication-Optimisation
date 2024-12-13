
#include <limits.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "timer.h"

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