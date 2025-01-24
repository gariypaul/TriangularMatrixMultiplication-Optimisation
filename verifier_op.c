#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// define the error threshold
#define ERROR_THRESHOLD 1.0e-3

// addition of external function interfaces to be used in test
extern void COMPUTE_OP_TEST(int m0, int n0, float *A_dist, float *B_dist,
                            float *C_dist);

extern void COMPUTE_OP_REF(int m0, int n0, float *A_dist, float *B_dist,
                           float *C_dist);

extern void DISTRIBUTE_ALLOCATION_TEST(int m0, int n0, float **A_dist,
                                       float **B_dist, float **C_dist);

extern void DISTRIBUTE_ALLOCATION_REF(int m0, int n0, float **A_dist,
                                      float **B_dist, float **C_dist);

extern void DISTRIBUTE_DATA_TEST(int m0, int n0, float *A_seq, float *B_seq,
                                 float *C_seq, float *A_dist, float *B_dist,
                                 float *C_dist);

extern void DISTRIBUTE_DATA_REF(int m0, int n0, float *A_seq, float *B_seq,
                                float *C_seq, float *A_dist, float *B_dist,
                                float *C_dist);

extern void COLLECTION_TEST(int m0, int n0, float *C_seq, float *C_dist);

extern void COLLECTION_REF(int m0, int n0, float *C_seq, float *C_dist);

extern void FREE_MEMORY_TEST(float *A_dist, float *B_dist, float *C_dist);

extern void FREE_MEMORY_REF(float *A_dist, float *B_dist, float *C_dist);

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

// pick max pairwise difference in the buffers
float max_pairwise_difference(float *A, float *B, int m, int n, int rs,
                              int cs) {
  float max_diff = 0.0;
  float res = 0.0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float diff = fabs(A[i * rs + j * cs] - B[i * rs + j * cs]);
      float sum = fabs(A[i * rs + j * cs]) + fabs(B[i * rs + j * cs]);

      if (sum > 0) {
        res = diff / sum;
      } else {
        diff = 0.0;
      }
      if (res > max_diff) {
        max_diff = res;
      }
    }
  }
  return max_diff;
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
  // use the root id to print the header on CSV file (dont want multiple
  // headers)
  if (rid == root_id) {
    fprintf(csv_file, "num_ranks,m0,n0,result\n");
  }

  for (int size = min_size; size <= max_size; size += step_size) {
    // scale the input sizes as per the step size
    int m0 = scale_steps(size, input_m0);
    int n0 = scale_steps(size, input_n0);

    // allocate memory for sequential buffers
    int A_seq_size = m0 * m0;
    int B_seq_size = m0 * n0;
    int C_seq_size = m0 * n0;

    float *A_seq = (float *)malloc(A_seq_size * sizeof(float));
    float *B_seq = (float *)malloc(B_seq_size * sizeof(float));
    float *C_seq = (float *)malloc(C_seq_size * sizeof(float));

    // verify memory allocation
    if (A_seq == NULL || B_seq == NULL || C_seq == NULL) {
      printf("Sequential Memory buffer allocation failed\n");
      exit(1);
    }

    // fill the buffers with random values(only done once with root id rank)
    if (rid == root_id) {
      fill_buffer_with_random_values(A_seq, A_seq_size);
      fill_buffer_with_random_values(B_seq, B_seq_size);
      fill_buffer_with_specified_value(C_seq, C_seq_size, 0.0);
    }

    /*
     Verifier section for the test
    */

    // allocate pointers for distributed buffers for verifier
    float *A_dist_ref;
    float *B_dist_ref;
    float *C_dist_ref;

    // allocate memory for distributed buffers for verifier
    DISTRIBUTE_ALLOCATION_REF(m0, n0, &A_dist_ref, &B_dist_ref, &C_dist_ref);

    // verify memory allocation
    if (A_dist_ref == NULL || B_dist_ref == NULL || C_dist_ref == NULL) {
      printf("Verifier: Distributed Memory buffer allocation failed\n");
      exit(1);
    }

    // distribute data for verifier
    DISTRIBUTE_DATA_REF(m0, n0, A_seq, B_seq, C_seq, A_dist_ref, B_dist_ref,
                        C_dist_ref);

    // compute the reference output
    COMPUTE_OP_REF(m0, n0, A_dist_ref, B_dist_ref, C_dist_ref);

    /*
     Section for operation under verification
    */

    // allocate pointers for distributed buffers
    float *A_dist_test;
    float *B_dist_test;
    float *C_dist_test;

    // distribute memory allocation
    DISTRIBUTE_ALLOCATION_TEST(m0, n0, &A_dist_test, &B_dist_test,
                               &C_dist_test);

    // verify memory allocation
    if (A_dist_test == NULL || B_dist_test == NULL || C_dist_test == NULL) {
      printf("Distributed Memory buffer allocation failed\n");
      exit(1);
    }

    // distribute data
    DISTRIBUTE_DATA_TEST(m0, n0, A_seq, B_seq, C_seq, A_dist_test, B_dist_test,
                         C_dist_test);

    // compute the test output
    COMPUTE_OP_TEST(m0, n0, A_dist_test, B_dist_test, C_dist_test);

    if (root_id == rid) {
      // verify the results
      float max_diff =
          max_pairwise_difference(C_dist_ref, C_dist_test, m0, n0, m0, 1);

      // print the results to the CSV file
      if (csv_file != NULL) {
        if (max_diff > ERROR_THRESHOLD) {
          fprintf(csv_file, "%d,%d,%d,FAIL\n", num_ranks, m0, n0);
        } else {
          fprintf(csv_file, "%d,%d,%d,PASS\n", num_ranks, m0, n0);
        }
      }
    }

    // free the memory allocated for the buffers
    FREE_MEMORY_TEST(A_dist_test, B_dist_test, C_dist_test);
    FREE_MEMORY_REF(A_dist_ref, B_dist_ref, C_dist_ref);

    // free the memory allocated for the sequential buffers
    free(A_seq);
    free(B_seq);
    free(C_seq);

    // set the pointers to NULL to avoid dangling pointers
    A_seq = NULL;
    B_seq = NULL;
    C_seq = NULL;
  }

  // close the file if opened
  if (csv_file != NULL && rid == root_id) {
    fclose(csv_file);
  }

  MPI_Finalize();

  return 0;
}
