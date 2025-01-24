# Triangular Matrix Multiplication Optimization

## Project Overview

This project focuses on optimizing the multiplication of lower triangular matrices using MPI (Message Passing Interface) for parallel processing. The goal is to implement and compare different optimization techniques to improve the performance of lower triangular matrix multiplication.

## Matrix Multiplication

The operation being optimized is `C = A * B`, where:
- `A` is a lower triangular matrix of size `m0 x m0`
- `B` is a matrix of size `m0 x n0`
- `C` is the result matrix of size `m0 x n0`

## Optimizations

The project includes several variants of the matrix multiplication algorithm, each implementing different optimization techniques:

### Baseline
The baseline implementation performs the matrix multiplication without any specific optimizations. It serves as a reference point for comparing the performance of other optimized variants.

### Variant 1
This variant optimizes the matrix multiplication by reducing branching, this is done by removing the conditional used to isolate the zero values in the lower triangular matrix and only performing the operation on the non zero values that are necessary for the matrix multiplication. This conditional is removed by modifying the innermost for loop bound.

### Variant 2
This variant combines the optimizations from Variant 1 and Variant 2. It uses a blocked approach for better cache utilization and implements load balancing for efficient parallel processing.

### Variant 3
This variant focuses on load balancing and efficient data distribution among MPI ranks. It ensures that each rank gets an equal amount of work, minimizing idle time and improving overall performance.



## Files

- `baseline.c`: Contains the baseline implementation of the matrix multiplication.
- `variant1.c`: Contains the first optimized variant of the matrix multiplication.
- `variant2.c`: Contains the second optimized variant of the matrix multiplication.
- `variant3.c`: Contains the third optimized variant of the matrix multiplication.
- `verifier_op.c`: Contains the code for verifying the correctness of the optimized implementations.
- `timer_op.c`: Contains the code for timing the performance of the optimized implementations.
- `Makefile`: Contains the build and run commands for the project.

## Building and Running
Before running the project the following should be done to install needed python packages: 
```bash
pip install -r requirements.txt
```
The project also uses mpicc as the default compiler (specified in the dispatch_variables.sh file) as the whole project works in an MPI environment. 
MPI ranks and test sizes and steps can be changed in the Makefile.
 
To build and run the project, use the following commands:

```bash
make all
```

This command will clean, build, and run the verifier and benchmark tests for all variants.

Other individual commands used in the full build process can be called individually as well for whatever purposes needed:
```bash
make clean
make clean-all
make run-bench
make build-bench
make run-verifier
make build-verifier
```

## Results

The results of the benchmarks and verifications are saved in CSV files. The results can be visualized using the provided Python script `result_plotter.py`.

## Conclusion

This project demonstrates the effectiveness of different optimization techniques for lower triangular matrix multiplication. By comparing the performance of various implementations, we can identify the best approach for optimizing matrix operations in parallel computing environments.
