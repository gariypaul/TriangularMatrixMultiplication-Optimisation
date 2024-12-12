#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#ifndef COMPUTEOPS
#define COMPUTEOPS baseline
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

void COMPUTEOPS(int m0, int n0, float *A, float *B, float *C)
{   

    //Matrices Row and Column Strides
    //Matrices are stored in Row Major Order
    int rs_A = m0;
    int CS_A = 1;

    int rs_B = m0;
    int CS_B = 1;

    int rs_C = m0;
    int CS_C = 1;


    //Lower Triangular Matrix Multiplication algorithm 
    for(int i0=0; i0<m0; i0++){
        for(int j0=0;j0<n0;j0++){
            float result = 0.0f;
            for(int k0=0;k0<m0;k0++){
                if(i0>=k0){
                    result += A[i0*rs_A + k0*CS_A] * B[k0*rs_B + j0*CS_B];
                }
            }
            C[i0*rs_C + j0*CS_C] = result;
        }
    }

}
