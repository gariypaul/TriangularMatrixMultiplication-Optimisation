BASELINE_VARIANT="baseline.c"

VARIANT_1="variant1.c"
VARIANT_2="variant2.c"
VARIANT_3="variant3.c"

#Compiler flags
CC=mpicc
CFLAGS="-std=c99 -O2 -mfma -mavx2"