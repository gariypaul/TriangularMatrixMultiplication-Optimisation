#Define vars for build 

#Parameters for the matrix ops runs
MIN_SIZE = 64
MAX_SIZE = 512
STEP_SIZE = 64

#MPI parameters
NUM_RANKS = 4

#Shell 
SHELL:= /bin/bash

clean:
	@echo "Cleaning up"
	rm -f *.x
	rm -f *.o
	rm -f *~

clean_all: clean
	@echo "Cleaning up all"
	rm -f *.csv
	rm -f *.png
	
all: clean run_bench 


run_bench: build_bench
	@echo "Running benchmarks"
	mpiexec -n ${NUM_RANKS} ./run_test_variant01.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_bench_var1.csv
	mpiexec -n ${NUM_RANKS} ./run_test_variant02.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_bench_var2.csv
	mpiexec -n ${NUM_RANKS} ./run_test_variant03.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_bench_var3.csv


build_bench:
	@echo "Building benchmarks"
	./build_test_op.sh