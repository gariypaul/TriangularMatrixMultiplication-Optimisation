#Define vars for build 

#Parameters for the matrix ops runs
MIN_SIZE = 64
MAX_SIZE = 512
STEP_SIZE = 16

#MPI parameters
NUM_RANKS = 4

#Shell 
SHELL:= /bin/bash

clean:
	@echo "Cleaning up"
	rm -f *.x
	rm -f *.o
	rm -f *~

clean-all: clean
	@echo "Cleaning up all"
	rm -f *.csv
	rm -f *.png
	
all: clean run-verifier run-bench 


run-bench: build-bench
	@echo "Running benchmarks"
	mpiexec -n ${NUM_RANKS} ./run_test_variant01.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_bench_var1.csv
	mpiexec -n ${NUM_RANKS} ./run_test_variant02.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_bench_var2.csv
	mpiexec -n ${NUM_RANKS} ./run_test_variant03.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_bench_var3.csv

	python3 ./result_plotter.py "Variant comparison plot" "Results_Plot.png" "result_bench_var1.csv" "result_bench_var2.csv" "result_bench_var3.csv"

build-bench:
	@echo "Building benchmarks"
	./build_test_op.sh

run-verifier: build-verifier
	@echo "Running verifier"
	mpiexec -n ${NUM_RANKS} ./run_verifier_variant01.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_verifier_var1.csv
	cat result_verifier_var1.csv
	mpiexec -n ${NUM_RANKS} ./run_verifier_variant02.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_verifier_var2.csv
	cat result_verifier_var2.csv
	mpiexec -n ${NUM_RANKS} ./run_verifier_variant03.x ${MIN_SIZE} ${MAX_SIZE} ${STEP_SIZE} 1 1 result_verifier_var3.csv
	cat result_verifier_var3.csv
	@echo "Number of FAILS: $$(grep -o "FAIL" result_verifier_var*.csv | wc -l)"

build-verifier:
	@echo "Building verifier"
	./build_verifier_op.sh