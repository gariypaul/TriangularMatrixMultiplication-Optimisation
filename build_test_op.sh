#turn on command echoing
set -x

source dispatch_variables.sh

#echo from build_test_op.sh
echo $BASELINE_VARIANT
echo $VARIANT_1
echo $VARIANT_2
echo $VARIANT_3
echo $CC
echo $CFLAGS

COMPUTE_NAME_REF="baseline"
DISTRIBUTED_ALLOCATE_NAME_REF="baseline_allocate"
DISTRIBUTED_FREE_NAME_REF="baseline_free"
DISTRIBUTED_DATA_NAME_REF="baseline_dist"
COLLECT_DATA_NAME_REF="baseline_collect_data"

COMPUTE_NAME_TST="test"
DISTRIBUTED_ALLOCATE_NAME_TST="test_allocate"
DISTRIBUTED_FREE_NAME_TST="test_free"
DISTRIBUTED_DATA_NAME_TST="test_dist"
COLLECT_DATA_NAME_TST="test_collect_data"

TEST_RIG="timer_op.c"

#BUILD VERIFICATION TEST
${CC} -std=c99 -c \
    -DCOMPUTE_OP_TEST=${COMPUTE_NAME_TST} \
    -DDISTRIBUTE_ALLOCATION_TEST=${DISTRIBUTED_ALLOCATE_NAME_TST} \
    -DDISTRIBUTE_DATA_TEST=${DISTRIBUTED_FREE_NAME_TST} \
    -DCOLLECTION_TEST=${DISTRIBUTED_DATA_NAME_TST} \
    -DFREE_MEMORY_TEST=${COLLECT_DATA_NAME_TST} \
    ${TEST_RIG} -o ${TEST_RIG}.o

#BUILD REFERENCE BASELINE 
${CC} -std=c99 -c\
    -DCOMPUTE_OP=${COMPUTE_NAME_REF} \
    -DDISTRIBUTE_ALLOCATION=${DISTRIBUTED_ALLOCATE_NAME_REF} \
    -DFREE_MEMORY=${DISTRIBUTED_FREE_NAME_REF} \
    -DDISTRIBUTE_DATA=${DISTRIBUTED_DATA_NAME_REF} \
    -DCOLLECTION=${COLLECT_DATA_NAME_REF} \
    ${BASELINE_VARIANT} -o ${BASELINE_VARIANT}.ref.o


#BUILD VARIANT 1
${CC} -std=c99 -c\
    -DCOMPUTE_OP=${COMPUTE_NAME_TST} \
    -DDISTRIBUTE_ALLOCATION=${DISTRIBUTED_ALLOCATE_NAME_TST} \
    -DFREE_MEMORY=${DISTRIBUTED_FREE_NAME_TST} \
    -DDISTRIBUTE_DATA=${DISTRIBUTED_DATA_NAME_TST} \
    -DCOLLECTION=${COLLECT_DATA_NAME_TST} \
    ${VARIANT_1} -o ${VARIANT_1}.o

#BUILD VARIANT 2
${CC} -std=c99 -c\
    -DCOMPUTE_OP=${COMPUTE_NAME_TST} \
    -DDISTRIBUTE_ALLOCATION=${DISTRIBUTED_ALLOCATE_NAME_TST} \
    -DFREE_MEMORY=${DISTRIBUTED_FREE_NAME_TST} \
    -DDISTRIBUTE_DATA=${DISTRIBUTED_DATA_NAME_TST} \
    -DCOLLECTION=${COLLECT_DATA_NAME_TST} \
    ${VARIANT_2} -o ${VARIANT_2}.o

#BUILD VARIANT 3
${CC} -std=c99 -c\
    -DCOMPUTE_OP=${COMPUTE_NAME_TST} \
    -DDISTRIBUTE_ALLOCATION=${DISTRIBUTED_ALLOCATE_NAME_TST} \
    -DFREE_MEMORY=${DISTRIBUTED_FREE_NAME_TST} \
    -DDISTRIBUTE_DATA=${DISTRIBUTED_DATA_NAME_TST} \
    -DCOLLECTION=${COLLECT_DATA_NAME_TST} \
    ${VARIANT_3} -o ${VARIANT_3}.o

#Build the test executables
${CC} ${CFLAGS} ${TEST_RIG}.o ${BASELINE_VARIANT}.ref.o ${VARIANT_1}.o -o ./run_test_variant01.x
${CC} ${CFLAGS} ${TEST_RIG}.o ${BASELINE_VARIANT}.ref.o ${VARIANT_2}.o -o ./run_test_variant02.x
${CC} ${CFLAGS} ${TEST_RIG}.o ${BASELINE_VARIANT}.ref.o ${VARIANT_3}.o -o ./run_test_variant03.x

echo "Build Test: complete"