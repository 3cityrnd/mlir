#!/bin/bash

FILE_PATH="$1"
FILE_NAME=$(basename ${FILE_PATH})
TEST_NAME=${FILE_NAME%.*}

DUMP_DIR="/artifacts/"

echo "Run test ${TEST_NAME}"

DUMP_DIR="${DUMP_DIR}/${TEST_NAME}_dump"
rm -fr $DUMP_DIR
mkdir $DUMP_DIR

export TRITON_DEBUG=1
export TORCH_LOGS="+all"
export TRITON_ENABLE_LLVM_DEBUG=1 
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_IR=ttir,llir,asm
export TRITON_DUMP_DIR="$DUMP_DIR/triton"
export TRITON_CACHE_DIR="$DUMP_DIR/triton_cache"
export TRITON_ALWAYS_COMPILE=1

export MLIR_ENABLE_DUMP=1 
export MLIR_DUMP_PATH="$DUMP_DIR/mlir_dump" 

LOG="$DUMP_DIR/run.log" 
touch $LOG
touch "${DUMP_DIR}/status"

python3 $1 > $LOG 2>&1  
RET_VAL=$?
if [ $RET_VAL -eq 0 ]; then
    echo "PASS"
    echo "PASS" > "${DUMP_DIR}/status"
else
    echo "FAIL"
    echo "FAIL" > "${DUMP_DIR}/status"
fi

exit ${RET_VAL}