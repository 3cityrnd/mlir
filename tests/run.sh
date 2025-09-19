#!/bin/bash

f="$1"

if [ ! -f "${f}" ]; then
      echo "File does not exists ${f}"
      exit 
fi

D="dumpkern"

if [ -d "${D}" ]; then
  rm -rf ${D}
fi

export TRITON_ALWAYS_COMPILE=1
export LLVM_IR_ENABLE_DUMP=1
#export MLIR_ENABLE_DUMP=triton_gather
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=`pwd`/${D}
export MLIR_ENABLE_DIAGNOSTICS=remarks
export TRITON_ENABLE_LLVM_DEBUG=1
python ${f}

echo "*** DUMP -> ${D} ***"
	


