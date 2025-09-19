#!/bin/bash
# This script runs inside the Docker container
# Usage: ./ci_build_job.sh <commit_id>

COMMIT_ID=$1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

echo "Sourcing conda..."
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda init
conda activate py311_test

git config --global http.sslVerify "false"

export LLVM_INSTALL_PREFIX=/root/llvm-install

echo "Building stack for commit $COMMIT_ID"

cd /root/triton-ascend
git fetch origin
git checkout -f "$COMMIT_ID"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Couldn't checkout triton ascend. Exiting"
  exit 1
fi

git submodule update --init

LLVM_HASH=$(cat /root/triton-ascend/llvm-hash.txt)
echo "Building LLVM with $LLVM_HASH"

cd /root/llvm-project/
git checkout -f "$LLVM_HASH"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Couldn't checkout llvm repo ascend. Exiting"
  exit 1
fi

rm -rf /root/llvm-project/build
mkdir /root/llvm-project/build

cd /root/llvm-project/build
cmake ../llvm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX} \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
ninja install

echo "Building Triton Ascend with $COMMIT_ID"
cd /root/triton-ascend
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
DEBUG=1 \
TRITON_PLUGIN_DIRS=./ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install

echo "Build finished successfully!"
exit 0
