#!/bin/bash

source ~/mlc/bin/activate

# Configure Ascend NPU environment (if using Ascend NPU)
if [ -d "/usr/local/Ascend" ]; then
    export CC=~/work/.conda/envs/cxx20/bin/aarch64-conda-linux-gnu-gcc
    export CXX=~/work/.conda/envs/cxx20/bin/aarch64-conda-linux-gnu-g++
    export AR=~/work/.conda/envs/cxx20/bin/aarch64-conda-linux-gnu-ar
    export LD=~/work/.conda/envs/cxx20/bin/aarch64-conda-linux-gnu-ld
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
    export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:$LD_LIBRARY_PATH
fi