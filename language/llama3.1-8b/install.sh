#!/bin/bash
# Install MLPerf LoadGen
cd ../../loadgen
pip install pybind11
pip install datasets
CFLAGS="-std=c++14" python setup.py install
cd ../language/llama3.1-8b

# Download CNN dataset
if [ ! -f "cnn_eval.json" ]; then
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
fi