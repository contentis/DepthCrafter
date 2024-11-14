#!/bin/sh
TRT_ROOT=""

export PATH=${TRT_ROOT}/bin:§PATH
export LD_LIBRARY_PATH=${TRT_ROOT}/lib:$LD_LIBRARY_PATH

trtexec --onnx=onnx/backbone.onnx --stronglyTyped --timingCacheFile=onnx/cache.trt --shapes=sample:1x110x8x72x128,encoder_hidden_states:1x110x1024 --useCudaGraph
# --profilingVerbosity=detailed  --exportLayerInfo=layers.json