#pragma once

#include "onnxruntime_cxx_api.h"

class AbstractTensor {
public:
    virtual Ort::Value getInitialisedTensor(Ort::MemoryInfo& memoryInfo) = 0;
};
