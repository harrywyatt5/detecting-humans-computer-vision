#pragma once

#include "GenericTensor.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>

class CPUTensor : public GenericTensor {
protected:
    void releaseMemory() override;
private:
    CPUTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor) 
        : GenericTensor(start, size, std::move(tensorShape), std::move(tensor)) {}
        
};
