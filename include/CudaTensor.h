#pragma once

#include "GenericTensor.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>

template<typename T>
class CudaTensor : public GenericTensor<T> {
protected:
    void releaseMemory() override;
private:
    CudaTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor) 
            : GenericTensor(start, size, std::move(tensorShape), std::move(tensor)) {};
public:
    ~CudaTensor();

    // Static
    static CudaTensor createCudaTensor<T>(std::vector<int64_t>& tensorSize, const Ort::MemoryInfo& gpuMemoryInfo);
};
