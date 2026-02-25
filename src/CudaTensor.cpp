#include "CudaTensor.h"

#include <stdexcept>
#include <cuda_runtime.h>

template<typename T>
void CudaTensor<T>::releaseMemory() override {
    if (start == nullptr) {
        return;
    }

    auto result = cudaFree((void**)&start);

    if (result != cudaSuccess) {
        throw std::runtime_error("Could not free CUDA memory...");
    }
}

template<typename T>
CudaTensor<T>::~CudaTensor() {
    this->releaseMemory();
}

template<typename T>
CudaTensor<T> CudaTensor::createCudaTensor<T>(std::vector<int64_t>& tensorSize, const Ort::MemoryInfo& gpuMemoryInfo) {
    size_t numValues = 1;
    for (size_t i = 0; i < tensorSize.size(); ++i) {
        numValues *= tensorSize[i];
    }

    T* ptr;
    auto result = cudaMalloc((void*)&ptr, numValues * sizeof(T));

    if (result != cudaSuccess) {
        throw std::runtime_error("Allocating to CUDA device failed!");
    }

    auto tensor = Ort::Value::CreateTensor<T>(gpuMemoryInfo, ptr, numValues, tensorSize.data(), tensorSize.size());
    return CudaTensor<T>(ptr, numValues, std::move(tensorSize), std::move(tensor));
}
