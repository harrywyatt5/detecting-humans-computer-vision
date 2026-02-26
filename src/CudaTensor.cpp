#include "CudaTensor.h"
#include "GenericTensor.h"

#include <stdexcept>
#include <cuda_runtime.h>

template<typename T>
void CudaTensor<T>::releaseMemory() {
    if (this->start == nullptr) {
        return;
    }

    auto result = cudaFree((void**)&this->start);

    if (result != cudaSuccess) {
        throw std::runtime_error("Could not free CUDA memory...");
    }
}

template<typename T>
CudaTensor<T>::~CudaTensor() {
    this->releaseMemory();
}

template<typename T>
template<typename E>
CudaTensor<E> CudaTensor<T>::createCudaTensor(std::vector<int64_t> tensorSize, const Ort::MemoryInfo& gpuMemoryInfo) {
    size_t numValues = GenericTensor<E>::getTensorCountFromShape(tensorSize);

    E* ptr;
    auto result = cudaMalloc((void**)&ptr, numValues * sizeof(E));

    if (result != cudaSuccess) {
        throw std::runtime_error("Allocating to CUDA device failed!");
    }

    auto tensor = Ort::Value::CreateTensor<E>(gpuMemoryInfo, ptr, numValues, tensorSize.data(), tensorSize.size());
    return CudaTensor<E>(ptr, numValues, std::move(tensorSize), std::move(tensor));
}
