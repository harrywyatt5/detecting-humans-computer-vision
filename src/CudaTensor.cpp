#include "CudaTensor.h"

#include "GenericTensor.h"
#include <stdexcept>
#include <memory>
#include <cuda_runtime.h>

template<typename T>
void CudaTensor<T>::releaseMemory() {
    if (this->start == nullptr) {
        return;
    }

    changeCudaDevice(this->deviceId);

    auto freeResult = cudaFree((void**)&this->start);
    if (freeResult != cudaSuccess) {
        throw std::runtime_error("Could not free CUDA memory.... Reason: " + cudaGetErrorString(freeResult));
    }
}

template<typename T>
CudaTensor<T>::~CudaTensor() {
    this->releaseMemory();
}

template<typename T>
void CudaTensor<T>::copyToBuffer(const std::vector<T>& sourceBuffer) {
    if (sourceBuffer.size() != this->size) {
        throw std::runtime_error(
            "src should be the same size as tensor buffer. Target: "
            + std::to_string(this->size) 
            + ". Actual: " 
            + std::to_string(sourceBuffer.size())
        );
    }

    changeCudaDevice(this->deviceId);
    auto result = cudaMemcpy((void*)this->start, (void*)sourceBuffer.data(), this->size * sizeof(T), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        throw std::runtime_error("Could not copy " + std::to_string(this->size) + " bytes to CUDA device. Reason " + cudaGetErrorString(result));
    }
}

template<typename T>
void CudaTensor<T>::changeCudaDevice(const int deviceId) {
    auto changeResult = cudaSetDevice(this->deviceId);
    if (changeResult != cudaSuccess) {
        throw std::runtime_error("Could not change to CUDA device. Reason: " + cudaGetErrorString(changeResult));
    }
}

template<typename T>
std::unique_ptr<CudaTensor<T>> CudaTensor<T>::createCudaTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext) {
    size_t numValues = GenericTensor<T>::getTensorCountFromShape(tensorSize);

    T* ptr;

    changeCudaDevice(samContext.getDeviceId());
    auto result = cudaMalloc((void**)&ptr, numValues * sizeof(T));

    if (result != cudaSuccess) {
        throw std::runtime_error("Allocating to CUDA device failed!");
    }

    auto tensor = Ort::Value::CreateTensor<T>(gpuMemoryInfo, ptr, numValues, tensorSize.data(), tensorSize.size());
    return std::make_unique<CudaTensor<T>>(ptr, numValues, std::move(tensorSize), std::move(tensor));
}
