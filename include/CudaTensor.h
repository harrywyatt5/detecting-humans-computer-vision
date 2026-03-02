#pragma once

#include "GenericTensor.h"
#include "Sam3Context.h"

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <cstdint>

template<typename T>
class CudaTensor : public GenericTensor<T> {
protected:
    void releaseMemory() override {
        if (this->start == nullptr) {
            return;
        }

        changeCudaDevice(this->deviceId);

        auto freeResult = cudaFree((void*)this->start);
        if (freeResult != cudaSuccess) {
            throw std::runtime_error(std::string("Could not free CUDA memory.... Reason: ") + cudaGetErrorString(freeResult));
        }
    }
private:
    int deviceId;
    CudaTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor, int deviceId) 
            : deviceId(deviceId), GenericTensor<T>(start, size, std::move(tensorShape), std::move(tensor)) {};
            
    static void changeCudaDevice(const int deviceId) {
        auto changeResult = cudaSetDevice(deviceId);
        if (changeResult != cudaSuccess) {
            throw std::runtime_error(
                std::string("Could not change to CUDA device. Reason: ") 
                + cudaGetErrorString(changeResult)
            );
        }
    }
public:
    // Unsafe (and probably shouldn't be public). Only use if you know what you're doing
    void copyToBuffer(const std::vector<T>& sourceBuffer) override {
        if (sourceBuffer.size() != this->size) {
            throw std::runtime_error(
                std::string("src should be the same size as tensor buffer. Target: ")
                + std::to_string(this->size) 
                + ". Actual: " 
                + std::to_string(sourceBuffer.size())
            );
        }

        changeCudaDevice(this->deviceId);
        auto result = cudaMemcpy((void*)this->start, (void*)sourceBuffer.data(), this->size * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            throw std::runtime_error(
                std::string("Could not copy ")
                + std::to_string(this->size) 
                + std::string(" bytes to CUDA device. Reason ") 
                + cudaGetErrorString(result)
            );
        }
    }

    ~CudaTensor() {
        releaseMemory();
    }

    // Static
    static std::unique_ptr<CudaTensor<T>> createCudaTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext) {
        size_t numValues = GenericTensor<T>::getTensorCountFromShape(tensorSize);

        T* ptr;

        changeCudaDevice(samContext.getDeviceId());
        auto result = cudaMalloc((void**)&ptr, numValues * sizeof(T));

        if (result != cudaSuccess) {
            throw std::runtime_error("Allocating to CUDA device failed!");
        }

        auto tensor = Ort::Value::CreateTensor<T>(samContext.getCudaMemoryInfo(), ptr, numValues, tensorSize.data(), tensorSize.size());
        return std::unique_ptr<CudaTensor<T>>(new CudaTensor<T>(ptr, numValues, std::move(tensorSize), std::move(tensor), samContext.getDeviceId()));
    }
    static std::unique_ptr<CudaTensor<uint8_t>> createBoolCudaTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext) {
        size_t numValues = GenericTensor<uint8_t>::getTensorCountFromShape(tensorSize);

        uint8_t* ptr;

        changeCudaDevice(samContext.getDeviceId());
        auto result = cudaMalloc((void**)&ptr, numValues * sizeof(uint8_t));

        if (result != cudaSuccess) {
            throw std::runtime_error("Allocating to CUDA device failed!");
        }

        auto tensor = Ort::Value::CreateTensor(samContext.getCudaMemoryInfo(), ptr, numValues, tensorSize.data(), tensorSize.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        return std::unique_ptr<CudaTensor<uint8_t>>(new CudaTensor<uint8_t>(ptr, numValues, std::move(tensorSize), std::move(tensor), samContext.getDeviceId()));
    }
};
