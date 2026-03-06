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
#include <iostream>

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
            // TODO: don't throw when releasing memory... We might be unwinding a call stack and if we throw again we will crash
            throw std::runtime_error(std::string("Could not free CUDA memory.... Reason: ") + cudaGetErrorString(freeResult));
        }
    }
private:
    int deviceId;
    CudaTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor, int deviceId) 
        : deviceId(deviceId), GenericTensor<T>(start, size, std::move(tensorShape), std::move(tensor)) {
            if (std::is_same<T, int64_t>::value) {
                std::cout << "Using int64_t on a CUDA tensor is not supported, and will likely be rejected by TensorRT. You should use a CPUTensor" << std::endl;
            }
    };
            
    static void changeCudaDevice(const int deviceId) {
        auto changeResult = cudaSetDevice(deviceId);
        if (changeResult != cudaSuccess) {
            throw std::runtime_error(
                std::string("Could not change to CUDA device. Reason: ") 
                + cudaGetErrorString(changeResult)
            );
        }
    }

    static void syncCudaDevice() {
        auto result = cudaDeviceSynchronize();

        if (result != cudaSuccess) {
            throw std::runtime_error(
                std::string("Could not sync with CUDA device before instruction. Reason: ")
                + cudaGetErrorString(result)
            );
        }
    }

    static T* createGpuMemory(size_t size) {
        T* ptr;
        auto result = cudaMalloc((void**)&ptr, size * sizeof(T));
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Allocating to CUDA device failed!");
        }

        auto populateResult = cudaMemset(ptr, 0, size * sizeof(T));

        if (populateResult != cudaSuccess) {
            throw std::runtime_error("Allocating to CUDA device failed! Could not fill buffer");
        }

        return ptr;
    }
public:
    // Unsafe (and probably shouldn't be public). Only use if you know what you're doing
    void copyToBuffer(const std::vector<T>& sourceBuffer) override {
        if (sourceBuffer.size() != this->size) {
            throw std::runtime_error(
                std::string("src should be the same size as tensor buffer. Target: ")
                + std::to_string(this->size * sizeof(T)) 
                + ". Actual: " 
                + std::to_string(sourceBuffer.size())
            );
        }

        changeCudaDevice(this->deviceId);
        syncCudaDevice();

        auto result = cudaMemcpy((void*)this->start, (void*)sourceBuffer.data(), this->size * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            throw std::runtime_error(
                std::string("Could not copy ")
                + std::to_string(this->size * sizeof(T)) 
                + std::string(" bytes to CUDA device. Reason ") 
                + cudaGetErrorString(result)
            );
        }
    }

    std::vector<T> readBuffer() const override {
        changeCudaDevice(this->deviceId);
        syncCudaDevice();

        std::vector<T> tempBuffer(this->size, 0);
        auto result = cudaMemcpy((void*)tempBuffer.data(), (void*)this->start, this->size * sizeof(T), cudaMemcpyDeviceToHost);

        if (result != cudaSuccess) {
            std::runtime_error(
                std::string("Could not read buffer on the GPU. Reason: ")
                + cudaGetErrorString(result)
            );
        }

        return tempBuffer;
    }

    void setCudaDeviceToTensor() const {
        changeCudaDevice(deviceId);
    }

    ~CudaTensor() {
        releaseMemory();
    }

    // Static
    static std::unique_ptr<CudaTensor<T>> createCudaTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext) {
        auto numValues = GenericTensor<T>::getTensorCountFromShape(tensorSize);

        changeCudaDevice(samContext.getDeviceId());
        T* ptr = createGpuMemory(numValues);

        // TODO: if CreateTensor throws, make sure to free cudaMalloc
        auto tensor = Ort::Value::CreateTensor<T>(samContext.getCudaMemoryInfo(), ptr, numValues, tensorSize.data(), tensorSize.size());
        return std::unique_ptr<CudaTensor<T>>(new CudaTensor<T>(ptr, numValues, std::move(tensorSize), std::move(tensor), samContext.getDeviceId()));
    }

    static std::unique_ptr<CudaTensor<T>> createCudaTensorWithTypeOverride(std::vector<int64_t> tensorSize, const Sam3Context& samContext, ONNXTensorElementDataType type) {
        auto numValues = GenericTensor<T>::getTensorCountFromShape(tensorSize);

        changeCudaDevice(samContext.getDeviceId());
        T* ptr = createGpuMemory(numValues);

        // TODO: if CreateTensor throws, make sure to free cudaMalloc
        auto tensor = Ort::Value::CreateTensor(samContext.getCudaMemoryInfo(), ptr, numValues, tensorSize.data(), tensorSize.size(), type);
        return std::unique_ptr<CudaTensor<T>>(new CudaTensor<T>(ptr, numValues, std::move(tensorSize), std::move(tensor), samContext.getDeviceId()));
    }
};
