#pragma once

#include "GenericTensor.h"
#include "Sam3Context.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include <algorithm>

template<typename T>
class CPUTensor : public GenericTensor<T> {
protected:
    void releaseMemory() override {
        if (this->start == nullptr) {
            return;
        }

        delete[] this->start;
    }
private:
    CPUTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor) 
        : GenericTensor<T>(start, size, std::move(tensorShape), std::move(tensor)) {}

    static T* createCPUMemory(size_t elementCount) {
        T* ptr = new T[elementCount];
        std::memset(ptr, 0, elementCount * sizeof(T));
        
        return ptr;
    } 
public:
    // Unsafe (and probably shouldn't be public). Only use if you know what you're doing
    void copyToBuffer(const std::vector<T>& sourceBuffer) override {
        if (sourceBuffer.size() != this->size) {
            throw std::runtime_error(
                std::string("src should be the same size as tensor buffer. Target: ")
                + std::to_string(this->size) 
                + std::string(". Actual: ")
                + std::to_string(sourceBuffer.size())
            );
        }

        std::copy(sourceBuffer.begin(), sourceBuffer.end(), this->start);
    }

    std::vector<T> readBuffer() const override {
        std::vector<T> tempBuffer(this->size, 0);

        for (auto i = 0; i < this->size; ++i) {
            tempBuffer[i] = this->start[i];
        }

        return tempBuffer;
    }

    ~CPUTensor() {
        releaseMemory();
    }

    static std::unique_ptr<CPUTensor<T>> createCPUTensorWithTypeOverride(std::vector<int64_t> tensorSize, const Sam3Context& samContext, ONNXTensorElementDataType dataType) {
        auto numValues = GenericTensor<T>::getTensorCountFromShape(tensorSize);

        T* ptr = createCPUMemory(numValues);

        auto tensor = Ort::Value::CreateTensor(samContext.getCpuMemoryInfo(), ptr, numValues, tensorSize.data(), tensorSize.size(), dataType);
        return std::unique_ptr<CPUTensor<T>>(new CPUTensor<T>(ptr, numValues, std::move(tensorSize), std::move(tensor)));
    }

    static std::unique_ptr<CPUTensor<T>> createCPUTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext) {
        auto numValues = GenericTensor<T>::getTensorCountFromShape(tensorSize);

        T* ptr = createCPUMemory(numValues);

        auto tensor = Ort::Value::CreateTensor<T>(samContext.getCpuMemoryInfo(), ptr, numValues, tensorSize.data(), tensorSize.size());
        return std::unique_ptr<CPUTensor<T>>(new CPUTensor<T>(ptr, numValues, std::move(tensorSize), std::move(tensor)));
    }
};
