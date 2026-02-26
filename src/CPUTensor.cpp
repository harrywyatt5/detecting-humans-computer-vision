#include "CPUTensor.h"

#include "Sam3Context.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <algorithm>

template<typename T>
void CPUTensor<T>::releaseMemory() {
    if (this->start == nullptr) {
        return;
    }

    free(&this->start);
}

template<typename T>
CPUTensor<T>::~CPUTensor() {
    this->releaseMemory();
}

template<typename T>
void CPUTensor<T>::copyToBuffer(const std::vector<T>& sourceBuffer) {
    if (sourceBuffer.size() != this->size) {
        throw std::runtime_error(
            "src should be the same size as tensor buffer. Target: "
            + std::to_string(this->size) 
            + ". Actual: " 
            + std::to_string(sourceBuffer.size())
        );
    }

    std::copy(sourceBuffer.begin(), sourceBuffer.end(), self->start);
}

template<typename T>
std::unique_ptr<CPUTensor<T>> CPUTensor<T>::createCPUTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext) {
    size_t numValues = GenericTensor<T>::getTensorCountFromShape(tensorSize);s

    T* ptr = (T*)malloc(numValues * sizeof(T));

    auto tensor = Ort::Value::CreateTensor<T>(samContext.getCpuMemoryInfo(), ptr, numValues, tensorSize.data(), tensorSize.size());
    return std::make_unique<CPUTensor<T>>(ptr, numValues, std::move(tensorSize), std::move(tensor));
}
