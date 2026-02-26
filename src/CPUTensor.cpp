#include "CPUTensor.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>

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
template<typename E>
CPUTensor<E> CPUTensor<T>::createCPUTensor(std::vector<int64_t> tensorSize, const Ort::MemoryInfo& cpuMemoryInfo) {
    size_t numValues = GenericTensor<E>::getTensorCountFromShape(tensorSize);

    E* ptr = (E*)malloc(numValues * sizeof(E));

    auto tensor = Ort::Value::CreateTensor<E>(cpuMemoryInfo, ptr, numValues, tensorSize.data(), tensorSize.size());
    return CPUTensor<E>(ptr, numValues, std::move(tensorSize), std::move(tensor));
}
