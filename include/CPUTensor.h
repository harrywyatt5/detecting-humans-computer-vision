#pragma once

#include "GenericTensor.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>

template<typename T>
class CPUTensor : public GenericTensor<T> {
protected:
    void releaseMemory() override;
private:
    CPUTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor) 
        : GenericTensor<T>(start, size, std::move(tensorShape), std::move(tensor)) {}
public:
    ~CPUTensor();

    template<typename E>
    static CPUTensor<E> createCPUTensor(std::vector<int64_t> tensorSize, const Ort::MemoryInfo& cpuMemoryInfo);
};
