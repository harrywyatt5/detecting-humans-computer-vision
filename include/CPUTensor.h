#pragma once

#include "GenericTensor.h"
#include "Sam3Context.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <cstdint>

template<typename T>
class CPUTensor : public GenericTensor<T> {
protected:
    void releaseMemory() override;
private:
    CPUTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor) 
        : GenericTensor<T>(start, size, std::move(tensorShape), std::move(tensor)) {}
public:
    // Unsafe (and probably shouldn't be public). Only use if you know what you're doing
    void copyToBuffer(const std::vector<T>& sourceBuffer) override;
    ~CPUTensor();

    static std::unique_ptr<CPUTensor<T>> createCPUTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext);
};
