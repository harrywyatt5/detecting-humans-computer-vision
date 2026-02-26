#pragma once

#include "GenericTensor.h"
#include "Sam3Context.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <cstdint>

template<typename T>
class CudaTensor : public GenericTensor<T> {
protected:
    void releaseMemory() override;
private:
    int deviceId;
    CudaTensor(T* start, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor, int deviceId) 
            : deviceId(deviceId), GenericTensor<T>(start, size, std::move(tensorShape), std::move(tensor)) {};
    static void changeCudaDevice(const int deviceId);
public:
    // Unsafe (and probably shouldn't be public). Only use if you know what you're doing
    void copyToBuffer(const std::vector<T>& sourceBuffer) override;
    ~CudaTensor();

    // Static
    static std::unique_ptr<CudaTensor<T>> createCudaTensor(std::vector<int64_t> tensorSize, const Sam3Context& samContext);
};
