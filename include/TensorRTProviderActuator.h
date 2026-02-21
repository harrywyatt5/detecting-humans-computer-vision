#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>

#include "ProviderActuator.h"

class TensorRTProviderActuator : public ProviderActuator {
private:
    std::string modelCachePath = "./model_cache";
    uint8_t deviceId = 0;
    long gpuMemorySize = 6LL * 1024 * 1024 * 1024; // 6GB
    bool enableFP16Mode = true;
public:
    TensorRTProviderActuator() {};
    void setModelCachePath(const std::string& modelPath);
    void setDeviceId(const uint8_t deviceId);
    void setGpuMemorySize(const long memorySize);
    void useFP16Mode(const bool doUseFP16);

    void registerProvider(const Ort::SessionOptions& sessionOptions) const;
};
