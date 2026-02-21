#include "TensorRTProviderActuator.h"

#include <onnxruntime_cxx_api.h>
#include <string>

void TensorRTProviderActuator::setDeviceId(const uint8_t deviceId) {
    this->deviceId = deviceId;
}

void TensorRTProviderActuator::setGpuMemorySize(const long memorySize) {
    this->gpuMemorySize = memorySize;
}

void TensorRTProviderActuator::setModelCachePath(const std::string& str) {
    this->modelCachePath = str;
}

void TensorRTProviderActuator::useFP16Mode(const bool useMode) {
    this->enableFP16Mode = useMode;
}

void TensorRTProviderActuator::registerProvider(const Ort::SessionOptions& options) const {

}
