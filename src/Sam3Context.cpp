#include "Sam3Context.h"

#include <onnxruntime_cxx_api.h>
#include <string>

const Ort::Env& Sam3Context::getEnvironment() const {
    return env;
}

const Ort::SessionOptions& Sam3Context::getSessionOptions() const {
    return sessionOptions;
}

const Ort::MemoryInfo& Sam3Context::getCudaMemoryInfo() const {
    return cudaMemoryInfo;
}

const Ort::MemoryInfo& Sam3Context::getCpuMemoryInfo() const {
    return cpuMemoryInfo;
}

int Sam3Context::getDeviceId() const {
    return deviceId;
}

const std::string& Sam3Context::getTextEncoderPath() const {
    return textEncoderPath;
}

const std::string& Sam3Context::getVisionEncoderPath() const {
    return visionEncoderPath;
}

const std::string& Sam3Context::getDecoderPath() const {
    return decoderPath;
}
