#include "Sam3Context.h"

#include <onnxruntime_cxx_api.h>

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
