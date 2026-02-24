#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>

class Sam3Context {
private:
    // Although these properties will eventually be sent to the PersistentSam3Model
    Ort::MemoryInfo cpuMemoryInfo;
    Ort::MemoryInfo cudaMemoryInfo;
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
public:
    Sam3Context(
        Ort::Env environment,
        Ort::SessionOptions sOptions,
        Ort::MemoryInfo cudaMemory,
        Ort::MemoryInfo cpuMemory
    ) : env(std::move(environment)),
        sessionOptions(std::move(sOptions)),
        cudaMemoryInfo(std::move(cudaMemory)),
        cpuMemoryInfo(std::move(cpuMemory)) {};

    const Ort::Env& getEnvironment() const;
    const Ort::SessionOptions& getSessionOptions() const;
    const Ort::MemoryInfo& getCudaMemoryInfo() const;
    const Ort::MemoryInfo& getCpuMemoryInfo() const;
};
