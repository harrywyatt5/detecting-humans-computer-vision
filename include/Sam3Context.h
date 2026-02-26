#pragma once

#include <memory>
#include <string>
#include <onnxruntime_cxx_api.h>

class Sam3Context {
private:
    // Although these properties will eventually be sent to the PersistentSam3Model
    Ort::MemoryInfo cpuMemoryInfo;
    Ort::MemoryInfo cudaMemoryInfo;
    int deviceId;
    std::string textEncoderPath;
    std::string visionEncoderPath;
    std::string decoderPath;
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
public:
    // You're probably looking for Sam3ContextBuilder.h
    Sam3Context(
        Ort::Env environment,
        Ort::SessionOptions sOptions,
        int deviceId,
        std::string textEncoderPath,
        std::string visionEncoderPath,
        std::string decoderPath,
        Ort::MemoryInfo cudaMemory,
        Ort::MemoryInfo cpuMemory
    ) : env(std::move(environment)),
        sessionOptions(std::move(sOptions)),
        deviceId(deviceId),
        textEncoderPath(std::move(textEncoderPath)),
        visionEncoderPath(std::move(visionEncoderPath)),
        decoderPath(std::move(decoderPath)),
        cudaMemoryInfo(std::move(cudaMemory)),
        cpuMemoryInfo(std::move(cpuMemory)) {}

    const Ort::Env& getEnvironment() const;
    const Ort::SessionOptions& getSessionOptions() const;
    const Ort::MemoryInfo& getCudaMemoryInfo() const;
    const Ort::MemoryInfo& getCpuMemoryInfo() const;
    int getDeviceId() const;
    const std::string& getTextEncoderPath() const;
    const std::string& getVisionEncoderPath() const;
    const std::string& getDecoderPath() const;
};
