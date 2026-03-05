#pragma once

#include "Sam3Context.h"

#include <string>
#include <vector>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

class Sam3ContextBuilder {
private:
    std::vector<const char*> tensorRTOptionsNames;
    std::vector<std::string> tensorRTOptions;
    OrtCUDAProviderOptions cudaOptions;

    std::string applicationName;
    std::string textEncoderPath;
    std::string visionEncoderPath;
    std::string decoderPath;
    int maxCPUThreads;
    int deviceId;
    int64_t batchLimit;
    int64_t numBoxesLimit;
    OrtLoggingLevel loggingLevel;
    GraphOptimizationLevel optimisationLevel;
public:
    Sam3ContextBuilder();

    Sam3ContextBuilder& withApplicationName(const std::string& name);
    Sam3ContextBuilder& withLoggingLevel(const OrtLoggingLevel& loggingLevel);
    Sam3ContextBuilder& withGraphOptimistionLevel(const GraphOptimizationLevel& optimisationLevel);
    Sam3ContextBuilder& withDeviceId(const int deviceId);
    Sam3ContextBuilder& withFP16Enabled(const bool enabled);
    Sam3ContextBuilder& withCPUThreadMax(const int count); 
    Sam3ContextBuilder& withMaxGPUMemory(const uint64_t max);
    Sam3ContextBuilder& withEngineCacheDir(const std::string& location);
    Sam3ContextBuilder& withTextEncoderPath(const std::string& location);
    Sam3ContextBuilder& withVisionEncoderPath(const std::string& location);
    Sam3ContextBuilder& withDecoderPath(const std::string& location);
    Sam3ContextBuilder& withBatchLimit(const int64_t count);
    Sam3ContextBuilder& withNumBoxesLimit(const int64_t count);
    Sam3Context build() const;

    ~Sam3ContextBuilder() = default;
};
