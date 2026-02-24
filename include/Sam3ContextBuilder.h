#pragma once

#include "Sam3Context.h"

#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

class Sam3ContextBuilder {
private:
    std::vector<const char*> tensorRTOptionsNames;
    std::vector<std::string> tensorRTOptions;
    OrtCUDAProviderOptions cudaOptions;

    std::string applicationName;
    int maxCPUThreads;
    int deviceId;
    OrtLoggingLevel loggingLevel;
    GraphOptimizationLevel optimisationLevel;
public:
    Sam3ContextBuilder();

    void withApplicationName(const std::string& name);
    void withLoggingLevel(const OrtLoggingLevel& loggingLevel);
    void withGraphOptimistionLevel(const GraphOptimizationLevel& optimisationLevel);
    void withDeviceId(const int deviceId);
    void withFP16Enabled(const bool enabled);
    void withCPUThreadMax(const int count); 
    void withMaxGPUMemory(const uint64_t max);
    void withEngineCacheDir(const std::string& location);
    Sam3Context build() const;

    ~Sam3ContextBuilder() = default;
};
