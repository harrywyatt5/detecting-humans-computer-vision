#include "Sam3ContextBuilder.h"

#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

Sam3ContextBuilder::Sam3ContextBuilder() {
    applicationName = "Sam3Model";
    loggingLevel = ORT_LOGGING_LEVEL_WARNING;
    optimisationLevel = GraphOptimzationLevel::ORT_ENABLE_ALL;
    cudaOptions.device_id = 0;
    maxCPUThreads = 1;

    tensorRTOptionsNames = {
        "device_id",
        "trt_fp16_enable",
        "trt_max_workspace_size",
        "trt_engine_cache_enable",
        "trt_engine_cache_path"
    };
    tensorRTOptions = {
        "0",
        "1",
        "6442450944", // 6GB
        "1",
        "./trt_cache"
    };
}

void Sam3ContextBuilder::withApplicationName(const std::string& name) {
    applicationName = name;
}

void Sam3ContextBuilder::withLoggingLevel(const OrtLoggingLevel& loggingLevel) {
    this->loggingLevel = loggingLevel;
}

void Sam3ContextBuilder::withGraphOptimistionLevel(const GraphOptimizationLevel& optimisationLevel) {
    this->optimisationLevel = optimisationLevel;
}

void Sam3ContextBuilder::withDeviceId(const int deviceId) {
    this->deviceId = deviceId;
    cudaOptions.device_id = deviceId;
    tensorRTOptions[0] = std::to_string(deviceId);
}

void Sam3ContextBuilder::withFP16Enabled(const bool enabled) {
    tensorRTOptions[1] = enabled ? "1" : "0";
}

void Sam3ContextBuilder::withCPUThreadMax(const int count) {
    maxCPUThreads = count;
}

void Sam3ContextBuilder::withMaxGPUMemory(const uint64_t max) {
    tensorRTOptions[2] = std::to_string(max);
}

void Sam3ContextBuilder::withEngineCacheDir(const std::string& location) {
    tensorRTOptions[4] = location;
}

Sam3Context build() const {
    Ort::Env env(loggingLevel, applicationName.c_str());
    auto api = Ort::GetApi();

    Ort::SessionOptions sessionOptions;
    OrtTensorRTProviderOptionsV2* tensorOptions = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorOptions));

    // We have to create an array of the values to feed to the provider options
    // std::string doesn't do! We need const char*
    std::vector<const char*> tensorValueArray;
    for (auto i = 0; i < tensorRTOptions.size(); ++i) {
        tensorValueArray.push_back(tensorRTOptions[i].c_str());
    }

    Ort::ThrowOnError(
        api.UpdateTensorRTProviderOptions(
            tensorOptions,
            tensorRTOptionsNames.data(),
            tensorValueArray.data(),
            tensorRTOptionsNames.size()
        )
    );
    sessionOptions.AppendExecutionProvider_TensorRT_V2(*tensorOptions);
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    api.ReleaseTensorRTProviderOptions(tensorOptions);

    sessionOptions.SetGraphOptimizationLevel(optimisationLevel);
    sessionOptions.SetIntraOpNumThreads(maxCPUThreads);
    sessionOptions.SetInterOpNumThreads(maxCPUThreads);

    return Sam3Context(
        env,
        sessionOptions,
        Ort::MemoryInfo("Cuda", OrtDeviceAllocator, deviceId, OrtMemTypeDefault),
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
}
