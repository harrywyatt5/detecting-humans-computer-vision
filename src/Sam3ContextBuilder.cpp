#include "Sam3ContextBuilder.h"

#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

Sam3ContextBuilder::Sam3ContextBuilder() {
    applicationName = "Sam3Model";
    textEncoderPath = "./sam3-onnx/text-encoder-fp16.onnx";
    visionEncoderPath = "./sam3-onnx/vision-encoder-fp16.onnx";
    decoderPath = "./sam3-onnx/geo-encoder-mask-decoder-fp16.onnx";
    loggingLevel = ORT_LOGGING_LEVEL_WARNING;
    optimisationLevel = GraphOptimizationLevel::ORT_ENABLE_ALL;
    deviceId = 0;
    cudaOptions.device_id = 0;
    maxCPUThreads = 1;
    batchLimit = 1;
    numBoxesLimit = 1;

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

Sam3ContextBuilder& Sam3ContextBuilder::withApplicationName(const std::string& name) {
    applicationName = name;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withLoggingLevel(const OrtLoggingLevel& loggingLevel) {
    this->loggingLevel = loggingLevel;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withGraphOptimistionLevel(const GraphOptimizationLevel& optimisationLevel) {
    this->optimisationLevel = optimisationLevel;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withDeviceId(const int deviceId) {
    this->deviceId = deviceId;
    cudaOptions.device_id = deviceId;
    tensorRTOptions[0] = std::to_string(deviceId);
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withFP16Enabled(const bool enabled) {
    tensorRTOptions[1] = enabled ? "1" : "0";
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withCPUThreadMax(const int count) {
    maxCPUThreads = count;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withMaxGPUMemory(const uint64_t max) {
    tensorRTOptions[2] = std::to_string(max);
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withEngineCacheDir(const std::string& location) {
    tensorRTOptions[4] = location;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withTextEncoderPath(const std::string& location) {
    textEncoderPath = location;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withVisionEncoderPath(const std::string& location) {
    visionEncoderPath = location;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withDecoderPath(const std::string& location) {
    decoderPath = location;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withBatchLimit(const int64_t count) {
    batchLimit = count;
    return *this;
}

Sam3ContextBuilder& Sam3ContextBuilder::withNumBoxesLimit(const int64_t count) {
    numBoxesLimit = count;
    return *this;
}

Sam3Context Sam3ContextBuilder::build() const {
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

    // 0 means that batch is unbounded, and can be any value
    if (batchLimit > 0) {
        sessionOptions.AddFreeDimensionOverrideByName("batch", batchLimit);
    }

    if (numBoxesLimit > 0) {
        sessionOptions.AddFreeDimensionOverrideByName("num_boxes", numBoxesLimit);
    }

    return Sam3Context(
        std::move(env),
        std::move(sessionOptions),
        deviceId,
        std::move(textEncoderPath),
        std::move(visionEncoderPath),
        std::move(decoderPath),
        Ort::MemoryInfo("Cuda", OrtDeviceAllocator, deviceId, OrtMemTypeDefault),
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
}
