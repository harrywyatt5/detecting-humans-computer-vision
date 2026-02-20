#include "TensorRTProviderBuilder.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <string>

TensorRTProviderBuilder& TensorRTProviderBuilder::withCachePath(const std::string& newValue) {
	this->cachePath = newValue;
	return *this;
}

TensorRTProviderBuilder& TensorRTProviderBuilder::withDeviceId(const int newValue) {
	this->deviceId = newValue;
	return *this;
}

TensorRTProviderBuilder& TensorRTProviderBuilder::withGpuMemorySize(const long newValue) {
	this->gpuMemorySize = newValue;
	return *this;
}

TensorRTProviderBuilder& TensorRTProviderBuilder::isFP16Enabled(const bool newValue) {
	this->useFP16 = newValue;
	return *this;
}

TensorRTProviderBuilder& TensorRTProviderBuilder::isFP32NormFallback(const bool newValue) {
	this->useFP32NormFallback = newValue;
	return *this;
}

OrtTensorRTProviderOptionsV2* TensorRTProviderBuilder::build() const {
	const auto& api = Ort::GetApi();
	OrtTensorRTProviderOptionsV2* rtProviderOptions = nullptr;
	Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&rtProviderOptions));

	std::vector<const char*> keys = {
		"device_id",
		"trt_fp16_enable",
		"trt_layer_norm_fp32_fallback",
		"trt_engine_cache_enable",
		"trt_engine_cache_path",
		"trt_max_workspace_size"
	};
	auto deviceIdString = std::to_string(deviceId);
	auto gpuMemorySizeString = std::to_string(gpuMemorySize);
	std::vector<const char*> values = {
		deviceIdString.c_str(),
		useFP16 ? "1" : "0",
		useFP32NormFallback ? "1" : "0",
		"1",
		cachePath.c_str(),
		gpuMemorySizeString.c_str()
	};

	Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
		rtProviderOptions,
		keys.data(),
		values.data(),
		keys.size()
	));
	
	// Ownership of this pointer is the responsibility of the caller
	return rtProviderOptions;
}

