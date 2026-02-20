#include "TensorRTProviderBuilder.h"

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

void TensorRTProviderBuilder::withCachePath(const std::string& newValue) {
	this->cachePath = newValue;
}

void TensorRTProviderBuilder::withDeviceId(const int newValue) {
	this->deviceId = newValue;
}

void TensorRTProviderBuilder::withGpuMemorySize(const long newValue) {
	this->gpuMemorySize = newValue;
}

void TensorRTProviderBuilder::isFP16Enabled(const bool newValue) {
	this->useFP16 = newValue;
}

void TensorRTProviderBuilder::isFP32NormFallback(const bool newValue) {
	this->useFP32NormFallback = newValue;
}

OrtTensorRTProviderOptionsV2 TensorRTProviderBuilder::build() const {
	const auto& api = Ort::GetApi();
	OrtTensorRTProviderOptionsV2* rtProviderOptions = nullptr;
	// TODO HERE RIGHT NOW
	OrtTensorRTProviderOptionsV2 rtProviderOptions{};
	rtProviderOptions.device_id = deviceId;
	rtProviderOptions.trt_fp16_enable = useFP16 ? 1 : 0;
	rtProviderOptions.trt_layer_norm_fp32_fallback = useFP32NormFallback ? 1 : 0;
	rtProviderOptions.trt_engine_cache_enable = 1;
	rtProviderOptions.trt_engine_cache_path = cachePath.c_str();
	rtProviderOptions.trt_max_workspace_size = gpuMemorySize;
	
	return rtProviderOptions;
}

bool TensorRTProviderBuilder::mountToSessionOptions(Ort::SessionOptions& sessionOptions, const bool noFail) const {
	auto rtProviderOptions = build();
	
	try {
		sessionOptions.AppendExecutionProvider_TensorRT(rtProviderOptions);
		return true;
	} catch (const std::exception& exception) {
		std::cerr << "TensorRT could not be used. Reason " << exception.what() << std::endl;
		if (noFail) {
			return false;
		} else {
			throw;
		}
	}
}

