#include "TensorRTProviderBuilder.h"

#include <iostream>

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

OrtTensorRTProviderOptions TensorRTProviderBuilder::build() const {
	OrtTensorRTProviderOptions rtProviderOptions{};
	rtProviderOptions.device_id = deviceId;
	rtProviderOptions.trt_fp16_enable = useFP16 ? 1 : 0;
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

