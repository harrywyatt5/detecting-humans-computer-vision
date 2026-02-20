#pragma once

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <string>

class TensorRTProviderBuilder {
private:
	std::string cachePath = "./model_cache";
	int deviceId = 0;
	long gpuMemorySize = 4LL * 1024 * 1024 * 1024;
	bool useFP16 = false;
	bool useFP32NormFallback = false;
public:
	TensorRTProviderBuilder() {};
	TensorRTProviderBuilder& withCachePath(const std::string& newValue);
	TensorRTProviderBuilder& withDeviceId(const int newValue);
	TensorRTProviderBuilder& withGpuMemorySize(const long newValue);
	TensorRTProviderBuilder& isFP16Enabled(const bool newValue);
	TensorRTProviderBuilder& isFP32NormFallback(const bool newValue);
	
	OrtTensorRTProviderOptionsV2* build() const;
};
