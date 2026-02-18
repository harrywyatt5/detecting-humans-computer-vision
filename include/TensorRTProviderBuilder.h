#pragma once

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <string>

class TensorRTProviderBuilder {
private:
	std::string cachePath = "./model_cache";
	int deviceId = 0;
	long gpuMemorySize = 6LL * 1024 * 1024 * 1024;
	bool useFP16 = true;
public:
	TensorRTProviderBuilder() {};
	void withCachePath(const std::string& newValue);
	void withDeviceId(const int newValue);
	void withGpuMemorySize(const long newValue);
	void isFP16Enabled(const bool newValue);
	
	OrtTensorRTProviderOptions build() const;
	bool mountToSessionOptions(Ort::SessionOptions& sessionOptions, const bool noFail) const;

};
