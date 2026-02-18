#pragma once

#include "TensorRTProviderBuilder.h"

#include "OnnxSession.h"
#include "OnnxSessionPartial.h"
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>
#include <optional>

class Sam3Model {
private:
	Ort::Env environment;
	Ort::SessionOptions sessionOptions;
	std::optional<OnnxSession> imageEncoderSession;
	std::optional<OnnxSession> textEncoderSession;
	std::optional<OnnxSession> decoderSession;
	Ort::AllocatorWithDefaultOptions allocator;
	
public:
	Sam3Model(const OnnxSessionPartial& imageEncoder, const OnnxSessionPartial& textEncoder, const OnnxSessionPartial& decoder, TensorRTProviderBuilder& trtBuilder);
};
