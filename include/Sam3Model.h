#pragma once

#include "TensorImage.h"
#include "TensorRTProviderBuilder.h"

#include "OnnxSession.h"
#include "OnnxSessionPartial.h"
#include "LanguageTensor.h"
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <optional>

class Sam3Model {
private:
	Ort::Env environment;
	Ort::SessionOptions sessionOptions;
	Ort::MemoryInfo memoryInfo;
	std::optional<OnnxSession> imageEncoderSession;
	std::optional<OnnxSession> textEncoderSession;
	std::optional<OnnxSession> decoderSession;
	Ort::AllocatorWithDefaultOptions allocator;
	
public:
	Sam3Model(const OnnxSessionPartial& imageEncoder, const OnnxSessionPartial& textEncoder, const OnnxSessionPartial& decoder, TensorRTProviderBuilder& trtBuilder);
	
	std::vector<Ort::Value> runImageEncoder(TensorImage& image);
	// TODO: REMOVE ME! This is very temp so that we can extract MemoryINFO and use it in the same loop for the skeleton code
	Ort::MemoryInfo* getMemoryInfo_Unsafe() {
		return &memoryInfo;
	};
	// TODO: add C++ operands for other encoders/decoders
	std::vector<Ort::Value> runTextEncoder(LanguageTensor& language);
	// TODO: so temp...
	std::vector<Ort::Value> runDecoder(std::vector<Ort::Value> tensors);
};
