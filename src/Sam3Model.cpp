#include "Sam3Model.h"
#include "LanguageTensor.h"
#include "TensorRTProviderBuilder.h"

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <vector>

Sam3Model::Sam3Model(
		const OnnxSessionPartial& imageEncoder,
		const TensorRTProviderBuilder& imageConfig,
		const OnnxSessionPartial& textEncoder,
		const TensorRTProviderBuilder& textConfig,
		const OnnxSessionPartial& decoder,
		const TensorRTProviderBuilder& decoderConfig
) : environment(ORT_LOGGING_LEVEL_WARNING, "Sam3Model"), memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
	providerPointers.reserve(3);

	// The SessionOptions are constructed with their default constructor
	processProvider(imageEncoderOptions, imageConfig);
	processProvider(textEncoderOptions, textConfig);
	processProvider(decoderOptions, decoderConfig);

	imageEncoderSession = imageEncoder.createSessionFromPartial(imageEncoderOptions, environment);
	textEncoderSession = textEncoder.createSessionFromPartial(textEncoderOptions, environment);
	decoderSession = decoder.createSessionFromPartial(decoderOptions, environment);
}

void Sam3Model::processProvider(Ort::SessionOptions& options, const TensorRTProviderBuilder& providerConfigPtr) {
	auto providerOptionsPtr = providerConfigPtr.build();
	providerPointers.push_back(providerOptionsPtr);

	options.SetIntraOpNumThreads(8);
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	options.AppendExecutionProvider_TensorRT_V2(*providerOptionsPtr);
}

std::vector<Ort::Value> Sam3Model::runImageEncoder(TensorImage& image) {
	// We have to explicitly put the Ort::Value in a vector as this is the format the session expects (even though
	// we are only passing one value)
	std::vector<Ort::Value> vec(1);
	vec[0] = image.getInitialisedTensor(memoryInfo);

	return imageEncoderSession->run(vec);
}

std::vector<Ort::Value> Sam3Model::runTextEncoder(LanguageTensor& language) {
	std::vector<Ort::Value> vec(1);
	vec[0] = language.getInitialisedTensor(memoryInfo);

	return textEncoderSession->run(vec);
}

std::vector<Ort::Value> Sam3Model::runDecoder(std::vector<Ort::Value> tensors) {
	return decoderSession->run(tensors);
}

Sam3Model::~Sam3Model() {
	auto& api = Ort::GetApi();

	for (auto& pointer : providerPointers) {
		api.ReleaseTensorRTProviderOptions(pointer);
	}
}
