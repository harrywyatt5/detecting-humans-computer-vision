#include "Sam3Model.h"
#include "LanguageTensor.h"

#include <onnxruntime_cxx_api.h>
#include <vector>

Sam3Model::Sam3Model(
		const OnnxSessionPartial& imageEncoder,
		const OnnxSessionPartial& textEncoder,
		const OnnxSessionPartial& decoder,
		TensorRTProviderBuilder& trtBuilder
) : environment(ORT_LOGGING_LEVEL_WARNING, "Sam3Model"), memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
	sessionOptions.SetIntraOpNumThreads(8);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	trtBuilder.mountToSessionOptions(sessionOptions, false);

	imageEncoderSession = imageEncoder.createSessionFromPartial(sessionOptions, environment);
	textEncoderSession = textEncoder.createSessionFromPartial(sessionOptions, environment);
	decoderSession = decoder.createSessionFromPartial(sessionOptions, environment);
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