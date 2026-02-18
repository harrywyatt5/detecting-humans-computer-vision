#include "Sam3Model.h"

Sam3Model::Sam3Model(
		const OnnxSessionPartial& imageEncoder,
		const OnnxSessionPartial& textEncoder,
		const OnnxSessionPartial& decoder,
		TensorRTProviderBuilder& trtBuilder
) : environment(ORT_LOGGING_LEVEL_WARNING, "Sam3Model") {

	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	trtBuilder.mountToSessionOptions(sessionOptions, false);

	imageEncoderSession = imageEncoder.createSessionFromPartial(sessionOptions, environment);
	textEncoderSession = textEncoder.createSessionFromPartial(sessionOptions, environment);
	decoderSession = decoder.createSessionFromPartial(sessionOptions, environment);
}
