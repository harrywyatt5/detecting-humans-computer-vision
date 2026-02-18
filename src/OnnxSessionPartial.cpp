#include "OnnxSessionPartial.h"

OnnxSession OnnxSessionPartial::createSessionFromPartial(const Ort::SessionOptions& options, const Ort::Env& env) const {
	std::unique_ptr<Ort::Session> newSession = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
	return OnnxSession(std::move(newSession), inputNames, outputNames);
}

