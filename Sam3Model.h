#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>

class Sam3Model {
private:
	Ort::Env environment;
	Ort::SessionOptions sessionOptions;
	std::unique_ptr<Ort::Session> imageEncoderSession;
	std::unique_ptr<Ort::Session> textEncoderSession;
	std::unique_ptr<Ort::Session> decoderSession;
public:
	Sam3Model();
};
