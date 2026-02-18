#pragma once

#include "OnnxSession.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

class OnnxSessionPartial {
private:
	std::string modelPath;
	std::shared_ptr<const std::vector<std::string>> inputNames;
	std::shared_ptr<const std::vector<std::string>> outputNames;
public:
	OnnxSessionPartial(
			const std::string& modelPath,
			std::shared_ptr<const std::vector<std::string>> inputNames,
			std::shared_ptr<const std::vector<std::string>> outputNames
	) : modelPath(modelPath), inputNames(inputNames), outputNames(outputNames) {};

	OnnxSession createSessionFromPartial(const Ort::SessionOptions& options, const Ort::Env& env) const;
};
