#include "OnnxSession.h"
#include <string>

OnnxSession::OnnxSession(std::unique_ptr<Ort::Session> sessionPtr, std::shared_ptr<const std::vector<std::string>> inputNames, std::shared_ptr<const std::vector<std::string>> outputNames) 
	: session(std::move(sessionPtr)), inputNames(inputNames), outputNames(outputNames) {}

std::vector<Ort::Value> OnnxSession::run(const std::vector<Ort::Value>& inputTensors) {
	// Sanity check, make sure same number as input tensors as names
	if (inputTensors.size() != inputNames->size()) {
		throw std::runtime_error("Tensor and name mismatch. There are " + std::to_string(inputTensors.size()) + " tensors and " + std::to_string(inputNames->size()) + " names!");
	}

	std::vector<const char*> inputCStrings;
	inputCStrings.reserve(inputNames->size());
	std::vector<const char*> outputCStrings;
	outputCStrings.reserve(outputNames->size());

	for (const auto& str : *inputNames) {
		inputCStrings.push_back(str.c_str());
	}
	for (const auto& str : *outputNames) {
		outputCStrings.push_back(str.c_str());
	}

	return session->Run(
			Ort::RunOptions{nullptr},
			inputCStrings.data(),
			inputTensors.data(),
			inputTensors.size(),
			outputCStrings.data(),
			outputCStrings.size()
	);
}
