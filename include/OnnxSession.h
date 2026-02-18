#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

class OnnxSession {
private:
	std::unique_ptr<Ort::Session> session;
	std::shared_ptr<const std::vector<std::string>> inputNames;
	std::shared_ptr<const std::vector<std::string>> outputNames;
public:
	// Passing a std::unique_ptr ensures that this object is the only owner of the session
	OnnxSession(std::unique_ptr<Ort::Session> sessionPtr, std::shared_ptr<const std::vector<std::string>> inputNames, std::shared_ptr<const std::vector<std::string>> outputNames);

	std::vector<Ort::Value> run(const std::vector<Ort::Value>& inputTensors);
	// Allow moving of values (disabled by default due to unique_ptr
	OnnxSession(OnnxSession&& other) = default;
	OnnxSession& operator=(OnnxSession&& other) = default;
};
