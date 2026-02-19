#pragma once

#include "AbstractTensor.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class LanguageTensor : public AbstractTensor {
private:
    std::vector<int64_t> promptData;

    LanguageTensor(std::vector<int64_t> promptDataArr) : promptData(std::move(promptDataArr)) {};

    static std::vector<int64_t> tokenShape;
public:
    Ort::Value getInitialisedTensor(Ort::MemoryInfo& memoryInfo);

    static const std::vector<int64_t>& getTokenDimensions() {
        return tokenShape;
    };
    static LanguageTensor loadFromFile(const std::string& fileName);
};
