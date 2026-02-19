#include "LanguageTensor.h"

#include <fstream>
#include <ios>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

std::vector<int64_t> LanguageTensor::tokenShape = {1, 32};

Ort::Value LanguageTensor::getInitialisedTensor(Ort::MemoryInfo& memoryInfo) {
    return Ort::Value::CreateTensor(memoryInfo, promptData.data(), promptData.size(), tokenShape.data(), tokenShape.size());
}

LanguageTensor LanguageTensor::loadFromFile(const std::string& fileName) {
    std::vector<int64_t> nums(32);
    std::ifstream file;
    file.open(fileName, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("File could not be opened");
    }

    // Cursor must be moved to the end else we get an incorrect read for tellg()
    file.seekg(0, std::ios::end);
    auto lengthOfFile = file.tellg();
    auto valueCount = file.tellg() / sizeof(int64_t);
    if ((lengthOfFile % sizeof(int64_t)) != 0 || valueCount != 32) {
        throw std::runtime_error("File must only contain 32 numbers");
    }

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(nums.data()), lengthOfFile);

    return LanguageTensor(std::move(nums));
}