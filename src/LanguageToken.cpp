#include "LanguageToken.h"

#include "GenericTensor.h"
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

int LanguageToken::numOfTokens = 32;

LanguageToken::LanguageToken(std::vector<int64_t> dataVector) : data(std::move(dataVector)) {
    if (data.size() != LanguageToken::numOfTokens) {
        throw std::runtime_error("LanguageToken should contain exactly " + std::to_string(LanguageToken::numOfTokens) + " numbers!");
    }
}

void LanguageToken::populateTensorWithToken(GenericTensor<int64_t>& tensor) const {
    if (LanguageToken::numOfTokens != tensor.getSize()) {
        throw std::runtime_error(
            "Incompatible tensor provided. Tensor expected " 
            + std::to_string(tensor.getSize()) 
            + " int64s but has " 
            + std::to_string(LanguageToken::numOfTokens)
        );
    }

    tensor.copyToBuffer(this->data);
}

LanguageToken LanguageToken::createFromFile(const std::string& filePath) {
    auto bytesToRead = LanguageToken::numOfTokens * sizeof(int64_t);
    std::vector<int64_t> arr(LanguageToken::numOfTokens);
    std::ifstream fileBuffer(filePath, std::ios::in | std::ios::binary);

    // Throw if the file could not be found
    if (!fileBuffer.is_open()) {
        throw std::runtime_error("File containing LanguageToken was not found");
    }

    fileBuffer.seekg(0, std::ios::end);
    auto byteCount = fileBuffer.tellg();

    if (byteCount != bytesToRead) {
        throw std::runtime_error("File does not have exactly " + std::to_string(LanguageToken::numOfTokens) + " tokens!");
    }

    fileBuffer.seekg(0, std::ios::beg);
    fileBuffer.read(reinterpret_cast<char*>(arr.data()), bytesToRead);
    fileBuffer.close();

    return LanguageToken(std::move(arr));
}
