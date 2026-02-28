#include "LanguageToken.h"

#include "GenericTensor.h"
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

int LanguageToken::numOfTokens = 32;

void LanguageToken::populateTensorWithToken(GenericTensor<int64_t>& tensor) const {
    long bufferSize = LanguageToken::numOfTokens * sizeof(int64_t);
    if (bufferSize != tensor.getSizeInBytes()) {
        throw std::runtime_error(
            "Incompatible tensor provided. Tensor expected " 
            + std::to_string(tensor.getSizeInBytes()) 
            + " bytes but has " 
            + std::to_string(bufferSize)
        );
    }

    tensor.copyToBuffer(this->data);
}

LanguageToken LanguageToken::createFromFile(const std::string& filePath) {
    long bytesToRead = LanguageToken::numOfTokens * sizeof(int64_t);
    int64_t* arr = new int64_t[LanguageToken::numOfTokens];
    std::ifstream fileBuffer(filePath, std::ios::in | std::ios::binary);

    // Throw if the file could not be found
    if (!fileBuffer.is_open()) {
        throw std::runtime_error("File containing LanguageToken was not found");
    }

    fileBuffer.seekg(std::ios::end);
    long byteCount = fileBuffer.tellg();

    if (byteCount != bytesToRead) {
        throw std::runtime_error("File does not have exactly " + std::to_string(LanguageToken::numOfTokens) + " tokens!");
    }

    fileBuffer.seekg(std::ios::beg);
    fileBuffer.read(reinterpret_cast<char*>(arr), bytesToRead);
    fileBuffer.close();

    return LanguageToken(std::vector<int64_t>(arr, arr + LanguageToken::numOfTokens));
}
