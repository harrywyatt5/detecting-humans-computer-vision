#include "LanguageToken.h"

#include "GenericTensor.h"
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

int LanguageToken::numOfTokens = 32;
int64_t LanguageToken::endToken = 49407;
int64_t LanguageToken::missingToken = 0;

LanguageToken::LanguageToken(std::vector<int64_t> dataVector) : data(std::move(dataVector)) {
    if (data.size() != LanguageToken::numOfTokens) {
        throw std::runtime_error("LanguageToken should contain exactly " + std::to_string(LanguageToken::numOfTokens) + " numbers!");
    }
}

void LanguageToken::populateTensorsWithToken(GenericTensor<int64_t>& textIds, GenericTensor<int64_t>& attentionMask) const {
    if (textIds.getSize() != LanguageToken::numOfTokens) {
        throw std::runtime_error(
            "Incompatible textIds tensor provided. Tensor expected " 
            + std::to_string(textIds.getSize()) 
            + " int64s but has " 
            + std::to_string(LanguageToken::numOfTokens)
        );
    }

    if (attentionMask.getSize() != LanguageToken::numOfTokens) {
        throw std::runtime_error(
            "Incompatible attentionMask tensor provided. Tensor expected " 
            + std::to_string(textIds.getSize()) 
            + " int64s but has " 
            + std::to_string(LanguageToken::numOfTokens)
        );
    }

    std::vector<int64_t> attentionMaskValues(LanguageToken::numOfTokens, 0);

    for (auto i = 0; i < LanguageToken::numOfTokens; ++i) {
        if (data[i] != LanguageToken::endToken && data[i] != LanguageToken::missingToken) {
            attentionMaskValues[i] = 1;
        }
    }

    textIds.copyToBuffer(this->data);
    attentionMask.copyToBuffer(attentionMaskValues);
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
