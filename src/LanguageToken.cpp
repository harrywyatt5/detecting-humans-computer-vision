#include "LanguageToken.h"

#include "GenericTensor.h"
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <iostream>

int LanguageToken::numOfTokens = 32;

LanguageToken::LanguageToken(std::vector<int64_t> dataVector, std::vector<uint8_t> attentionMaskInput) 
    : data(std::move(dataVector)), attentionMask(std::move(attentionMaskInput)) 
{
    if (data.size() != LanguageToken::numOfTokens) {
        throw std::runtime_error("LanguageToken should contain exactly " + std::to_string(LanguageToken::numOfTokens) + " numbers!");
    }

    if (attentionMask.size() != LanguageToken::numOfTokens) {
        throw std::runtime_error("AttentionMask should contain exactly " + std::to_string(LanguageToken::numOfTokens) + " numbers!");
    }
}

void LanguageToken::populateAttentionMaskTensor(GenericTensor<uint8_t>& attentionMaskTensor) const {
    if (attentionMaskTensor.getSize() != LanguageToken::numOfTokens) {
        throw std::runtime_error(
            "Incompatible attention_mask tensor provided. Tensor expected " 
            + std::to_string(attentionMaskTensor.getSize()) 
            + " uint8s but has " 
            + std::to_string(LanguageToken::numOfTokens)
        );
    }

    attentionMaskTensor.copyToBuffer(this->attentionMask);
}

void LanguageToken::populateAttentionMaskTensor(GenericTensor<int64_t>& attentionMaskTensor) const {
    if (attentionMaskTensor.getSize() != LanguageToken::numOfTokens) {
        throw std::runtime_error(
            "Incompatible attention_mask tensor provided. Tensor expected " 
            + std::to_string(attentionMaskTensor.getSize()) 
            + " int64s but has " 
            + std::to_string(LanguageToken::numOfTokens)
        );
    }

    std::vector<int64_t> attentionMask64(LanguageToken::numOfTokens, 0);
    for (auto i = 0; i < LanguageToken::numOfTokens; ++i) {
        attentionMask64[i] = static_cast<int64_t>(attentionMask[i]);
    }

    attentionMaskTensor.copyToBuffer(attentionMask64);
}

void LanguageToken::populateTextIdsTensor(GenericTensor<int64_t>& textIdsTensor) const {
    if (textIdsTensor.getSize() != LanguageToken::numOfTokens) {
        throw std::runtime_error(
            "Incompatible text_ids tensor provided. Tensor expected " 
            + std::to_string(textIdsTensor.getSize()) 
            + " int64_t but has " 
            + std::to_string(LanguageToken::numOfTokens)
        );
    }

    textIdsTensor.copyToBuffer(this->data);
}

std::unique_ptr<LanguageToken> LanguageToken::createFromFile(const std::string& filePath) {
    auto bytesToRead = LanguageToken::numOfTokens * sizeof(int64_t);
    std::vector<int64_t> arr(LanguageToken::numOfTokens);
    std::vector<uint8_t> attentionMaskBuff(LanguageToken::numOfTokens, 0);

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

    // Read through the array buffer and check which values in the attention mask we have to flip
    for (auto i = 0; i < LanguageToken::numOfTokens; ++i) {
        if (arr[i] != 0) {
            attentionMaskBuff[i] = 1;
        }
    }

    return std::unique_ptr<LanguageToken>(new LanguageToken(std::move(arr), std::move(attentionMaskBuff)));
}
