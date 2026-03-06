#pragma once

#include "GenericTensor.h"
#include "SessionInitialiser.h"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

class LanguageToken {
private:
    std::vector<int64_t> data;
    std::vector<uint8_t> attentionMask;

    // Private. Use createFromFile function
    LanguageToken(std::vector<int64_t> dataVector, std::vector<uint8_t> attentionMask);

    static int numOfTokens;
public:
    void populateAttentionMaskTensor(GenericTensor<int64_t>& attentionMaskTensor) const;
    void populateAttentionMaskTensor(GenericTensor<uint8_t>& attentionMaskTensor) const;
    void populateTextIdsTensor(GenericTensor<int64_t>& textIdsTensor) const;

    static std::unique_ptr<LanguageToken> createFromFile(const std::string& filePath);
};
