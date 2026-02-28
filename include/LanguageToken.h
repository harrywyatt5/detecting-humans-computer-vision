#pragma once

#include "GenericTensor.h"

#include <vector>
#include <string>
#include <cstdint>

class LanguageToken {
private:
    std::vector<int64_t> data;

    // Private. Use createFromFile or createFromBuffer static functions
    LanguageToken(std::vector<int64_t> dataVector);

    static int numOfTokens;
public:
    void populateTensorWithToken(GenericTensor<int64_t>& targetTesnsor) const;

    static LanguageToken createFromFile(const std::string& filePath);
    // TODO: implement createFromBuffer
};
