#pragma once

#include "GenericTensor.h"

#include <vector>
#include <cstdint>

template<typename T>
class CudaTensor : public GenericTensor<T> {
public:
    CudaTensor(size_t numValues, std::vector<int64_t> tensorSize);
};
