#pragma once

#include "CudaTensor.h"
#include "CPUTensor.h"
#include <memory>

class OutputProcessor {
public:
    virtual void processOutput(
        const CudaTensor<float>& outputMasksTensor,
        const CPUTensor<float>& outputBoxesTensor,
        const CPUTensor<float>& outputLogitsTensor,
        const CPUTensor<float>& outputLogicTensor
    ) = 0;

    virtual ~OutputProcessor() = default;
};
