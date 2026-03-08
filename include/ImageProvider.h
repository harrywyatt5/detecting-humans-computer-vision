#pragma once

#include "CudaTensor.h"

class ImageProvider {
public:
    virtual void writeImageToCudaTensor(CudaTensor<float>& cudaTensor) = 0;
    virtual int getOriginalX() const = 0;
    virtual int getOriginalY() const = 0;
};
