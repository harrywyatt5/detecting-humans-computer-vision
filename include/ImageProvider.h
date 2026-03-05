#pragma once

#include "CudaTensor.h"

class ImageProvider {
public:
    virtual void writeImageToCudaTensor(CudaTensor& cudaTensor) = 0;
};
