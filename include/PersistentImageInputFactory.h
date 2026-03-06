#pragma once

#include "PersistentImageInput.h"
#include <memory>

class PersistentImageInputFactory {
public:
    PersistentImageInputFactory() {}
    std::unique_ptr<PersistentImageInput> createPersistentImageInput(
        int inputImageSizeX,
        int inputImageSizeY, 
        int intermediateSizeX,
        int intermediateSizeY,
        const Sam3Context& samContext
    ) const;
};
