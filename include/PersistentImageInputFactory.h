#pragma once

#include "PersistentImageInput.h"

class PersistentImageInputFactory {
public:
    PersistentImageInputFactory() {}
    PersistentImageInput createPersistentImageInput(
        int inputImageSizeX,
        int inputImageSizeY, 
        int intermediateSizeX,
        int intermediateSizeY,
        const Sam3Context& samContext
    ) const;
};
