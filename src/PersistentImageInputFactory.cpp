#include "PersistentImageInputFactory.h"

#include "PersistentImageInput.h"
#include <memory>

std::unique_ptr<PersistentImageInput> PersistentImageInputFactory::createPersistentImageInput(
        int inputImageSizeX,
        int inputImageSizeY, 
        int intermediateSizeX,
        int intermediateSizeY,
        const Sam3Context& samContext
) const {
    return std::make_unique<PersistentImageInput>(
        inputImageSizeX,
        inputImageSizeY,
        intermediateSizeX,
        intermediateSizeY,
        samContext.getDeviceId()
    );
}
