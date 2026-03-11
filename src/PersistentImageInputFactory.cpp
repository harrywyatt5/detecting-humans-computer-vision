#include "PersistentImageInputFactory.h"

#include "PersistentImageInput.h"

PersistentImageInput PersistentImageInputFactory::createPersistentImageInput(
        int inputImageSizeX,
        int inputImageSizeY, 
        int intermediateSizeX,
        int intermediateSizeY,
        const Sam3Context& samContext
) const {
    return PersistentImageInput(
        inputImageSizeX,
        inputImageSizeY,
        intermediateSizeX,
        intermediateSizeY,
        samContext.getDeviceId()
    );
}
