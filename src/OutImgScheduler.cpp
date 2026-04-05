#include "OutImgScheduler.h"

#include <memory>
#include <stdexcept>

OutImgScheduler::OutImgScheduler(int imageX, int imageY, int instances, int devId) : deviceId(devId) {
    for (int i = 0; i < instances; ++i) {
        imageInstances.push_back(std::make_shared<ManagedGpuImage>(imageX, imageY, devId));
    }
}

std::shared_ptr<ManagedGpuImage> OutImgScheduler::getNextFreeImage() {
    for (auto& managedGpuImage : imageInstances) {
        if (managedGpuImage->isBeingUsed()) {
            continue;
        }

        // Claim the image on the caller's behalf. It is then the caller
        // who is responsible for freeing this image up again
        managedGpuImage->claim();
        return managedGpuImage;
    }

    // None were free, throw
    throw std::runtime_error("No available ManagedGpuImages to use");
}
