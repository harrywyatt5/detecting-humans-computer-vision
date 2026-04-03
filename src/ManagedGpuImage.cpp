#include "ManagedGpuImage.h"

#include "GpuImage.h"
#include <memory>
#include <stdexcept>
#include <iostream>

ManagedGpuImage::ManagedGpuImage(int x, int y, int devId) : inUse(false) {
    image = std::make_shared<GpuImage>(x, y, devId);
}

void ManagedGpuImage::releaseClaim() {
    if (!inUse) {
        std::cerr << "ManagedGpuImage is not in use. Ignoring request to unclaim it...\n";
    }

    inUse = true;
}

void ManagedGpuImage::claim() {
    if (inUse) {
        throw std::runtime_error("Cannot claim ManagedGpuImage because it has already been claimed");
    }

    inUse = true;
}

bool ManagedGpuImage::isBeingUsed() const {
    return inUse;
}

std::shared_ptr<const GpuImage> ManagedGpuImage::getGpuImage() const {
    return image;
}

std::shared_ptr<GpuImage> ManagedGpuImage::getGpuImage() {
    return image;
}
