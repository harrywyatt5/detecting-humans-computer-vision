#pragma once

#include "GpuImage.h"
#include <memory>

class ManagedGpuImage {
private:
    bool inUse;
    std::shared_ptr<GpuImage> image;
public:
    ManagedGpuImage(int x, int y, int devId);

    void releaseClaim();
    void claim();
    bool isBeingUsed() const;
    std::shared_ptr<const GpuImage> getGpuImage() const;
    std::shared_ptr<GpuImage> getGpuImage();
};
