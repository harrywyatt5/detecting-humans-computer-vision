#pragma once

#include "ManagedGpuImage.h"
#include <vector>
#include <memory>

class OutImgScheduler {
private:
    std::vector<std::shared_ptr<ManagedGpuImage>> imageInstances;
    int deviceId;
public:
    OutImgScheduler(int imageX, int imageY, int instances, int devId);
    std::shared_ptr<ManagedGpuImage> getNextFreeImage();
};
