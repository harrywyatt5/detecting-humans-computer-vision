#pragma once

#include "CudaDevice.h"
#include <memory>
#include <vector>

class CudaDevicesSingleton {
private:
    std::vector<std::shared_ptr<CudaDevice>> devices;

    CudaDevicesSingleton() {}

    static std::shared_ptr<CudaDevicesSingleton> singleton;
public:
    std::shared_ptr<CudaDevice> getForId(int id);

    static std::shared_ptr<CudaDevicesSingleton> getInstance();
};
