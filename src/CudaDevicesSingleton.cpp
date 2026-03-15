#include "CudaDevicesSingleton.h"

#include "CudaDevice.h"
#include <memory>

std::shared_ptr<CudaDevicesSingleton> CudaDevicesSingleton::singleton = nullptr;

std::shared_ptr<CudaDevicesSingleton> CudaDevicesSingleton::getInstance() {
    if (CudaDevicesSingleton::singleton == nullptr) {
        CudaDevicesSingleton::singleton = std::shared_ptr<CudaDevicesSingleton>(new CudaDevicesSingleton());
    }

    return singleton;
}

std::shared_ptr<CudaDevice> CudaDevicesSingleton::getForId(int id) {
    for (auto& entry : devices) {
        if (entry->getDeviceId() == id) {
            return entry;
        }
    }

    auto newDevice = std::make_shared<CudaDevice>(id);
    devices.push_back(newDevice);

    return newDevice;
}
