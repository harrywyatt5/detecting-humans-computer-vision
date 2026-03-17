#pragma once

#ifdef __CUDACC__
    #define CUDA_HOST_DEV __host__ __device__
#else
    #define CUDA_HOST_DEV
#endif

#include <cstdint>

class MappedMask {
private:
    bool present;
    uint8_t remappedTarget;
public:
    CUDA_HOST_DEV MappedMask() : present(false), remappedTarget(0) {}
    CUDA_HOST_DEV MappedMask(bool isPresent, uint8_t target) : present(isPresent), remappedTarget(target) {}

    CUDA_HOST_DEV bool isPresent() const {
        return present;
    }
    CUDA_HOST_DEV uint8_t getRemappedTarget() const {
        return remappedTarget;
    }
    CUDA_HOST_DEV void setRemappedTarget(uint8_t newTarget) {
        remappedTarget = newTarget;
    }
};
