#pragma once 

#include <cstdint>

class MappedMask {
private:
    bool present;
    uint8_t remappedTarget;
public:
    MappedMask();
    MappedMask(bool isPresent, uint8_t target);

    bool isPresent() const;
    uint8_t getRemappedTarget() const;
    void setRemappedTarget(uint8_t newTarget);
};
