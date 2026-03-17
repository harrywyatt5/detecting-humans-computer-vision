#include "MappedMask.h"

#include <cstdint>

MappedMask::MappedMask() : present(false), remappedTarget(0) {}
MappedMask::MappedMask(bool isPresent, uint8_t target) : present(isPresent), remappedTarget(target) {}

bool MappedMask::isPresent() const {
    return present;
}

uint8_t MappedMask::getRemappedTarget() const {
    return remappedTarget;
}

void MappedMask::setRemappedTarget(uint8_t newTarget) {
    remappedTarget = newTarget;
}
