#pragma once

#include <vector>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

template <typename T>
class GenericTensor {
protected:
    T* start;
    // Inteface note: size should hold the number of bytes allocated
    // rather than the number of instances
    size_t size;
    std::vector<int64_t> tensorShape;
    Ort::Value tensor{nullptr};
public:
    T* getStartPtr() {
        return start;
    };
    size_t getMemoryOccupancy() const {
        return size;
    };
    const std::vector<int64_t>& getTensorShape() const {
        return tensorShape
    };
    const Ort::Value& getTensor() const {
        return tensor;
    };

    virtual ~GenericTensor() = default;
};
