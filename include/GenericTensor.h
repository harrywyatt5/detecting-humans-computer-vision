#pragma once

#include "BlankGenericTensor.h"
#include <vector>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

template <typename T>
class GenericTensor : public BlankGenericTensor {
protected:
    T* start;
    // Inteface note: size should hold the number of elements allocated
    // rather than the raw number of bytes, to be in line with objects like
    // std::vector
    size_t size;
    std::vector<int64_t> tensorShape;
    Ort::Value tensor{nullptr};

    GenericTensor(T* ptr, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor) 
            : start(ptr), size(size), BlankGenericTensor(std::move(tensorShape), std::move(tensor)) {}
    virtual void releaseMemory() = 0;

    static size_t getTensorCountFromShape(const std::vector<int64_t>& shape) {
        size_t count = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            count *= shape[i];
        }

        return count;
    }
public:
    virtual void copyToBuffer(const std::vector<T>& sourceBuffer) = 0;
    virtual std::vector<T> readBuffer() const = 0;
    T* getStartPtr() {
        return start;
    }
    const T* getConstStartPtr() const {
        return start;
    }
    size_t getSize() const override {
        return size;
    }
    size_t getSizeInBytes() const override {
        return size * sizeof(T);
    }

    // Delete copies
    GenericTensor(const GenericTensor&) = delete;
    GenericTensor& operator=(const GenericTensor&) = delete;
    // Allow moves
    GenericTensor(GenericTensor&& other) noexcept 
        : start(other.start), size(other.size), tensorShape(std::move(other.tensorShape)), tensor(std::move(other.tensor)) {
            other.start = nullptr;
            other.size = 0;
            other.tensor = Ort::Value{nullptr};
    }
    GenericTensor& operator=(GenericTensor&& other) noexcept {
        // This escapes if we are just moving this object to itself 
        if (this == &other) {
            return *this;
        }

        this->releaseMemory();
        this->start = other.start;
        this->size = other.size;
        this->tensorShape = std::move(other.tensorShape);
        this->tensor = std::move(other.tensor);

        other.start = nullptr;
        other.size = 0;
        other.tensor = Ort::Value{nullptr};
        return *this;
    }
    virtual ~GenericTensor() = default;
};
