#pragma once

#include <vector>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

template <typename T>
class GenericTensor {
protected:
    T* start;
    // Inteface note: size should hold the number of elements allocated
    // rather than the raw number of bytes, to be in line with objects like
    // std::vector
    size_t size;
    std::vector<int64_t> tensorShape;
    Ort::Value tensor{nullptr};

    GenericTensor(T* ptr, size_t size, std::vector<int64_t> tensorShape, Ort::Value tensor) 
            : start(ptr), size(size), tensorShape(std::move(tensorShape)), tensor(std::move(tensor)) {}
    virtual void releaseMemory() = 0;
public:
    T* getStartPtr() {
        return start;
    }
    size_t getSize() const {
        return size;
    }
    const std::vector<int64_t>& getTensorShape() const {
        return tensorShape;
    }
    const Ort::Value& getTensor() const {
        return tensor;
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
