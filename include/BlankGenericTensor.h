#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>

class BlankGenericTensor {
private:
    std::vector<int64_t> tensorShape;
    Ort::Value tensor{nullptr};
protected:
    BlankGenericTensor(std::vector<int64_t> tShape, Ort::Value t) : tensorShape(std::move(tShape)), tensor(std::move(t)) {};
public:
    virtual size_t getSize() const = 0;
    virtual size_t getSizeInBytes() const = 0;
    const std::vector<int64_t>& getTensorShape() {
        return tensorShape;
    }
    const Ort::Value& getTensor() const {
        return tensor;
    }

    // Delete copy
    BlankGenericTensor(const BlankGenericTensor&) = delete;
    BlankGenericTensor& operator=(const BlankGenericTensor&) = delete;
    virtual ~BlankGenericTensor() = default;
};
