#include "Binding.h"

#include "BlankGenericTensor.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

const std::string& Binding::getBindingName() const {
    return name;
}

std::shared_ptr<const BlankGenericTensor> Binding::getTensor() const {
    return tensor;
}

Binding::BindingType Binding::getBindingType() const {
    return type;
}

std::string Binding::getBindingTypeAsString() const {
    switch (type) {
        case Binding::BindingType::INPUT: 
            return "input";
        case Binding::BindingType::OUTPUT:
            return "output";
        default:
            return "";
    }
}
