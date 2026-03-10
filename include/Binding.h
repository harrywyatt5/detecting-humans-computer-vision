#pragma once

#include "BlankGenericTensor.h"
#include <string>
#include <memory>

class Binding {
public:
    enum class BindingType {
        INPUT,
        OUTPUT
    };

    const std::string& getBindingName() const;
    std::shared_ptr<const BlankGenericTensor> getTensor() const;
    BindingType getBindingType() const;
    std::string getBindingTypeAsString() const;
private:
    std::string name;
    std::shared_ptr<BlankGenericTensor> tensor;
    BindingType type;
public:
    Binding(const std::string& bindName, std::shared_ptr<BlankGenericTensor> t, BindingType bindType) 
        : name(bindName), tensor(t), type(bindType) {}
};
