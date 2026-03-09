#pragma once

#include <onnxruntime_cxx_api.h>
#include "Binding.h"

class BindingBlueprint {
private:
    std::vector<Binding> bindings;
public:
    BindingBlueprint() {}
    void AddBinding(Binding binding);
    Ort::IoBinding createIoBindingObject(Ort::Session& session) const;
};
