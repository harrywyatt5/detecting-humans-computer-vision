#include "BindingBlueprint.h"

#include "Binding.h"
#include <onnxruntime_cxx_api.h>

void BindingBlueprint::AddBinding(Binding binding) {
    bindings.push_back(binding);
}

Ort::IoBinding BindingBlueprint::createIoBindingObject(Ort::Session& session) const {
    Ort::IoBinding ioBinding{session};

    for (const auto& binding : bindings) {
        if (binding.getBindingType() == Binding::BindingType::INPUT) {
            ioBinding.BindInput(binding.getBindingName().c_str(), binding.getTensor()->getTensor());
        } else {
            ioBinding.BindOutput(binding.getBindingName().c_str(), binding.getTensor()->getTensor());
        }
    }

    return ioBinding;
}
