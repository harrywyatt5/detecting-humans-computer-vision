#pragma once

#include <onnxruntime_cxx_api.h>

// Interface
class ProviderActuator {
public:
    virtual void registerProvider(const Ort::SessionOptions& sessionOptions) const = 0;
};