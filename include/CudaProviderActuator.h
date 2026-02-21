#pragma once

#include "ProviderActuator.h"
#include <onnxruntime_cxx_api.h>

class CudaProviderActuator : public ProviderActuator {
    void registerProvider(const Ort::SessionOptions& sessionOptions) const {

    };
};
