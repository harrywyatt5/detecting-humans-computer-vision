#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>

class AbstractSession {
public:
    virtual void run() = 0;
    virtual std::vector<Ort::Value> runWithResult() = 0;

    virtual ~AbstractSession() = default;
};