#pragma once

#include <onnxruntime_cxx_api.h>

class Sam3Context {
private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
public:
    Sam3Context(Ort::Env environment, Ort::SessionOptions sOptions) 
        : env(std::move(environment)), sessionOptions(std::move(sOptions)) {};
    
    const Ort::Env& getEnvironment() const {
        return env;
    };

    const Ort::SessionOptions& getSessionOptions() const {
        return sessionOptions;
    };
};
