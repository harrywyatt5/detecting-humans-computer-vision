#pragma once

#include "AbstractSession.h"
#include <stdexcept>

template<typename T>
class StartFlowSession : public AbstractSession {
protected:
    bool isInitialised;
    StartFlowSession<T>() : isInitialised(false) {}
    void thowIfNotInitialised() const {
        if (!isInitialised) {
            throw std::runtime_error("Session has not been initialised fully");
        }
    }
public:
    virtual void initialiseSession(T& initialiser) = 0;
    virtual ~StartFlowSession<T>() = default;
};
