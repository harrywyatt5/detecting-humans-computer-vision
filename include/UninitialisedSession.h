#pragma once

#include "AbstractSession.h"
#include <stdexcept>

class UninitialisedSession : public AbstractSession {
protected:
    bool isInitialised;
    UninitialisedSession() : isInitialised(false) {}
    void throwIfNotInitialised() const {
        if (!isInitialised) {
            throw std::runtime_error("Session has not been initialised fully");
        }
    }
public:
    virtual void setSessionAsInitialised() {
        isInitialised = true;
    }
    virtual ~UninitialisedSession() = default;
};
