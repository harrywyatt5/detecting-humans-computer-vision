#pragma once

#include "AbstractSession.h"
#include "Sam3Context.h"
#include <memory>

class AbstractStartSessionFactory {
public:
    virtual std::unique_ptr<AbstractSession> createSession(const Sam3Context& context) const = 0;
};
