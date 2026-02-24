#pragma once

#include "AbstractSession.h"
#include "Sam3Context.h"

class AbstractSessionFactory {
public:
    virtual AbstractSession createSession(const Sam3Context& context) const;
};
