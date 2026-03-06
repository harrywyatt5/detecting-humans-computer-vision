#pragma once

template<typename T>
class SessionInitialiser {
public:
    virtual void initialiseSession(T& session) = 0;

    virtual ~SessionInitialiser() = default;
};
