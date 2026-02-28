#pragma once

#include <string>

class Sam3ModelState {
public:
    enum class State {
        NONE,
        MOUNTED_LANGUAGE_TOKEN,
        READY
    };

    Sam3ModelState() : state(State::NONE) {}
    void processMountLanguage();
    void processWarmup();
    void processDetect();
    std::string toString();
private:
    void nextState();
    State state;
};
