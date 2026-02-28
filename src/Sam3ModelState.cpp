#include "Sam3ModelState.h"

#include <string>

void Sam3ModelState::processMountLanguage() {
    
}

void Sam3ModelState::processWarmup() {

}

void Sam3ModelState::processDetect() {

}

std::string Sam3ModelState::toString() {
    switch (state) {
        case State::NONE:
            return "NONE";
        case State::MOUNTED_LANGUAGE_TOKEN:
            return "MOUNTED_LANGUAGE_TOKEN";
        case State::READY:
            return "READY";
        default:
            return "UNKNOWN";
    }
}
