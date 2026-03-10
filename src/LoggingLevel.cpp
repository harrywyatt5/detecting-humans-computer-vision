#include "LoggingLevel.h"

#include <onnxruntime_c_api.h>
#include <stdexcept>
#include <string>

std::string LoggingLevel::toString() const {
    switch (logLevel) {
        case LoggingLevel::Level::DEBUG:
            return "debug";
        case LoggingLevel::Level::INFO:
            return "info";
        case LoggingLevel::Level::WARNING:
            return "warning";
        case LoggingLevel::Level::ERROR:
            return "error";
        case LoggingLevel::Level::FATAL:
            return "fatal";
        default:
            throw std::runtime_error("LoggingLevel in invalid state");
    }
}

OrtLoggingLevel LoggingLevel::toOrtLoggingLevel() const {
    switch (logLevel) {
        case LoggingLevel::Level::DEBUG:
            return OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
        case LoggingLevel::Level::INFO:
            return OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
        case LoggingLevel::Level::WARNING:
            return OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
        case LoggingLevel::Level::ERROR:
            return OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
        case LoggingLevel::Level::FATAL:
            return OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL;
        default:
            throw std::runtime_error("LoggingLevel in invalid state"); 
    }
}

LoggingLevel LoggingLevel::fromString(const std::string& string, bool throwOnInvalid) {
    if (string == "debug") {
        return LoggingLevel::Level::DEBUG;
    } else if (string == "info") {
        return LoggingLevel::Level::INFO;
    } else if (string == "warning") {
        return LoggingLevel::Level::WARNING;
    } else if (string == "error") {
        return LoggingLevel::Level::ERROR;
    } else if (string == "fatal") {
        return LoggingLevel::Level::FATAL;
    } else {
        if (throwOnInvalid) {
            throw std::runtime_error(string + " is not a valid option for LoggingLevel::fromString");
        } else {
            return LoggingLevel::Level::WARNING;
        }
    }
}
