#pragma once

#include <string>
#include <onnxruntime_c_api.h>

class LoggingLevel {
public:
    enum class Level {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    };
private:
    Level logLevel;
public:
    LoggingLevel(Level level) : logLevel(level) {}

    std::string toString() const;
    OrtLoggingLevel toOrtLoggingLevel() const;
    bool operator==(const LoggingLevel& rhs) const {
        return logLevel == rhs.logLevel;
    }
    bool operator!=(const LoggingLevel& rhs) const {
        return !(*this == rhs);
    }
    bool operator==(const std::string& rhs) const {
        return toString() == rhs;
    }
    bool operator!=(const std::string& rhs) const {
        return toString() != rhs;
    }

    static LoggingLevel fromString(const std::string& value, bool throwOnInvalid);

    static const LoggingLevel DEBUG;
    static const LoggingLevel INFO;
    static const LoggingLevel WARNING;
    static const LoggingLevel ERROR;
    static const LoggingLevel FATAL;
};
