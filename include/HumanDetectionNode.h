#pragma once

#include "PersistentSam3Model.h"
#include "LanguageToken.h"
#include <rclcpp/rclcpp.hpp>
#include <memory>

class HumanDetectionNode : public rclcpp::Node {
private:
    std::unique_ptr<PersistentSam3Model> samModel;
    std::unique_ptr<PersistentImageInput> imageInput;
    LanguageToken promptToken;
public:
    HumanDetectionNode();
};
