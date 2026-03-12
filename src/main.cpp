#include "HumanDetectionNode.h"
#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HumanDetectionNode>());
    rclcpp::shutdown();
    return 0;
}
