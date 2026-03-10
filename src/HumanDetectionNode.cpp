#include "HumanDetectionNode.h"

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <string>

HumanDetectionNode::HumanDetectionNode() 
    : samModel(nullptr), Node("human_detection_node") {
    auto shareLocation = ament_index_cpp::get_package_share_directory("real_time_humans");

    this->declare_parameter<std::string>("sam3_engine_cache_dir", "~/.human_detection");
    this->declare_parameter<int>("max_cpu_threads", 1);
    this->declare_parameter<bool>("use_fp16", true);
    this->declare_parameter<int>("cuda_device", 0);
    this->declare_parameter<std::string>("log_level", "info");
    this->declare_parameter<std::string>("sam3_text_encoder_path", shareLocation + "/sam3-onnx/text-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_vision_encoder_path", shareLocation + "/sam3-onnx/vision-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_decoder_path", shareLocation + "/sam3-onnx/geo-encoder-mask-decoder-fp16.onnx");
    this->declare_parameter<std::string>("encoded_prompt_path", shareLocation + "language.token");
    this->declare_parameter<std::string>("camera_left_topic", "");
    this->declare_parameter<std::string>("camera_right_topic", "");

}