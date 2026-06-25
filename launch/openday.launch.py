import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. RealSense Camera Launch Inclusion
    realsense_launch_dir = os.path.join(get_package_share_directory('realsense2_camera'), 'launch')
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(realsense_launch_dir, 'rs_launch.py')),
        launch_arguments={
            'camera_name': 'left_eye_cam',
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'enable_color': 'true',
            'rgb_camera.profile': '1920x1080x30',
        }.items()
    )

    # 2. Real Time Humans Node
    real_time_humans_node = Node(
        package='real_time_humans',
        executable='real_time_humans',
        name='real_time_humans',
        output='screen',
        parameters=[{
            'use_sim_time': False, 
            'sam3_text_encoder_path': '/workspace/ros_packages/detecting-humans-computer-vision/sam3-onnx/text-encoder-static.onnx',
            'sam3_vision_encoder_path': '/workspace/ros_packages/detecting-humans-computer-vision/sam3-onnx/vision-encoder-static.onnx',
            'sam3_decoder_path': '/workspace/ros_packages/detecting-humans-computer-vision/sam3-onnx/decoder-static.onnx'
        }]
    )

    # 3. Detecting Groups Node
    detecting_groups_node = Node(
        package='detecting_groups',
        executable='detecting_groups',
        name='detecting_groups',
        output='screen',
        parameters=[{
            'gemma_model_location': '/workspace/ros_packages/detecting-groups/gemma-model/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf',
            'image_encoder_location': '/workspace/ros_packages/detecting-groups/gemma-model/mmproj-F16.gguf',
            'processing_period': 1.0
        }]
    )

    # Build and return the launch description
    return LaunchDescription([
        realsense_node,
        real_time_humans_node,
        detecting_groups_node
    ])
