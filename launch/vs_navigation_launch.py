#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='phenobot'),
        DeclareLaunchArgument('tf_prefix', default_value=''),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        
        Node(
            package='visual_multi_crop_row_navigation',
            executable='vs_navigator.py',
            name='vs_navigation',
            output='screen',
            parameters=[PathJoinSubstitution([
                FindPackageShare('visual_multi_crop_row_navigation'),
                'configs',
                'params.yaml'
            ])]
        )
    ])