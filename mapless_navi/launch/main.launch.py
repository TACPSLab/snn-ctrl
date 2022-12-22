from os import pathsep
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, ThisLaunchFileDir
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

pkg_share_dir = get_package_share_directory('mapless_navi')

def generate_launch_description():

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [ThisLaunchFileDir(), '/robot_state_publisher.launch.py']),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    gz_resource_path = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', [  # TODO: Use the line below for Gazebo >= 7
    # set_gz_resource_path = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', [
        str(Path(pkg_share_dir)/'models'), pathsep +
        str(Path(pkg_share_dir)/'worlds'),
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments = {'gz_args': '-r maze.sdf'}.items(),
    )

    gz_spawn_tb4 = Node(
        package    = 'ros_gz_sim',
        executable = 'create',
        output     = 'screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'turtlebot4',
            '-x', '0',
            '-y', '0',
            '-z', '0.01'],
    )

    return LaunchDescription([
        # robot_state_publisher,
        gz_resource_path,
        gz_sim,
        # gz_spawn_tb4,
    ])
