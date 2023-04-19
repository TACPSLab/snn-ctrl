import atexit
from time import sleep

import torch
from attrs import define
from rospy import ServiceProxy, Publisher, init_node, wait_for_service, wait_for_message
from roslaunch import configure_logging
from roslaunch.parent import ROSLaunchParent
from roslaunch.rlutil import get_or_generate_uuid
from rospkg import RosPack
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from sensor_msgs.msg import LaserScan, Imu
from std_srvs.srv import Empty
from typing import Any, Dict, Optional, Tuple
from torch import Tensor


@define
class Env:
    """
    Assume Gazebo service connections are stable.

    FIXME Shutting down gracefully
        Tried __del__ but does not work.

    FIXME https://github.com/ros-simulation/gazebo_ros_pkgs/issues/864

    FIXME Know if a model has been spawned before setting its state
        https://answers.ros.org/question/266043/know-if-a-model-exists-in-my-gazebo-simulation/

    FIXME Allow robot and goal penetrate into each other
        https://answers.gazebosim.org//question/20191/gazebo-not-updating-visual-position-in-gzclient/

    TODO Timing
        - Gazebo Classic step sim
        - ROS clock?
    """

    _device: torch.device

    _uuid: Any = get_or_generate_uuid(None, False)
    _reset_simulation: ServiceProxy = ServiceProxy("/gazebo/reset_simulation", Empty)
    _get_model_state: ServiceProxy = ServiceProxy("/gazebo/get_model_state", GetModelState)
    _set_model_state: ServiceProxy = ServiceProxy("/gazebo/set_model_state", SetModelState)
    _cmd_vel: Publisher = Publisher("/cmd_vel", Twist, queue_size=1)

    _last_distance: float = 0.0

    def __attrs_post_init__(self) -> None:
        configure_logging(self._uuid)  # Scripts using roslaunch MUST call configure_logging
        launch = ROSLaunchParent(
            self._uuid,
            [RosPack().get_path("mapless_navi") + "/launch/main.launch"],
        )
        launch.start()
        atexit.register(launch.shutdown)

        init_node("env")
        wait_for_service("/gazebo/reset_simulation")
        wait_for_service("/gazebo/get_model_state")
        wait_for_service("/gazebo/set_model_state")

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        self._cmd_vel.publish(
            linear  = Vector3(*action[0:3]),
            angular = Vector3(*action[3:6]),
        )

        # FIXME: https://github.com/ros-simulation/gazebo_ros_pkgs/issues/1268
        sleep(0.05)

        obs = self._get_obs()
        reward = torch.zeros(1, device=self._device, dtype=torch.float32)
        terminated = torch.tensor([False], device=self._device, dtype=torch.bool)
        truncated = torch.tensor([False], device=self._device, dtype=torch.bool)

        scan = obs[:36]
        separation = obs[36:39]  # <x, y, z> starts at the robot frame origin, terminates at the goal frame origin
        distance = separation.norm(p="fro")
        relative_bearing = torch.arccos(separation[0] / distance)  # Angle between the vector and x-axis

        # Collide
        if scan.min() < 0.2:
            print("Collide")
            terminated = torch.tensor([True], device=self._device, dtype=torch.bool)
            reward -= 100

        if distance < 0.3:
            print("Goal!")
            terminated = torch.tensor([True], device=self._device, dtype=torch.bool)
            reward += 100
            # TODO: Reward if stop at the goal. Punish if rush past it.

        reward += 30 * (self._last_distance - distance)  # TODO: Expose the factor
        self._last_distance = distance

        # TODO: Truncate if standstill

        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Tensor, Dict[str, Any]]:
        self._reset_simulation()

        # TODO: Random geometry points
        self._set_model_state(ModelState(
            model_name="goal",
            reference_frame="world",
            pose=Pose(
                position=Point(1, 1, 0),
                orientation=Quaternion(0, 0, 0, 0),
            ),
            twist=Twist(
                Vector3(0, 0, 0),
                Vector3(0, 0, 0),
            ),
        ))

        return self._get_obs(), {}

    def _get_obs(self) -> Tensor:
        scan: LaserScan = wait_for_message("/scan", LaserScan)
        imu: Imu = wait_for_message("/imu", Imu)
        separation: ModelState = self._get_model_state(
            model_name="goal",
            relative_entity_name="turtlebot3_waffle_pi",
        )

        obs = torch.tensor((
            *scan.ranges[::10],  # Reduce samples size from 360 to 36
            separation.pose.position.x,
            separation.pose.position.y,
            separation.pose.position.z,
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w,
        ), dtype=torch.float32, device=self._device)

        return torch.nan_to_num(obs, posinf=3.5)
