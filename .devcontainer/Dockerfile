FROM osrf/ros:noetic-desktop-focal AS base

SHELL ["/bin/bash", "-c"]

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
ENV NVIDIA_DRIVER_CAPABILITIES all

# Create the user
ARG USERNAME=tacps
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN <<-EOF
	groupadd --gid $USER_GID $USERNAME
	useradd --shell /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME
	apt-get update
	apt-get install -y sudo
	echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
	chmod 0440 /etc/sudoers.d/$USERNAME
	rm -rf /var/lib/apt/lists/*
EOF
# Set the default user
USER $USERNAME

# Setup ROS env and catkin workspace on every Bash
RUN <<EOF
printf '\n# Set up ROS env\n' >> $HOME/.bashrc
echo '. /opt/ros/$ROS_DISTRO/setup.bash' >> $HOME/.bashrc
mkdir --parents $HOME/catkin_ws/src
printf '\n# Set up catkin workspace\n' >> $HOME/.bashrc
echo '[ -f $HOME/catkin_ws/devel/setup.bash ] && . $HOME/catkin_ws/devel/setup.bash' >> $HOME/.bashrc
EOF

RUN <<EOF
sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
	curl wget git \
	python-is-python3 python3-pip \
	ros-$ROS_DISTRO-gazebo-ros-pkgs ros-$ROS_DISTRO-gazebo-ros-control \
	ros-$ROS_DISTRO-turtlebot3 \
	bash-completion
sudo rm -rf /var/lib/apt/lists/*
python3 -m pip install --upgrade pip
EOF

# Workaround: https://answers.gazebosim.org/question/8577/gazebo-22-cant-open-usrsharegazebosetupsh/?answer=21515#post-id-21515
# Fix not merged: https://github.com/ros-simulation/gazebo_ros_pkgs/pull/1100/commits/7d0ed4ceeea0c77a606ca59c539aa8d5d7e56dc8
RUN sudo sed -i '/setup_path=/ s|$(pkg-config --variable=prefix gazebo)|/usr|' /opt/ros/noetic/lib/gazebo_ros/{gazebo,gzserver,gzclient}
