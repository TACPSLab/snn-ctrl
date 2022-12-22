FROM osrf/ros:humble-simulation-jammy AS base

SHELL ["/bin/bash", "-c"]

ARG USERNAME=tacps
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --shell /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Clean up
    && rm -rf /var/lib/apt/lists/*
# Set the default user
USER $USERNAME

# Underlay
RUN echo '[ -f /opt/ros/$ROS_DISTRO/setup.bash ] && source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
# Overlay
RUN echo '[ -f ~/ws/install/local_setup.bash ] && source ~/ws/install/local_setup.bash' >> ~/.bashrc

# Enable colcon_cd
RUN echo '[ -f /usr/share/colcon_cd/function/colcon_cd.sh ] && source /usr/share/colcon_cd/function/colcon_cd.sh' >> ~/.bashrc && \
    echo 'export _colcon_cd_root=/opt/ros/humble/' >> ~/.bashrc
# Enable argcomplete of colon
RUN echo '[ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ] && source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash' >> ~/.bashrc

RUN sudo apt-get update && sudo apt-get install --yes --no-install-recommends \
        python3-pip \
        # https://launchpad.net/ubuntu/jammy/+package/python-is-python3
        python-is-python3 \
    && sudo rm -rf /var/lib/apt/lists/*

# Workaround: https://github.com/colcon/colcon-core/issues/454
ENV PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources

# Upgrade pre-installed attrs otherwise Pylance cannot resolve importation of it
RUN python3 -m pip install --upgrade attrs

# Make workspace folder
RUN mkdir -p ~/ws/src


FROM base AS devel

RUN python3 -m pip install mypy

