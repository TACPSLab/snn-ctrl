cmake_minimum_required(VERSION 3.0.2)
project(mapless_navi)

find_package(catkin REQUIRED)

catkin_package()

catkin_python_setup()

install(DIRECTORY
  launch
  models
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
