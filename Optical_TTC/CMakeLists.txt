cmake_minimum_required( VERSION 2.8 )
project( Optical_TTC )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )         # 添加 C++ 11 标准支持

find_package( OpenCV )                          # 寻找 OpenCV 库
include_directories( ${OpenCV_INCLUDE_DIRS} )   # 添加头文件

add_executable( useLK useLK.cpp )
target_link_libraries( useLK ${OpenCV_LIBS} )   # 添加动态链接库
