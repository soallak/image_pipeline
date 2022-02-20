cmake_minimum_required(VERSION 3.5)
project(image_proc)

# ROS2 Flags
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# LTTng flame chart and graphs
# see https://archive.eclipse.org/tracecompass/doc/stable/org.eclipse.tracecompass.doc.user/LTTng-UST-Analyses.html#Flame_Chart_View
set(CMAKE_CXX_FLAGS "-g -O2 -finstrument-functions")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(ament_vitis)
find_package(vitis_common REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED) 
# find_package(XRT REQUIRED)  # needs to be reviewed, issues with sysroot isolation
if(OpenCV_VERSION VERSION_LESS "3.2.0")
  message(FATAL "Minimum OpenCV version is 3.2.0 (found version ${OpenCV_VERSION})")
endif()

# image_proc library
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/${PROJECT_NAME}/processor.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

# rectify library
ament_auto_add_library(rectify SHARED
  src/rectify.cpp)
target_compile_definitions(rectify
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
rclcpp_components_register_nodes(rectify "image_proc::RectifyNode")
set(node_plugins "${node_plugins}image_proc::RectifyNode;$<TARGET_FILE:rectify>\n")

# debayer library
ament_auto_add_library(debayer SHARED
  src/debayer.cpp
  src/edge_aware.cpp
)
target_compile_definitions(debayer
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
rclcpp_components_register_nodes(debayer "image_proc::DebayerNode")
set(node_plugins "${node_plugins}image_proc::DebayerNode;$<TARGET_FILE:debayer>\n")

# resize library
ament_auto_add_library(resize SHARED
  src/resize.cpp
)
target_compile_definitions(resize
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
rclcpp_components_register_nodes(resize "image_proc::ResizeNode")
set(node_plugins "${node_plugins}image_proc::ResizeNode;$<TARGET_FILE:resize>\n")

# resize_fpga component (with OpenCL)
ament_auto_add_library(resize_fpga SHARED src/resize_fpga.cpp)
target_include_directories(resize_fpga PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include)
target_link_libraries(resize_fpga ${OpenCL_LIBRARY})
target_compile_definitions(resize_fpga PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(resize_fpga "image_proc::ResizeNodeFPGA")
set(node_plugins "${node_plugins}image_proc::ResizeNodeFPGA;$<TARGET_FILE:resize_fpga>\n")

# rectify_fpga component (with OpenCL)
ament_auto_add_library(rectify_fpga SHARED src/rectify_fpga.cpp)
target_include_directories(rectify_fpga PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include)
target_link_libraries(rectify_fpga ${OpenCL_LIBRARY})
target_compile_definitions(rectify_fpga PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(rectify_fpga "image_proc::RectifyNodeFPGA")
set(node_plugins "${node_plugins}image_proc::RectifyNodeFPGA;$<TARGET_FILE:rectify_fpga>\n")

##########
## Integrated components
#
# Integrated components include functionality into a single ROS Component
##########

# rectify_resize_fpga component (integrated, with OpenCL) - 1 Component, 1 kernel
ament_auto_add_library(rectify_resize_fpga SHARED src/rectify_resize_fpga_integrated.cpp)
target_include_directories(rectify_resize_fpga PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
)
target_link_libraries(rectify_resize_fpga ${OpenCL_LIBRARY})
target_compile_definitions(rectify_resize_fpga PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(rectify_resize_fpga "image_proc::RectifyResizeNodeFPGA")
set(node_plugins "${node_plugins}image_proc::RectifyResizeNodeFPGA;$<TARGET_FILE:rectify_resize_fpga>\n")

# rectify_resize_fpga_xrt component (integrated, with XRT) - 1 Component, 1 kernel
ament_auto_add_library(rectify_resize_fpga_xrt SHARED src/rectify_resize_fpga_integrated_xrt.cpp)
if (DEFINED CMAKE_SYSROOT)
  target_link_options(rectify_resize_fpga_xrt PRIVATE "LINKER:-lxrt_coreutil")
endif()
target_include_directories(rectify_resize_fpga_xrt PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
  /home/xilinx/XRT/build/Debug/opt/xilinx/xrt/include  # TODO: replace with ROS 2 header-only library
  /usr/include/xrt  # TODO: replace with ROS 2 header-only library
)
target_link_libraries(rectify_resize_fpga_xrt ${OpenCL_LIBRARY})
target_compile_definitions(rectify_resize_fpga_xrt PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(rectify_resize_fpga_xrt "image_proc::RectifyResizeNodeFPGAXRT")
set(node_plugins "${node_plugins}image_proc::RectifyResizeNodeFPGAXRT;$<TARGET_FILE:rectify_resize_fpga_xrt>\n")

##########
## Streamlined components
#
# Streamlined components use separated kernels (2) and a stream interface between
# them to pass the data from one to the other. Two variants are presented, first
# an integrated one wherein both (rectify and resize) are implemented within the
# same node, i.e. integrated, e.g. `rectify_resize_fpga_streamlined`. The second 
# variant implements a function per Node, e.g. `rectify_fpga_streamlined` that 
# needs to be composed at launch time. 
#
# Additional combinations with and without XRT (_xrt) are also included.
##########

# rectify_resize_fpga_streamlined component - 1 Component, 2 kernels
ament_auto_add_library(rectify_resize_fpga_streamlined SHARED src/rectify_resize_fpga_streamlined.cpp)
target_include_directories(rectify_resize_fpga_streamlined PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
)
target_link_libraries(rectify_resize_fpga_streamlined ${OpenCL_LIBRARY})
target_compile_definitions(rectify_resize_fpga_streamlined PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(rectify_resize_fpga_streamlined "image_proc::RectifyResizeNodeFPGAStreamlined")
set(node_plugins "${node_plugins}image_proc::RectifyResizeNodeFPGAStreamlined;$<TARGET_FILE:rectify_resize_fpga_streamlined>\n")

# rectify_resize_fpga_streamlined_xrt component - 1 Component, 2 kernels
ament_auto_add_library(rectify_resize_fpga_streamlined_xrt SHARED src/rectify_resize_fpga_streamlined_xrt.cpp)
# TODO: figure out how to make proper use of XRT CMake integration instead
#   i.e. use "find_package(XRT REQUIRED)"
if (DEFINED CMAKE_SYSROOT)
  target_link_options(rectify_resize_fpga_streamlined_xrt PRIVATE "LINKER:-lxrt_coreutil")
endif()
target_include_directories(rectify_resize_fpga_streamlined_xrt PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
  /home/xilinx/XRT/build/Debug/opt/xilinx/xrt/include  # TODO: replace with ROS 2 header-only library
  /usr/include/xrt  # TODO: replace with ROS 2 header-only library
)
target_link_libraries(rectify_resize_fpga_streamlined_xrt ${OpenCL_LIBRARY})
target_compile_definitions(rectify_resize_fpga_streamlined_xrt PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(rectify_resize_fpga_streamlined_xrt "image_proc::RectifyResizeNodeFPGAStreamlinedXRT")
set(node_plugins "${node_plugins}image_proc::RectifyResizeNodeFPGAStreamlinedXRT;$<TARGET_FILE:rectify_resize_fpga_streamlined_xrt>\n")

# rectify_fpga_streamlined component - 1 Kernel, 1 Component (part of group)
ament_auto_add_library(rectify_fpga_streamlined SHARED src/rectify_fpga_streamlined.cpp)
target_include_directories(rectify_fpga_streamlined PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
)
target_link_libraries(rectify_fpga_streamlined ${OpenCL_LIBRARY})
target_compile_definitions(rectify_fpga_streamlined PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(rectify_fpga_streamlined "image_proc::RectifyNodeFPGAStreamlined")
set(node_plugins "${node_plugins}image_proc::RectifyNodeFPGAStreamlined;$<TARGET_FILE:rectify_fpga_streamlined>\n")

# resize_fpga_streamlined component - 1 Kernel, 1 Component (part of group)
ament_auto_add_library(resize_fpga_streamlined SHARED src/resize_fpga_streamlined.cpp)
target_include_directories(resize_fpga_streamlined PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
)
target_link_libraries(resize_fpga_streamlined ${OpenCL_LIBRARY})
target_compile_definitions(resize_fpga_streamlined PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(resize_fpga_streamlined "image_proc::ResizeNodeFPGAStreamlined")
set(node_plugins "${node_plugins}image_proc::ResizeNodeFPGAStreamlined;$<TARGET_FILE:resize_fpga_streamlined>\n")

# rectify_fpga_streamlined_xrt component - 1 Kernel, 1 Component (part of group)
ament_auto_add_library(rectify_fpga_streamlined_xrt SHARED src/rectify_fpga_streamlined_xrt.cpp)
# TODO: figure out how to make proper use of XRT CMake integration instead
#   i.e. use "find_package(XRT REQUIRED)"
if (DEFINED CMAKE_SYSROOT)
  target_link_options(rectify_fpga_streamlined_xrt PRIVATE "LINKER:-lxrt_coreutil")
endif()
target_include_directories(rectify_fpga_streamlined_xrt PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
  /home/xilinx/XRT/build/Debug/opt/xilinx/xrt/include  # TODO: replace with ROS 2 header-only library
  /usr/include/xrt  # TODO: replace with ROS 2 header-only library
)
target_link_libraries(rectify_fpga_streamlined_xrt ${OpenCL_LIBRARY})
target_compile_definitions(rectify_fpga_streamlined_xrt PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(rectify_fpga_streamlined_xrt "image_proc::RectifyNodeFPGAStreamlinedXRT")
set(node_plugins "${node_plugins}image_proc::RectifyNodeFPGAStreamlinedXRT;$<TARGET_FILE:rectify_fpga_streamlined_xrt>\n")

# resize_fpga_streamlined_xrt component - 1 Kernel, 1 Component (part of group)
ament_auto_add_library(resize_fpga_streamlined_xrt SHARED src/resize_fpga_streamlined_xrt.cpp)
# TODO: figure out how to make proper use of XRT CMake integration instead
#   i.e. use "find_package(XRT REQUIRED)"
if (DEFINED CMAKE_SYSROOT)
  target_link_options(resize_fpga_streamlined_xrt PRIVATE "LINKER:-lxrt_coreutil")
endif()
target_include_directories(resize_fpga_streamlined_xrt PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include
  /home/xilinx/XRT/build/Debug/opt/xilinx/xrt/include  # TODO: replace with ROS 2 header-only library
  /usr/include/xrt  # TODO: replace with ROS 2 header-only library
)
target_link_libraries(resize_fpga_streamlined_xrt ${OpenCL_LIBRARY})
target_compile_definitions(resize_fpga_streamlined_xrt PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(resize_fpga_streamlined_xrt "image_proc::ResizeNodeFPGAStreamlinedXRT")
set(node_plugins "${node_plugins}image_proc::ResizeNodeFPGAStreamlinedXRT;$<TARGET_FILE:resize_fpga_streamlined_xrt>\n")

# # individual kernels, each kernel its own xclbin
# #
# if(ROS_ACCELERATION)
#   # resize_accel kernel
#   vitis_acceleration_kernel(
#     NAME resize_accel
#     FILE src/image_proc/xf_resize_accel.cpp
#     CONFIG cfg/kv260.cfg
#     INCLUDE
#       include/image_proc
#       ${CMAKE_INSTALL_PREFIX}/include
#     TYPE hw
#     # CLOCK 100000000:resize_accel
#     LINK
#     # PACKAGE
#   )

#   # rectify_accel kernel
#   vitis_acceleration_kernel(
#     NAME rectify_accel
#     FILE src/image_proc/xf_rectify_accel.cpp
#     CONFIG cfg/kv260.cfg
#     INCLUDE
#       include/image_proc
#       ${CMAKE_INSTALL_PREFIX}/include
#     TYPE hw
#     CLOCK 100000000:rectify_accel
#     LINK
#     # PACKAGE
#   )
# endif()

# image_proc kernel, all kernels smashed into a single xclbin
#
if(ROS_ACCELERATION)
  
  #####################
  # Synth
  #####################

  # # resize_accel kernel
  # vitis_acceleration_kernel(
  #   NAME resize_accel
  #   FILE src/image_proc/xf_resize_accel.cpp
  #   CONFIG cfg/kv260.cfg
  #   INCLUDE
  #     include/image_proc
  #     ${CMAKE_INSTALL_PREFIX}/include
  #   TYPE hw
  # )

  # # rectify_accel kernel
  # vitis_acceleration_kernel(
  #   NAME rectify_accel
  #   FILE src/image_proc/xf_rectify_accel.cpp
  #   CONFIG cfg/kv260.cfg
  #   INCLUDE
  #     include/image_proc
  #     ${CMAKE_INSTALL_PREFIX}/include
  #   TYPE hw
  # )

  # rectify_accel_streamlined kernel
  vitis_acceleration_kernel(
    NAME rectify_accel_streamlined   
    FILE src/image_proc/xf_rectify_accel_streamlined.cpp
    CONFIG cfg/kv260_image_proc_streamlined.cfg
    INCLUDE
      include/image_proc
      ${CMAKE_INSTALL_PREFIX}/include
    TYPE hw 
  )

  # resize_accel_streamlined kernel
  vitis_acceleration_kernel(
    NAME resize_accel_streamlined
    FILE src/image_proc/xf_resize_accel_streamlined.cpp
    CONFIG cfg/kv260_image_proc_streamlined.cfg
    INCLUDE 
      include/image_proc 
      ${CMAKE_INSTALL_PREFIX}/include
    TYPE hw
  )

  # # rectify_resize_accel kernel
  # vitis_acceleration_kernel(
  #   NAME rectify_resize_accel
  #   FILE src/image_proc/xf_rectify_resize_accel.cpp
  #   CONFIG cfg/kv260_image_proc_integrated.cfg
  #   INCLUDE
  #     include/image_proc
  #     ${CMAKE_INSTALL_PREFIX}/include
  #   TYPE hw 
  # )  

  #####################
  # Place and route
  #####################

  # # image_proc kernel
  # vitis_link_kernel(
  #   OUTPUT image_proc
  #   KERNELS resize_accel rectify_accel
  #   CONFIG cfg/kv260_image_proc.cfg
  # )

  # image_proc_streamlined kernel
  vitis_link_kernel(
    OUTPUT image_proc_streamlined
    KERNELS resize_accel_streamlined rectify_accel_streamlined
    CONFIG cfg/kv260_image_proc_streamlined.cfg 
  )

  # # image_proc_integrated kernel
  # vitis_link_kernel(
  #   OUTPUT image_proc_integrated  
  #   KERNELS rectify_resize_accel
  #   CONFIG cfg/kv260_image_proc_integrated.cfg
  # )

endif()

# crop_decimate library
ament_auto_add_library(crop_decimate SHARED
  src/crop_decimate.cpp 
)
target_compile_definitions(crop_decimate
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
rclcpp_components_register_nodes(crop_decimate "image_proc::CropDecimateNode")
set(node_plugins "${node_plugins}image_proc::CropDecimateNode;$<TARGET_FILE:crop_decimate>\n")

# crop_non_zero library  
ament_auto_add_library(crop_non_zero SHARED
  src/crop_non_zero.cpp
)
target_compile_definitions(crop_non_zero
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
rclcpp_components_register_nodes(crop_non_zero "image_proc::CropNonZeroNode")
set(node_plugins "${node_plugins}image_proc::CropNonZeroNode;$<TARGET_FILE:crop_non_zero>\n")

# image_proc example node
ament_auto_add_executable(image_proc_exe
  src/image_proc.cpp
)
target_link_libraries(image_proc_exe
  debayer
  rectify
  ament_index_cpp::ament_index_cpp
)
set_target_properties(image_proc_exe PROPERTIES OUTPUT_NAME image_proc)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(image_proc "stdc++fs")
endif()

install(
  TARGETS
    image_proc_exe
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
