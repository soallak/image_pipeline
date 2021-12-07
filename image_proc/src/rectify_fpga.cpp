/*
      ____  ____
     /   /\/   /
    /___/  \  /   Copyright (c) 2021, Xilinx®.
    \   \   \/    Author: Víctor Mayoral Vilches <victorma@xilinx.com>
     \   \
     /   /        Licensed under the Apache License, Version 2.0 (the "License");
    /___/   /\    you may not use this file except in compliance with the License.
    \   \  /  \   You may obtain a copy of the License at
     \___\/\___\            http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Inspired by rectify.cpp authored by Willow Garage, Inc., Andreas Klintberg,
      Joshua Whitley
*/

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.hpp>
#include <vitis_common/common/xf_headers.hpp>
#include <vitis_common/common/utilities.hpp>

#include <thread>
#include <memory>
#include <vector>

#include "image_proc/rectify_fpga.hpp"

namespace image_proc
{

RectifyNodeFPGA::RectifyNodeFPGA(const rclcpp::NodeOptions & options)
: Node("RectifyNodeFPGA", options)
{
  queue_size_ = this->declare_parameter("queue_size", 5);
  interpolation = this->declare_parameter("interpolation", 1);
  pub_rect_ = image_transport::create_publisher(this, "image_rect");
  subscribeToCamera();

  // Load the acceleration kernel
  // NOTE: hardcoded path according to dfx-mgrd conventions
  // TODO: generalize this
  cl_int err;
  unsigned fileBufSize;

  std::vector<cl::Device> devices = get_xilinx_devices();  // Get the device:
  cl::Device device = devices[0];
  OCL_CHECK(err, context_ = new cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, queue_ = new cl::CommandQueue(*context_, device, CL_QUEUE_PROFILING_ENABLE, &err));
  OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
  std::cout << "INFO: Device found - " << device_name << std::endl;
  char* fileBuf = read_binary_file("/lib/firmware/xilinx/image_proc/rectify_accel.xclbin", fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  devices.resize(1);
  OCL_CHECK(err, cl::Program program(*context_, devices, bins, NULL, &err));
  OCL_CHECK(err, krnl_ = new cl::Kernel(program, "rectify_accel", &err));
}

// Handles (un)subscribing when clients (un)subscribe
void RectifyNodeFPGA::subscribeToCamera()
{
  std::lock_guard<std::mutex> lock(connect_mutex_);

  /*
  *  SubscriberStatusCallback not yet implemented
  *
  if (pub_rect_.getNumSubscribers() == 0)
    sub_camera_.shutdown();
  else if (!sub_camera_)
  {
  */
  sub_camera_ = image_transport::create_camera_subscription(
    this, "image", std::bind(
      &RectifyNodeFPGA::imageCb,
      this, std::placeholders::_1, std::placeholders::_2), "raw");
  // }
}

void RectifyNodeFPGA::imageCb(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg)
{
  RCLCPP_INFO(this->get_logger(), "RectifyNodeFPGA::imageCb.");
  RCLCPP_INFO(this->get_logger(), "RectifyNodeFPGA::imageCb 2.");


  // if (pub_rect_.getNumSubscribers() < 1) {
  //   return;
  // }
  //
  // // Verify camera is actually calibrated
  // if (info_msg->k[0] == 0.0) {
  //   RCLCPP_ERROR(
  //     this->get_logger(), "Rectified topic '%s' requested but camera publishing '%s' "
  //     "is uncalibrated", pub_rect_.getTopic().c_str(), sub_camera_.getInfoTopic().c_str());
  //   return;
  // }
  //
  // // If zero distortion, just pass the message along
  // bool zero_distortion = true;
  //
  // for (size_t i = 0; i < info_msg->d.size(); ++i) {
  //   if (info_msg->d[i] != 0.0) {
  //     zero_distortion = false;
  //     break;
  //   }
  // }
  //
  // // This will be true if D is empty/zero sized
  // if (zero_distortion) {
  //   pub_rect_.publish(image_msg);
  //   return;
  // }

  bool gray =
    (sensor_msgs::image_encodings::numChannels(image_msg->encoding) == 1);

  // // TODO: removethis if not needed
  // // Update the camera model
  // model_.fromCameraInfo(info_msg);

  // Create cv::Mat views onto both buffers
  const cv::Mat src = cv_bridge::toCvShare(image_msg)->image;
  cv::Mat rect;
  cv::Mat ocv_remapped, hls_remapped, diff;

  // Allocate memory for the outputs:
  // TODO, do this in the initialization
  std::cout << "INFO: Allocate memory for input and output data." << std::endl;
  ocv_remapped.create(src.rows, src.cols, src.type()); // opencv result
  hls_remapped.create(src.rows, src.cols, src.type()); // create memory for output images
  diff.create(src.rows, src.cols, src.type());

  // Flip the image
  // TODO, removeme ///////////////
  cv::Mat map_x, map_y;
  // Allocate memory
  map_x.create(src.rows, src.cols, CV_32FC1);          // Mapx for opencv remap function
  map_y.create(src.rows, src.cols, CV_32FC1);          // Mapy for opencv remap function

  for (int i = 0; i < src.rows; i++) {
         for (int j = 0; j < src.cols; j++) {
             float valx = (float)(src.cols - j - 1), valy = (float)i;
             map_x.at<float>(i, j) = valx;
             map_y.at<float>(i, j) = valy;
         }
  }
  // TODO, removeme///////////////

  ////////////////////////////////
  // CPU

  // initRectificationMaps and remap
  // model_.rectifyImage(image, rect, interpolation);

  // remap
  // example map generation, flips the image horizontally
  // cv::remap(src, ocv_remapped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::remap(src, ocv_remapped, map_x, map_y, interpolation);
  ////////////////////////////////


  ////////////////////////////////
  // FPGA

  // initRectificationMaps
  // TODO


  // remap
  // OpenCL section:
  int channels = gray ? 1 : 3;
  size_t image_in_size_bytes = src.rows * src.cols * sizeof(unsigned char) * channels;
  size_t map_in_size_bytes = src.rows * src.cols * sizeof(float);
  size_t image_out_size_bytes = image_in_size_bytes;

  cl_int err;

  // Allocate the buffers:
  OCL_CHECK(err, cl::Buffer buffer_inImage(*context_, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_inMapX(*context_, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_inMapY(*context_, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_outImage(*context_, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

  // Set kernel arguments:
  int rows = src.rows;
  int cols = src.cols;
  OCL_CHECK(err, err = krnl_->setArg(0, buffer_inImage));
  OCL_CHECK(err, err = krnl_->setArg(1, buffer_inMapX));
  OCL_CHECK(err, err = krnl_->setArg(2, buffer_inMapY));
  OCL_CHECK(err, err = krnl_->setArg(3, buffer_outImage));
  OCL_CHECK(err, err = krnl_->setArg(4, rows));
  OCL_CHECK(err, err = krnl_->setArg(5, cols));

  // Initialize the buffers:
  cl::Event event;

  OCL_CHECK(err,
            queue_->enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                     CL_TRUE,             // blocking call
                                     0,                   // buffer offset in bytes
                                     image_in_size_bytes, // Size in bytes
                                     src.data,            // Pointer to the data to copy
                                     nullptr, &event));

  OCL_CHECK(err,
            queue_->enqueueWriteBuffer(buffer_inMapX,     // buffer on the FPGA
                                     CL_TRUE,           // blocking call
                                     0,                 // buffer offset in bytes
                                     map_in_size_bytes, // Size in bytes
                                     map_x.data,        // Pointer to the data to copy
                                     nullptr, &event));

  OCL_CHECK(err,
            queue_->enqueueWriteBuffer(buffer_inMapY,     // buffer on the FPGA
                                     CL_TRUE,           // blocking call
                                     0,                 // buffer offset in bytes
                                     map_in_size_bytes, // Size in bytes
                                     map_y.data,        // Pointer to the data to copy
                                     nullptr, &event));

  // Execute the kernel:
  OCL_CHECK(err, err = queue_->enqueueTask(*krnl_));

  // Copy Result from Device Global Memory to Host Local Memory
  queue_->enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                          CL_TRUE,         // blocking call
                          0,               // offset
                          image_out_size_bytes,
                          hls_remapped.data, // Data will be stored here
                          nullptr, &event);

  // Clean up:
  queue_->finish();
  ////////////////////////////////

  // Results verification:
  cv::absdiff(ocv_remapped, hls_remapped, diff);
  // Find minimum and maximum differences.
  float err_per;
  xf::cv::analyzeDiff(diff, 0, err_per);

  if (err_per > 0.0f) {
      fprintf(stderr, "ERROR: Test Failed.\n ");
      RCLCPP_ERROR(this->get_logger(), "Test failed.");
      return;
  } else {
    RCLCPP_INFO(this->get_logger(), "Test successful.");
  }

  // Allocate new rectified image message
  sensor_msgs::msg::Image::SharedPtr rect_msg =
    cv_bridge::CvImage(image_msg->header, image_msg->encoding, hls_remapped).toImageMsg();
  pub_rect_.publish(rect_msg);
}

}  // namespace image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(image_proc::RectifyNodeFPGA)
