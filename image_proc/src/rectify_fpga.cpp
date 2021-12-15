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
      Joshua Whitley. Inspired also by PinholeCameraModel class.
*/

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.hpp>
#include <vitis_common/common/xf_headers.hpp>
#include <vitis_common/common/utilities.hpp>

#include <thread>
#include <memory>
#include <vector>
#include <chrono>

#include "image_proc/rectify_fpga.hpp"

namespace image_geometry
{

// From pinhole_camera_model.cpp
enum DistortionState { NONE, CALIBRATED, UNKNOWN };

struct PinholeCameraModelFPGA::Cache
{
  DistortionState distortion_state;

  cv::Mat_<double> K_binned, P_binned; // Binning applied, but not cropping

  mutable bool full_maps_dirty;
  mutable cv::Mat full_map1, full_map2;

  mutable bool reduced_maps_dirty;
  mutable cv::Mat reduced_map1, reduced_map2;

  mutable bool rectified_roi_dirty;
  mutable cv::Rect rectified_roi;

  Cache()
    : full_maps_dirty(true),
      reduced_maps_dirty(true),
      rectified_roi_dirty(true)
  {
  }
};

PinholeCameraModelFPGA::PinholeCameraModelFPGA()
{
  // Load the acceleration kernel
  // NOTE: hardcoded path according to dfx-mgrd conventions
  // TODO: generalize this in the future
  cl_int err;
  unsigned fileBufSize;

  std::vector<cl::Device> devices = get_xilinx_devices();  // Get the device:
  cl::Device device = devices[0];
  OCL_CHECK(err, context_ = new cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, queue_ = new cl::CommandQueue(*context_, device, CL_QUEUE_PROFILING_ENABLE, &err));
  OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
  std::cout << "INFO: Device found - " << device_name << std::endl;
  // TODO: Reconsider this file hardcoded in here
  char* fileBuf = read_binary_file("/lib/firmware/xilinx/image_proc/rectify_accel.xclbin", fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  devices.resize(1);
  OCL_CHECK(err, cl::Program program(*context_, devices, bins, NULL, &err));
  OCL_CHECK(err, krnl_ = new cl::Kernel(program, "rectify_accel", &err));
}

void PinholeCameraModelFPGA::rectifyImageFPGA(const cv::Mat& raw, cv::Mat& rectified, bool gray, int interpolation) const
{
  assert( initialized() );

  switch (cache_->distortion_state) {
    case NONE:
      raw.copyTo(rectified);
      break;
    case CALIBRATED:
      {
        initRectificationMaps();

        cv::Mat hls_remapped;
        cv::Mat map_x, map_y;
        std::vector<cv::Mat> channels_map;

        hls_remapped.create(raw.rows, raw.cols, raw.type()); // create memory for output images
        // if (cache_->reduced_map1.channels() == 2) {
        if (cache_->reduced_map1.type() == CV_32FC2 || cache_->reduced_map1.type() == CV_16SC2) {
          cv::convertMaps(cache_->reduced_map1, cache_->reduced_map2, map_x, map_y, CV_32FC1);
        } else if(cache_->reduced_map1.channels() == 1 && cache_->reduced_map2.channels() == 1) {
          map_x = cache_->reduced_map1;
          map_y = cache_->reduced_map2;
        } else {
          throw Exception("Unexpected amount of channels in warp maps: " + cache_->reduced_map1.channels());
        }

        //  REVIEW: Vitis Vision Library reviewed does not seem to offer
        //  capabilities to differentiate between CV_32F and CV_64F. It'd be nice
        //  to consult with the appropriate team specialized on this.
        //
        int channels = gray ? 1 : 3;
        size_t image_in_size_bytes = raw.rows * raw.cols * sizeof(unsigned char) * channels;
        size_t map_in_size_bytes = raw.rows * raw.cols * sizeof(float);
        size_t image_out_size_bytes = image_in_size_bytes;
        cl_int err;

        // Allocate the buffers:
        OCL_CHECK(err, cl::Buffer buffer_inImage(*context_, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_inMapX(*context_, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_inMapY(*context_, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_outImage(*context_, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

        // Set kernel arguments:
        int rows = raw.rows;
        int cols = raw.cols;
        OCL_CHECK(err, err = krnl_->setArg(0, buffer_inImage));
        OCL_CHECK(err, err = krnl_->setArg(1, buffer_inMapX));
        OCL_CHECK(err, err = krnl_->setArg(2, buffer_inMapY));
        OCL_CHECK(err, err = krnl_->setArg(3, buffer_outImage));
        OCL_CHECK(err, err = krnl_->setArg(4, rows));
        OCL_CHECK(err, err = krnl_->setArg(5, cols));

        // Initialize the buffers:
        cl::Event event;

        OCL_CHECK(err,
                  queue_->enqueueWriteBuffer(buffer_inImage,    // buffer on the FPGA
                                           CL_TRUE,             // blocking call
                                           0,                   // buffer offset in bytes
                                           image_in_size_bytes, // Size in bytes
                                           raw.data,            // Pointer to the data to copy
                                           nullptr, &event));

        OCL_CHECK(err,
                  queue_->enqueueWriteBuffer(buffer_inMapX,   // buffer on the FPGA
                                           CL_TRUE,           // blocking call
                                           0,                 // buffer offset in bytes
                                           map_in_size_bytes, // Size in bytes
                                           map_x.data,        // Pointer to the data to copy
                                           nullptr, &event));

        OCL_CHECK(err,
                  queue_->enqueueWriteBuffer(buffer_inMapY,   // buffer on the FPGA
                                           CL_TRUE,           // blocking call
                                           0,                 // buffer offset in bytes
                                           map_in_size_bytes, // Size in bytes
                                           map_y.data,        // Pointer to the data to copy
                                           nullptr, &event));

        // Execute the kernel:
        OCL_CHECK(err, err = queue_->enqueueTask(*krnl_));

        // Copy Result from Device Global Memory to Host Local Memory
        queue_->enqueueReadBuffer(buffer_outImage,    // This buffers data will be read
                                CL_TRUE,              // blocking call
                                0,                    // offset
                                image_out_size_bytes,
                                hls_remapped.data,    // Data will be stored here
                                nullptr, &event);

        // Clean up:
        queue_->finish();
        rectified = hls_remapped;
        break;
      }
    default:
      assert(cache_->distortion_state == UNKNOWN);
      throw Exception("Cannot call rectifyImage when distortion is unknown.");
  }
}

void PinholeCameraModelFPGA::rectifyImageFPGA_debug(const cv::Mat& raw, cv::Mat& rectified, bool gray, int interpolation) const
{
  assert( initialized() );

  cv::Mat ocv_remapped, hls_remapped, diff;
  ocv_remapped.create(raw.rows, raw.cols, raw.type()); // opencv result
  hls_remapped.create(raw.rows, raw.cols, raw.type()); // create memory for output images
  diff.create(raw.rows, raw.cols, raw.type());

  initRectificationMaps();

  // Extract map_x and map_y if smashed into cache_->reduced_map1 as (x,y).
  // (see OpenCV remap's documentation to
  // understand the rationale https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4)
  cv::Mat map_x, map_y;
  std::vector<cv::Mat> channels_map;
  if (cache_->reduced_map1.type() == CV_32FC2 || cache_->reduced_map1.type() == CV_16SC2) {
    cv::convertMaps(cache_->reduced_map1, cache_->reduced_map2, map_x, map_y, CV_32FC1);
  } else if(cache_->reduced_map1.channels() == 1 && cache_->reduced_map2.channels() == 1) {
    map_x = cache_->reduced_map1;
    map_y = cache_->reduced_map2;
  } else {
    throw Exception("Unexpected amount of channels in warp maps: " + cache_->reduced_map1.channels());
  }

  // Debug
  std::cout << "cache_->reduced_map1.row(0): " << cache_->reduced_map1.row(0) << std::endl;
  std::cout << "cache_->reduced_map2.row(0): " << cache_->reduced_map2.row(0) << std::endl;
  std::cout << "map_x.row(0): " << map_x.row(0) << std::endl;
  std::cout << "map_y.row(0): " << map_y.row(0) << std::endl;

  //  CV_16SC2 - 11
  //  CV_32FC1 - 5
  //  CV_32FC2 - 13
  //  CV_16UC1 - 2
  std::cout << "CV_32FC2: " << CV_32FC2 << std::endl;
  std::cout << "CV_16SC2: " << CV_16SC2 << std::endl;
  std::cout << "CV_32FC1: " << CV_32FC1 << std::endl;
  std::cout << "CV_16UC1: " << CV_16UC1 << std::endl;
  std::cout << "cache_->reduced_map1.type(): " << cache_->reduced_map1.type() << std::endl;
  std::cout << "cache_->reduced_map2.type(): " << cache_->reduced_map2.type() << std::endl;
  std::cout << "map_x.type(): " << map_x.type() << std::endl;
  std::cout << "map_y.type(): " << map_y.type() << std::endl;

  //  REVIEW: Vitis Vision Library reviewed does not seem to offer
  //  capabilities to differentiate between CV_32F and CV_64F. It'd be nice
  //  to consult with the appropriate team specialized on this.
  //
  //
  // CPU implementation
  auto start_cpu = std::chrono::high_resolution_clock::now();
  if (raw.depth() == CV_32F || raw.depth() == CV_64F)
  {
    std::cout << "raw.depth() == CV_32F || CV_64F\n";
    cv::remap(raw, ocv_remapped, map_x, map_y, interpolation, cv::BORDER_CONSTANT, std::numeric_limits<float>::quiet_NaN());
  }
  else {
    cv::remap(raw, ocv_remapped, map_x, map_y, interpolation);
  }
  auto finish_cpu = std::chrono::high_resolution_clock::now();

  // FPGA implementation
  auto start_fpga = std::chrono::high_resolution_clock::now();
  int channels = gray ? 1 : 3;
  size_t image_in_size_bytes = raw.rows * raw.cols * sizeof(unsigned char) * channels;
  size_t map_in_size_bytes = raw.rows * raw.cols * sizeof(float);
  size_t image_out_size_bytes = image_in_size_bytes;
  cl_int err;

  // Allocate the buffers:
  OCL_CHECK(err, cl::Buffer buffer_inImage(*context_, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_inMapX(*context_, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_inMapY(*context_, CL_MEM_READ_ONLY, map_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_outImage(*context_, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

  // Set kernel arguments:
  int rows = raw.rows;
  int cols = raw.cols;
  OCL_CHECK(err, err = krnl_->setArg(0, buffer_inImage));
  OCL_CHECK(err, err = krnl_->setArg(1, buffer_inMapX));
  OCL_CHECK(err, err = krnl_->setArg(2, buffer_inMapY));
  OCL_CHECK(err, err = krnl_->setArg(3, buffer_outImage));
  OCL_CHECK(err, err = krnl_->setArg(4, rows));
  OCL_CHECK(err, err = krnl_->setArg(5, cols));

  // Initialize the buffers:
  cl::Event event;

  OCL_CHECK(err,
            queue_->enqueueWriteBuffer(buffer_inImage,    // buffer on the FPGA
                                     CL_TRUE,             // blocking call
                                     0,                   // buffer offset in bytes
                                     image_in_size_bytes, // Size in bytes
                                     raw.data,            // Pointer to the data to copy
                                     nullptr, &event));

  OCL_CHECK(err,
            queue_->enqueueWriteBuffer(buffer_inMapX,   // buffer on the FPGA
                                     CL_TRUE,           // blocking call
                                     0,                 // buffer offset in bytes
                                     map_in_size_bytes, // Size in bytes
                                     // cache_->reduced_map1.data, // Pointer to the data to copy
                                     map_x.data, // Pointer to the data to copy
                                     nullptr, &event));

  OCL_CHECK(err,
            queue_->enqueueWriteBuffer(buffer_inMapY,   // buffer on the FPGA
                                     CL_TRUE,           // blocking call
                                     0,                 // buffer offset in bytes
                                     map_in_size_bytes, // Size in bytes
                                     // cache_->reduced_map2.data, // Pointer to the data to copy
                                     map_y.data, // Pointer to the data to copy
                                     nullptr, &event));

  // Execute the kernel:
  OCL_CHECK(err, err = queue_->enqueueTask(*krnl_));

  // Copy Result from Device Global Memory to Host Local Memory
  queue_->enqueueReadBuffer(buffer_outImage,    // This buffers data will be read
                          CL_TRUE,              // blocking call
                          0,                    // offset
                          image_out_size_bytes,
                          hls_remapped.data,    // Data will be stored here
                          nullptr, &event);

  // Clean up:
  queue_->finish();
  auto finish_fpga = std::chrono::high_resolution_clock::now();

  // Assign rectified to either output of CPU or FPGA
  // rectified = ocv_remapped;  // CPU
  rectified = hls_remapped;  // FPGA

  // Time evaluation
  std::chrono::duration<double> elapsed_cpu = finish_cpu - start_cpu;
  std::chrono::duration<double> elapsed_fpga = finish_fpga - start_fpga;
  std::cout << "Elapsed time CPU: " << elapsed_cpu.count() << " s\n";
  std::cout << "Elapsed time FPGA: " << elapsed_fpga.count() << " s\n";
  std::cout << "CPU/FPGA: " << elapsed_cpu.count()/elapsed_fpga.count() << "\n";

  // Results verification:
  cv::absdiff(ocv_remapped, hls_remapped, diff);
  float err_per;
  xf::cv::analyzeDiff(diff, 0, err_per);

  if (err_per > 0.0f) {
      std::cout << "ERROR: Test Failed.\n";
      return;
  } else {
    std::cout << "Test successful.\n";
  }
}

} //namespace image_geometry

namespace image_proc
{

RectifyNodeFPGA::RectifyNodeFPGA(const rclcpp::NodeOptions & options)
: Node("RectifyNodeFPGA", options)
{
  queue_size_ = this->declare_parameter("queue_size", 5);
  interpolation = this->declare_parameter("interpolation", 1);
  pub_rect_ = image_transport::create_publisher(this, "image_rect");
  subscribeToCamera();
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

  if (pub_rect_.getNumSubscribers() < 1) {
    return;
  }

  // Verify camera is actually calibrated
  if (info_msg->k[0] == 0.0) {
    RCLCPP_ERROR(
      this->get_logger(), "Rectified topic '%s' requested but camera publishing '%s' "
      "is uncalibrated", pub_rect_.getTopic().c_str(), sub_camera_.getInfoTopic().c_str());
    return;
  }

  // If zero distortion, just pass the message along
  bool zero_distortion = true;

  for (size_t i = 0; i < info_msg->d.size(); ++i) {
    if (info_msg->d[i] != 0.0) {
      zero_distortion = false;
      break;
    }
  }

  // This will be true if D is empty/zero sized
  if (zero_distortion) {
    pub_rect_.publish(image_msg);
    return;
  }

  bool gray =
    (sensor_msgs::image_encodings::numChannels(image_msg->encoding) == 1);

  // Update the camera model
  model_.fromCameraInfo(info_msg);

  // Create cv::Mat views onto both buffers
  const cv::Mat image = cv_bridge::toCvShare(image_msg)->image;
  cv::Mat rect;

  // Rectify
  // model_.rectifyImage(image, rect, interpolation);  // CPU computation
  // model_.rectifyImageFPGA(image, rect, gray, interpolation);  // FPGA computation
  model_.rectifyImageFPGA_debug(image, rect, gray, interpolation);  // CPU and FPGA debug computations

  // Allocate new rectified image message
  sensor_msgs::msg::Image::SharedPtr rect_msg =
    cv_bridge::CvImage(image_msg->header, image_msg->encoding, rect).toImageMsg();
  pub_rect_.publish(rect_msg);
}

}  // namespace image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(image_proc::RectifyNodeFPGA)
