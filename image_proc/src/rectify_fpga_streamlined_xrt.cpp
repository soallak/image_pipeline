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

#include "image_proc/rectify_fpga_streamlined_xrt.hpp"
#include "tracetools_image_pipeline/tracetools.h"

namespace image_geometry
{

// From pinhole_camera_model.cpp
enum DistortionState { NONE, CALIBRATED, UNKNOWN };

struct PinholeCameraModelFPGAStreamlinedXRT::Cache
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

PinholeCameraModelFPGAStreamlinedXRT::PinholeCameraModelFPGAStreamlinedXRT()
{
  // XRT approach
  // create kernel
  device = xrt::device(0);  // Open the device, default to 0
  // TODO: generalize this using launch extra_args for composable Nodes
  // see https://github.com/ros2/launch_ros/blob/master/launch_ros/launch_ros/descriptions/composable_node.py#L45
  //
  // use "extra_arguments" from ComposableNode and propagate the value to this class constructor
  uuid = device.load_xclbin("/lib/firmware/xilinx/image_proc_streamlined/image_proc_streamlined.xclbin");
  krnl_rectify = xrt::kernel(device, uuid, "rectify_accel_streamlined");
}

void PinholeCameraModelFPGAStreamlinedXRT::rectifyImageFPGA(const cv::Mat& raw, cv::Mat& rectified, bool gray) const
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
        //  capabilities to differentiate between CV_32F and CV_64F. It'd be
        //  nice to consult with the appropriate team specialized on this.
        //
        int channels = gray ? 1 : 3;
        size_t image_in_size_count = raw.rows * raw.cols * channels;
        size_t image_in_size_bytes = image_in_size_count * sizeof(unsigned char);
        // size_t image_in_size_bytes = raw.rows * raw.cols
        //                               * sizeof(unsigned char)
        //                               * channels;

        size_t map_in_size_count = raw.rows * raw.cols;
        size_t map_in_size_bytes = map_in_size_count * sizeof(float);
        // size_t map_in_size_bytes = raw.rows * raw.cols * sizeof(float);
        // size_t image_out_size_bytes = image_in_size_bytes;
        cl_int err;

        // Allocate the buffers in global memory
        auto buffer_inImage = xrt::bo(device,
                                      image_in_size_bytes,
                                      krnl_rectify.group_id(0));
        auto buffer_inMapX = xrt::bo(device,
                                     map_in_size_bytes,
                                     krnl_rectify.group_id(1));
        auto buffer_inMapY = xrt::bo(device,
                                     map_in_size_bytes,
                                     krnl_rectify.group_id(2));
        // auto buffer_outImage = xrt::bo(device,
        //                              image_out_size_bytes,
        //                              krnl_rectify.group_id(3));

        // Map the contents of the buffer object into host memory
        auto buffer_inImage_map = buffer_inImage.map<unsigned char*>();
        auto buffer_inMapX_map = buffer_inMapX.map<float*>();
        auto buffer_inMapY_map = buffer_inMapY.map<float*>();
        // auto buffer_outImage_map = buffer_outImage.map<int*>();

        // Copy data
        // std::fill(buffer_inImage_map, buffer_inImage_map + image_in_size_count, 0);  // NOLINT
        // std::fill(buffer_inMapX_map, buffer_inMapX_map + map_in_size_count, 0);  // NOLINT
        // std::fill(buffer_inMapY_map, buffer_inMapY_map + map_in_size_count, 0);  // NOLINT
        // std::fill(buffer_outImage_map, buffer_outImage_map + image_out_size_count, 0);  // NOLINT
        std::memcpy(buffer_inImage_map, raw.data, image_in_size_bytes);  // NOLINT
        std::memcpy(buffer_inMapX_map, map_x.data, map_in_size_bytes);  // NOLINT
        std::memcpy(buffer_inMapY_map, map_y.data, map_in_size_bytes);  // NOLINT        

        // Synchronize buffers with device side
        buffer_inImage.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        buffer_inMapX.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        buffer_inMapY.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // buffer_outImage.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Set kernel arguments, run and wait for it to finish
        auto run = xrt::run(krnl_rectify);
        run.set_arg(0, buffer_inImage);
        run.set_arg(1, buffer_inMapX);
        run.set_arg(2, buffer_inMapY);
        // run.set_arg(3, buffer_outImage);
        run.set_arg(4, raw.rows);
        run.set_arg(5, raw.cols);
        run.start();

        run.wait();
        break;
      }
    default:
      assert(cache_->distortion_state == UNKNOWN);
      throw Exception("Cannot call rectifyImage when distortion is unknown.");
  }
}

} //namespace image_geometry

namespace image_proc
{

RectifyNodeFPGAStreamlinedXRT::RectifyNodeFPGAStreamlinedXRT(const rclcpp::NodeOptions & options)
: Node("RectifyNodeFPGAStreamlinedXRT", options)
{
  queue_size_ = this->declare_parameter("queue_size", 5);
  interpolation = this->declare_parameter("interpolation", 1);
  pub_rect_ = image_transport::create_publisher(this, "image_rect");
  subscribeToCamera();
}

// Handles (un)subscribing when clients (un)subscribe
void RectifyNodeFPGAStreamlinedXRT::subscribeToCamera()
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
      &RectifyNodeFPGAStreamlinedXRT::imageCb,
      this, std::placeholders::_1, std::placeholders::_2), "raw");
  // }
}

void RectifyNodeFPGAStreamlinedXRT::imageCb(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg)
{

  std::cout << "RectifyNodeFPGAStreamlinedXRT::imageCb XRT" << std::endl;
  TRACEPOINT(
    image_proc_rectify_cb_init,
    static_cast<const void *>(this),
    static_cast<const void *>(&(*image_msg)),
    static_cast<const void *>(&(*info_msg)));

  if (pub_rect_.getNumSubscribers() < 1) {
    TRACEPOINT(
      image_proc_rectify_cb_fini,
      static_cast<const void *>(this),
      static_cast<const void *>(&(*image_msg)),
      static_cast<const void *>(&(*info_msg)));
    return;
  }

  // Verify camera is actually calibrated
  if (info_msg->k[0] == 0.0) {
    RCLCPP_ERROR(
      this->get_logger(), "Rectified topic '%s' requested but camera publishing '%s' "
      "is uncalibrated", pub_rect_.getTopic().c_str(), sub_camera_.getInfoTopic().c_str());
    TRACEPOINT(
      image_proc_rectify_cb_fini,
      static_cast<const void *>(this),
      static_cast<const void *>(&(*image_msg)),
      static_cast<const void *>(&(*info_msg)));
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
    TRACEPOINT(
      image_proc_rectify_cb_fini,
      static_cast<const void *>(this),
      static_cast<const void *>(&(*image_msg)),
      static_cast<const void *>(&(*info_msg)));
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
  TRACEPOINT(
    image_proc_rectify_init,
    static_cast<const void *>(this),
    static_cast<const void *>(&(*image_msg)),
    static_cast<const void *>(&(*info_msg)));
  model_.rectifyImageFPGA(image, rect, gray);  // FPGA computation
  TRACEPOINT(
    image_proc_rectify_fini,
    static_cast<const void *>(this),
    static_cast<const void *>(&(*image_msg)),
    static_cast<const void *>(&(*info_msg)));

  // // Allocate new rectified image message
  // sensor_msgs::msg::Image::SharedPtr rect_msg =
  //   cv_bridge::CvImage(image_msg->header, image_msg->encoding, rect).toImageMsg();
  // pub_rect_.publish(rect_msg);

  TRACEPOINT(
    image_proc_rectify_cb_fini,
    static_cast<const void *>(this),
    static_cast<const void *>(&(*image_msg)),
    static_cast<const void *>(&(*info_msg)));
}

}  // namespace image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(image_proc::RectifyNodeFPGAStreamlinedXRT)
