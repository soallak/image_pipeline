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
*/

// #include "xf_remap_config.h"

#include "hls_stream.h"

#include <vitis_common/common/xf_common.hpp>
#include <vitis_common/common/xf_utility.hpp>
#include <vitis_common/imgproc/xf_remap.hpp>
#include <vitis_common/imgproc/xf_resize.hpp>

#define XF_INTERPOLATION_TYPE XF_INTERPOLATION_BILINEAR  // interpolation type
#define XF_USE_URAM false  // Enable to map some structures to UltraRAM instead of BRAM

// #define PTR_IMG_WIDTH 8  // for GRAY
// #define TYPE XF_8UC1
// #define CHANNELS 1

#define PTR_IMG_WIDTH 32  // if RGB
#define TYPE XF_8UC3
#define CH_TYPE XF_RGB
#define CHANNELS 3

#define TYPE_XY XF_32FC1
#define PTR_MAP_WIDTH 32
#define NPC XF_NPPC1  // Number of pixels to be processed per cycle
// #define NPC XF_NPPC2
// #define NPC XF_NPPC4
#define NPC_RESIZE XF_NPPC8
#define XF_WIN_ROWS 50  // Number of input image rows to be buffered inside

#define HEIGHT 480  // 480 with default RealSense drivers
#define WIDTH 640  // 640 with default RealSense drivers
// #define WIDTH 1280
// #define HEIGHT 960
// #define WIDTH 960
// #define HEIGHT 720

#define XF_CAMERA_MATRIX_SIZE 9
#define XF_DIST_COEFF_SIZE 5

#define RGB 1
// port widths
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 256
// For Nearest Neighbor & Bilinear Interpolation, max down scale
// factor 2 for all 1-pixel modes, and for upscale in x direction
#define MAXDOWNSCALE 2
#define INTERPOLATION 1  // Bilinear Interpolation

/* Output image Dimensions */
#define NEWWIDTH 1280  // Maximum output image width
#define NEWHEIGHT 960 // Maximum output image height
// #define NEWWIDTH 2560  // Maximum output image width
// #define NEWHEIGHT 1920 // Maximum output image height
// #define NEWWIDTH 1920  // Maximum output image width
// #define NEWHEIGHT 1440 // Maximum output image height


extern "C" {

  void rectify_resize_accel(
    ap_uint<PTR_IMG_WIDTH>* img_in,
    const float* map_x,
    const float* map_y,
    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
    int rows,
    int cols,
    int rows_out,
    int cols_out)
{
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0  // NOLINT
    #pragma HLS INTERFACE m_axi      port=map_x         offset=slave  bundle=gmem1  // NOLINT
    #pragma HLS INTERFACE m_axi      port=map_y         offset=slave  bundle=gmem2  // NOLINT
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem3  // NOLINT
    #pragma HLS INTERFACE s_axilite  port=rows  // NOLINT
    #pragma HLS INTERFACE s_axilite  port=cols  // NOLINT
    #pragma HLS INTERFACE s_axilite  port=rows_out  // NOLINT  
    #pragma HLS INTERFACE s_axilite  port=cols_out  // NOLINT
    #pragma HLS INTERFACE s_axilite  port=return  // NOLINT

    // Get arguments for remap operation (remap)
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> imgInput(rows, cols);
    xf::cv::Mat<TYPE_XY, HEIGHT, WIDTH, NPC> mapX(rows, cols);
    xf::cv::Mat<TYPE_XY, HEIGHT, WIDTH, NPC> mapY(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> imgOutput_rectify(rows, cols);
    xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC> imgOutput_resize(rows_out, cols_out);  // NOLINT


    const int HEIGHT_WIDTH_LOOPCOUNT = HEIGHT * WIDTH / XF_NPIXPERCYCLE(NPC);
    for (unsigned int i = 0; i < rows * cols; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT_WIDTH_LOOPCOUNT  // NOLINT
        #pragma HLS PIPELINE II=1  // NOLINT
        // clang-format on
        float map_x_val = map_x[i];
        float map_y_val = map_y[i];
        mapX.write_float(i, map_x_val);
        mapY.write_float(i, map_y_val);
    }

    #pragma HLS STREAM variable=imgInput.data depth=2  // NOLINT
    #pragma HLS STREAM variable=mapX.data depth=2  // NOLINT
    #pragma HLS STREAM variable=mapY.data depth=2  // NOLINT
    #pragma HLS STREAM variable=imgOutput_rectify.data depth=2  // NOLINT
    #pragma HLS stream variable=imgOutput_resize.data depth=2  // NOLINT

    #pragma HLS DATAFLOW

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IMG_WIDTH, TYPE, HEIGHT, WIDTH, NPC>(img_in, imgInput);

    // TODO: this is done outside, consider integrating it in here
    //   
    // // obtain a mapping between the distorted image and an undistorted one
    // xf::cv::InitUndistortRectifyMapInverse<XF_CAMERA_MATRIX_SIZE, XF_DIST_COEFF_SIZE, XF_32FC1, HEIGHT, WIDTH,
    //                                  XF_NPPC1>(K_binned_fix, distC_l_fix, irA_l_fix, mapxLMat, mapyLMat,
    //                                            _cm_size, _dc_size);

    // remap
    xf::cv::remap<XF_WIN_ROWS, XF_INTERPOLATION_TYPE, TYPE, TYPE_XY, TYPE, HEIGHT, WIDTH, NPC, XF_USE_URAM>(
        imgInput, imgOutput_rectify, mapX, mapY);

    // resize
    xf::cv::resize<INTERPOLATION, TYPE, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC, MAXDOWNSCALE>(imgOutput_rectify, imgOutput_resize);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, NEWHEIGHT, NEWWIDTH, NPC>(imgOutput_resize, img_out);


    return;
} // End of kernel

} // End of extern C
