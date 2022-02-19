/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "ap_int.h"

#include "xf_resize_config.h"

#define DWIDTH 24

extern "C" {
    void resize_accel_streamlined(
                    // ap_uint<INPUT_PTR_WIDTH>* img_inp,
                    // hls::stream<ap_axiu<DWIDTH, 0, 0, 0>>& img_inp,
                    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T>& img_inp,
                    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                    int rows_in,
                    int cols_in,
                    int rows_out,
                    int cols_out) {
        // clang-format off
        // #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
        #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
        #pragma HLS INTERFACE s_axilite port=rows_in
        #pragma HLS INTERFACE s_axilite port=cols_in
        #pragma HLS INTERFACE s_axilite port=rows_out
        #pragma HLS INTERFACE s_axilite port=cols_out
        #pragma HLS INTERFACE s_axilite port=return
        // clang-format on

        xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T> in_mat(rows_in, cols_in);
        #pragma HLS stream variable=in_mat.data depth=2

        xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC_T> out_mat(rows_out, cols_out);
        #pragma HLS stream variable=out_mat.data depth=2
        #pragma HLS DATAFLOW

        // xf::cv::Array2xfMat<INPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC_T>(img_inp, in_mat);
        // ap_axiu<DWIDTH, 0, 0, 0> stream_in = img_inp.read();

    //     int readindex = 0, writeindex = 0;
    // Row_Loop:
    //     for (auto row = 0; row < HEIGHT; row++) {
    // // clang-format off
    // // #pragma HLS LOOP_TRIPCOUNT min=HEIGHT max=HEIGHT
    // #pragma HLS LOOP_FLATTEN off
    //     // clang-format on
    //     Col_Loop:
    //         for (auto col = 0; col < WIDTH; col++) {
    // // clang-format off
    // // #pragma HLS LOOP_TRIPCOUNT min=WIDTH/NPC max=WIDTH/NPC
    // #pragma HLS pipeline
    //             // clang-format on

    //             // XF_TNAME(TYPE, NPC) tmp_src;
    //             // tmp_src = imgOutput.read(readindex++);
    //             // img_out.write(writeindex++, tmp_src);

    //             // ap_uint<DWIDTH> aux;
    //             // img_inp.read(aux);
    //             ap_axiu<DWIDTH, 0, 0, 0> aux;
    //             aux = img_inp.read();
    //             in_mat.write(writeindex++, aux.data);

    //         }
    //     }

        xf::cv::resize<INTERPOLATION, TYPE, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC_T, MAXDOWNSCALE>(img_inp, out_mat);
        xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, NEWHEIGHT, NEWWIDTH, NPC_T>(out_mat, img_out);
        return;
    }
}
