#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "rcnnnet_params.h"
#include "images_xr.h"

uint64_t* rcnnnet_function_1(int block, bool weight_direct_dram, int num_array, int cid)
{
  uint64_t start, end;
  uint64_t total_fc_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
  uint64_t conv_cycles[112];
  static uint64_t cycles[113];
  bool input_direct_dram = false; bool output_direct_dram = false; bool bias_direct_dram = false; 

  if(block == -1 || block == 0){
        start = read_cycles();

	tiled_opcode_conv_default(
		conv_0_rcnn1_params.batch_size, conv_0_rcnn1_params.in_dim, conv_0_rcnn1_params.in_channels,
		conv_0_rcnn1_params.out_channels, conv_0_rcnn1_params.out_dim,
		conv_0_rcnn1_params.stride, 1, conv_0_rcnn1_params.padding, conv_0_rcnn1_params.kernel_size,
		conv_0_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) image_rcnnnet, (elem_t*) conv_0_rcnn1_w, (acc_t*) conv_0_rcnn1_b, (elem_t*) conv_0_rcnn1_out,

		RELU, conv_0_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[0] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_1_rcnn1_params.I, conv_1_rcnn1_params.J, conv_1_rcnn1_params.K,
		conv_1_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_0_rcnn1_out, (elem_t*) conv_1_rcnn1_w, (acc_t*) conv_1_rcnn1_b, (elem_t*) conv_1_rcnn1_out,
		RELU, conv_1_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[1] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_2_rcnn1_params.batch_size, conv_2_rcnn1_params.in_dim, conv_2_rcnn1_params.in_channels,
		conv_2_rcnn1_params.out_channels, conv_2_rcnn1_params.out_dim,
		conv_2_rcnn1_params.stride, 1, conv_2_rcnn1_params.padding, conv_2_rcnn1_params.kernel_size,
		conv_2_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_1_rcnn1_out, (elem_t*) conv_2_rcnn1_w, (acc_t*) conv_2_rcnn1_b, (elem_t*) conv_2_rcnn1_out,

		RELU, conv_2_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[2] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_3_rcnn1_params.I, conv_3_rcnn1_params.J, conv_3_rcnn1_params.K,
		conv_3_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_2_rcnn1_out, (elem_t*) conv_3_rcnn1_w, (acc_t*) conv_3_rcnn1_b, (elem_t*) conv_3_rcnn1_out,
		RELU, conv_3_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[3] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_4_rcnn1_params.I, conv_4_rcnn1_params.J, conv_4_rcnn1_params.K,
		conv_4_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_3_rcnn1_out, (elem_t*) conv_4_rcnn1_w, (acc_t*) conv_4_rcnn1_b, (elem_t*) conv_4_rcnn1_out,
		RELU, conv_4_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[4] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_5_rcnn1_params.I, conv_5_rcnn1_params.J, conv_5_rcnn1_params.K,
		conv_5_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_4_rcnn1_out, (elem_t*) conv_5_rcnn1_w, (acc_t*) conv_5_rcnn1_b, (elem_t*) conv_5_rcnn1_out,
		RELU, conv_5_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[5] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_6_rcnn1_params.batch_size, conv_6_rcnn1_params.in_dim, conv_6_rcnn1_params.in_channels,
		conv_6_rcnn1_params.out_channels, conv_6_rcnn1_params.out_dim,
		conv_6_rcnn1_params.stride, 1, conv_6_rcnn1_params.padding, conv_6_rcnn1_params.kernel_size,
		conv_6_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_5_rcnn1_out, (elem_t*) conv_6_rcnn1_w, (acc_t*) conv_6_rcnn1_b, (elem_t*) conv_6_rcnn1_out,

		RELU, conv_6_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[6] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_7_rcnn1_params.I, conv_7_rcnn1_params.J, conv_7_rcnn1_params.K,
		conv_7_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_6_rcnn1_out, (elem_t*) conv_7_rcnn1_w, (acc_t*) conv_7_rcnn1_b, (elem_t*) conv_7_rcnn1_out,
		RELU, conv_7_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[7] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_8_rcnn1_params.I, conv_8_rcnn1_params.J, conv_8_rcnn1_params.K,
		conv_8_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_7_rcnn1_out, (elem_t*) conv_8_rcnn1_w, (acc_t*) conv_8_rcnn1_b, (elem_t*) conv_8_rcnn1_out,
		RELU, conv_8_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[8] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_9_rcnn1_params.batch_size, conv_9_rcnn1_params.in_dim, conv_9_rcnn1_params.in_channels,
		conv_9_rcnn1_params.out_channels, conv_9_rcnn1_params.out_dim,
		conv_9_rcnn1_params.stride, 1, conv_9_rcnn1_params.padding, conv_9_rcnn1_params.kernel_size,
		conv_9_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_8_rcnn1_out, (elem_t*) conv_9_rcnn1_w, (acc_t*) conv_9_rcnn1_b, (elem_t*) conv_9_rcnn1_out,

		RELU, conv_9_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[9] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_10_rcnn1_params.I, conv_10_rcnn1_params.J, conv_10_rcnn1_params.K,
		conv_10_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_9_rcnn1_out, (elem_t*) conv_10_rcnn1_w, (acc_t*) conv_10_rcnn1_b, (elem_t*) conv_10_rcnn1_out,
		RELU, conv_10_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[10] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_11_rcnn1_params.I, conv_11_rcnn1_params.J, conv_11_rcnn1_params.K,
		conv_11_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_10_rcnn1_out, (elem_t*) conv_11_rcnn1_w, (acc_t*) conv_11_rcnn1_b, (elem_t*) conv_11_rcnn1_out,
		RELU, conv_11_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[11] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_12_rcnn1_params.batch_size, conv_12_rcnn1_params.in_dim, conv_12_rcnn1_params.in_channels,
		conv_12_rcnn1_params.out_channels, conv_12_rcnn1_params.out_dim,
		conv_12_rcnn1_params.stride, 1, conv_12_rcnn1_params.padding, conv_12_rcnn1_params.kernel_size,
		conv_12_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_11_rcnn1_out, (elem_t*) conv_12_rcnn1_w, (acc_t*) conv_12_rcnn1_b, (elem_t*) conv_12_rcnn1_out,

		RELU, conv_12_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[12] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_13_rcnn1_params.I, conv_13_rcnn1_params.J, conv_13_rcnn1_params.K,
		conv_13_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_12_rcnn1_out, (elem_t*) conv_13_rcnn1_w, (acc_t*) conv_13_rcnn1_b, (elem_t*) conv_13_rcnn1_out,
		RELU, conv_13_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[13] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_14_rcnn1_params.I, conv_14_rcnn1_params.J, conv_14_rcnn1_params.K,
		conv_14_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_13_rcnn1_out, (elem_t*) conv_14_rcnn1_w, (acc_t*) conv_14_rcnn1_b, (elem_t*) conv_14_rcnn1_out,
		RELU, conv_14_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[14] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_15_rcnn1_params.I, conv_15_rcnn1_params.J, conv_15_rcnn1_params.K,
		conv_15_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_14_rcnn1_out, (elem_t*) conv_15_rcnn1_w, (acc_t*) conv_15_rcnn1_b, (elem_t*) conv_15_rcnn1_out,
		RELU, conv_15_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[15] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_16_rcnn1_params.batch_size, conv_16_rcnn1_params.in_dim, conv_16_rcnn1_params.in_channels,
		conv_16_rcnn1_params.out_channels, conv_16_rcnn1_params.out_dim,
		conv_16_rcnn1_params.stride, 1, conv_16_rcnn1_params.padding, conv_16_rcnn1_params.kernel_size,
		conv_16_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_15_rcnn1_out, (elem_t*) conv_16_rcnn1_w, (acc_t*) conv_16_rcnn1_b, (elem_t*) conv_16_rcnn1_out,

		RELU, conv_16_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[16] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_17_rcnn1_params.I, conv_17_rcnn1_params.J, conv_17_rcnn1_params.K,
		conv_17_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_16_rcnn1_out, (elem_t*) conv_17_rcnn1_w, (acc_t*) conv_17_rcnn1_b, (elem_t*) conv_17_rcnn1_out,
		RELU, conv_17_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[17] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_18_rcnn1_params.I, conv_18_rcnn1_params.J, conv_18_rcnn1_params.K,
		conv_18_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_17_rcnn1_out, (elem_t*) conv_18_rcnn1_w, (acc_t*) conv_18_rcnn1_b, (elem_t*) conv_18_rcnn1_out,
		RELU, conv_18_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[18] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_19_rcnn1_params.batch_size, conv_19_rcnn1_params.in_dim, conv_19_rcnn1_params.in_channels,
		conv_19_rcnn1_params.out_channels, conv_19_rcnn1_params.out_dim,
		conv_19_rcnn1_params.stride, 1, conv_19_rcnn1_params.padding, conv_19_rcnn1_params.kernel_size,
		conv_19_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_18_rcnn1_out, (elem_t*) conv_19_rcnn1_w, (acc_t*) conv_19_rcnn1_b, (elem_t*) conv_19_rcnn1_out,

		RELU, conv_19_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[19] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_20_rcnn1_params.I, conv_20_rcnn1_params.J, conv_20_rcnn1_params.K,
		conv_20_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_19_rcnn1_out, (elem_t*) conv_20_rcnn1_w, (acc_t*) conv_20_rcnn1_b, (elem_t*) conv_20_rcnn1_out,
		RELU, conv_20_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[20] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_21_rcnn1_params.I, conv_21_rcnn1_params.J, conv_21_rcnn1_params.K,
		conv_21_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_20_rcnn1_out, (elem_t*) conv_21_rcnn1_w, (acc_t*) conv_21_rcnn1_b, (elem_t*) conv_21_rcnn1_out,
		RELU, conv_21_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[21] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_22_rcnn1_params.batch_size, conv_22_rcnn1_params.in_dim, conv_22_rcnn1_params.in_channels,
		conv_22_rcnn1_params.out_channels, conv_22_rcnn1_params.out_dim,
		conv_22_rcnn1_params.stride, 1, conv_22_rcnn1_params.padding, conv_22_rcnn1_params.kernel_size,
		conv_22_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_21_rcnn1_out, (elem_t*) conv_22_rcnn1_w, (acc_t*) conv_22_rcnn1_b, (elem_t*) conv_22_rcnn1_out,

		RELU, conv_22_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[22] = end - start;
  }
  if(block == -1 || block == 1){

        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_23_rcnn1_params.I, conv_23_rcnn1_params.J, conv_23_rcnn1_params.K,
		conv_23_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_22_rcnn1_out, (elem_t*) conv_23_rcnn1_w, (acc_t*) conv_23_rcnn1_b, (elem_t*) conv_23_rcnn1_out,
		RELU, conv_23_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[23] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_24_rcnn1_params.I, conv_24_rcnn1_params.J, conv_24_rcnn1_params.K,
		conv_24_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_23_rcnn1_out, (elem_t*) conv_24_rcnn1_w, (acc_t*) conv_24_rcnn1_b, (elem_t*) conv_24_rcnn1_out,
		RELU, conv_24_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[24] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_25_rcnn1_params.batch_size, conv_25_rcnn1_params.in_dim, conv_25_rcnn1_params.in_channels,
		conv_25_rcnn1_params.out_channels, conv_25_rcnn1_params.out_dim,
		conv_25_rcnn1_params.stride, 1, conv_25_rcnn1_params.padding, conv_25_rcnn1_params.kernel_size,
		conv_25_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_24_rcnn1_out, (elem_t*) conv_25_rcnn1_w, (acc_t*) conv_25_rcnn1_b, (elem_t*) conv_25_rcnn1_out,

		RELU, conv_25_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[25] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_26_rcnn1_params.I, conv_26_rcnn1_params.J, conv_26_rcnn1_params.K,
		conv_26_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_25_rcnn1_out, (elem_t*) conv_26_rcnn1_w, (acc_t*) conv_26_rcnn1_b, (elem_t*) conv_26_rcnn1_out,
		RELU, conv_26_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[26] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_27_rcnn1_params.I, conv_27_rcnn1_params.J, conv_27_rcnn1_params.K,
		conv_27_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_26_rcnn1_out, (elem_t*) conv_27_rcnn1_w, (acc_t*) conv_27_rcnn1_b, (elem_t*) conv_27_rcnn1_out,
		RELU, conv_27_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[27] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_28_rcnn1_params.I, conv_28_rcnn1_params.J, conv_28_rcnn1_params.K,
		conv_28_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_27_rcnn1_out, (elem_t*) conv_28_rcnn1_w, (acc_t*) conv_28_rcnn1_b, (elem_t*) conv_28_rcnn1_out,
		RELU, conv_28_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[28] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_29_rcnn1_params.batch_size, conv_29_rcnn1_params.in_dim, conv_29_rcnn1_params.in_channels,
		conv_29_rcnn1_params.out_channels, conv_29_rcnn1_params.out_dim,
		conv_29_rcnn1_params.stride, 1, conv_29_rcnn1_params.padding, conv_29_rcnn1_params.kernel_size,
		conv_29_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_28_rcnn1_out, (elem_t*) conv_29_rcnn1_w, (acc_t*) conv_29_rcnn1_b, (elem_t*) conv_29_rcnn1_out,

		RELU, conv_29_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[29] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_30_rcnn1_params.I, conv_30_rcnn1_params.J, conv_30_rcnn1_params.K,
		conv_30_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_29_rcnn1_out, (elem_t*) conv_30_rcnn1_w, (acc_t*) conv_30_rcnn1_b, (elem_t*) conv_30_rcnn1_out,
		RELU, conv_30_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[30] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_31_rcnn1_params.I, conv_31_rcnn1_params.J, conv_31_rcnn1_params.K,
		conv_31_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_30_rcnn1_out, (elem_t*) conv_31_rcnn1_w, (acc_t*) conv_31_rcnn1_b, (elem_t*) conv_31_rcnn1_out,
		RELU, conv_31_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[31] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_32_rcnn1_params.batch_size, conv_32_rcnn1_params.in_dim, conv_32_rcnn1_params.in_channels,
		conv_32_rcnn1_params.out_channels, conv_32_rcnn1_params.out_dim,
		conv_32_rcnn1_params.stride, 1, conv_32_rcnn1_params.padding, conv_32_rcnn1_params.kernel_size,
		conv_32_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_31_rcnn1_out, (elem_t*) conv_32_rcnn1_w, (acc_t*) conv_32_rcnn1_b, (elem_t*) conv_32_rcnn1_out,

		RELU, conv_32_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[32] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_33_rcnn1_params.I, conv_33_rcnn1_params.J, conv_33_rcnn1_params.K,
		conv_33_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_32_rcnn1_out, (elem_t*) conv_33_rcnn1_w, (acc_t*) conv_33_rcnn1_b, (elem_t*) conv_33_rcnn1_out,
		RELU, conv_33_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[33] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_34_rcnn1_params.I, conv_34_rcnn1_params.J, conv_34_rcnn1_params.K,
		conv_34_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_33_rcnn1_out, (elem_t*) conv_34_rcnn1_w, (acc_t*) conv_34_rcnn1_b, (elem_t*) conv_34_rcnn1_out,
		RELU, conv_34_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[34] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_35_rcnn1_params.batch_size, conv_35_rcnn1_params.in_dim, conv_35_rcnn1_params.in_channels,
		conv_35_rcnn1_params.out_channels, conv_35_rcnn1_params.out_dim,
		conv_35_rcnn1_params.stride, 1, conv_35_rcnn1_params.padding, conv_35_rcnn1_params.kernel_size,
		conv_35_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_34_rcnn1_out, (elem_t*) conv_35_rcnn1_w, (acc_t*) conv_35_rcnn1_b, (elem_t*) conv_35_rcnn1_out,

		RELU, conv_35_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[35] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_36_rcnn1_params.I, conv_36_rcnn1_params.J, conv_36_rcnn1_params.K,
		conv_36_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_35_rcnn1_out, (elem_t*) conv_36_rcnn1_w, (acc_t*) conv_36_rcnn1_b, (elem_t*) conv_36_rcnn1_out,
		RELU, conv_36_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[36] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_37_rcnn1_params.I, conv_37_rcnn1_params.J, conv_37_rcnn1_params.K,
		conv_37_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_36_rcnn1_out, (elem_t*) conv_37_rcnn1_w, (acc_t*) conv_37_rcnn1_b, (elem_t*) conv_37_rcnn1_out,
		RELU, conv_37_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[37] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_38_rcnn1_params.batch_size, conv_38_rcnn1_params.in_dim, conv_38_rcnn1_params.in_channels,
		conv_38_rcnn1_params.out_channels, conv_38_rcnn1_params.out_dim,
		conv_38_rcnn1_params.stride, 1, conv_38_rcnn1_params.padding, conv_38_rcnn1_params.kernel_size,
		conv_38_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_37_rcnn1_out, (elem_t*) conv_38_rcnn1_w, (acc_t*) conv_38_rcnn1_b, (elem_t*) conv_38_rcnn1_out,

		RELU, conv_38_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[38] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_39_rcnn1_params.I, conv_39_rcnn1_params.J, conv_39_rcnn1_params.K,
		conv_39_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_38_rcnn1_out, (elem_t*) conv_39_rcnn1_w, (acc_t*) conv_39_rcnn1_b, (elem_t*) conv_39_rcnn1_out,
		RELU, conv_39_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[39] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_40_rcnn1_params.I, conv_40_rcnn1_params.J, conv_40_rcnn1_params.K,
		conv_40_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_39_rcnn1_out, (elem_t*) conv_40_rcnn1_w, (acc_t*) conv_40_rcnn1_b, (elem_t*) conv_40_rcnn1_out,
		RELU, conv_40_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[40] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_41_rcnn1_params.batch_size, conv_41_rcnn1_params.in_dim, conv_41_rcnn1_params.in_channels,
		conv_41_rcnn1_params.out_channels, conv_41_rcnn1_params.out_dim,
		conv_41_rcnn1_params.stride, 1, conv_41_rcnn1_params.padding, conv_41_rcnn1_params.kernel_size,
		conv_41_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_40_rcnn1_out, (elem_t*) conv_41_rcnn1_w, (acc_t*) conv_41_rcnn1_b, (elem_t*) conv_41_rcnn1_out,

		RELU, conv_41_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[41] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_42_rcnn1_params.I, conv_42_rcnn1_params.J, conv_42_rcnn1_params.K,
		conv_42_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_41_rcnn1_out, (elem_t*) conv_42_rcnn1_w, (acc_t*) conv_42_rcnn1_b, (elem_t*) conv_42_rcnn1_out,
		RELU, conv_42_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[42] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_43_rcnn1_params.I, conv_43_rcnn1_params.J, conv_43_rcnn1_params.K,
		conv_43_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_42_rcnn1_out, (elem_t*) conv_43_rcnn1_w, (acc_t*) conv_43_rcnn1_b, (elem_t*) conv_43_rcnn1_out,
		RELU, conv_43_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[43] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_44_rcnn1_params.batch_size, conv_44_rcnn1_params.in_dim, conv_44_rcnn1_params.in_channels,
		conv_44_rcnn1_params.out_channels, conv_44_rcnn1_params.out_dim,
		conv_44_rcnn1_params.stride, 1, conv_44_rcnn1_params.padding, conv_44_rcnn1_params.kernel_size,
		conv_44_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_43_rcnn1_out, (elem_t*) conv_44_rcnn1_w, (acc_t*) conv_44_rcnn1_b, (elem_t*) conv_44_rcnn1_out,

		RELU, conv_44_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[44] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_45_rcnn1_params.I, conv_45_rcnn1_params.J, conv_45_rcnn1_params.K,
		conv_45_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_44_rcnn1_out, (elem_t*) conv_45_rcnn1_w, (acc_t*) conv_45_rcnn1_b, (elem_t*) conv_45_rcnn1_out,
		RELU, conv_45_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[45] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_46_rcnn1_params.I, conv_46_rcnn1_params.J, conv_46_rcnn1_params.K,
		conv_46_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_45_rcnn1_out, (elem_t*) conv_46_rcnn1_w, (acc_t*) conv_46_rcnn1_b, (elem_t*) conv_46_rcnn1_out,
		RELU, conv_46_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[46] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_47_rcnn1_params.batch_size, conv_47_rcnn1_params.in_dim, conv_47_rcnn1_params.in_channels,
		conv_47_rcnn1_params.out_channels, conv_47_rcnn1_params.out_dim,
		conv_47_rcnn1_params.stride, 1, conv_47_rcnn1_params.padding, conv_47_rcnn1_params.kernel_size,
		conv_47_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_46_rcnn1_out, (elem_t*) conv_47_rcnn1_w, (acc_t*) conv_47_rcnn1_b, (elem_t*) conv_47_rcnn1_out,

		RELU, conv_47_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[47] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_48_rcnn1_params.I, conv_48_rcnn1_params.J, conv_48_rcnn1_params.K,
		conv_48_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_47_rcnn1_out, (elem_t*) conv_48_rcnn1_w, (acc_t*) conv_48_rcnn1_b, (elem_t*) conv_48_rcnn1_out,
		RELU, conv_48_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[48] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_49_rcnn1_params.I, conv_49_rcnn1_params.J, conv_49_rcnn1_params.K,
		conv_49_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_48_rcnn1_out, (elem_t*) conv_49_rcnn1_w, (acc_t*) conv_49_rcnn1_b, (elem_t*) conv_49_rcnn1_out,
		RELU, conv_49_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[49] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_50_rcnn1_params.batch_size, conv_50_rcnn1_params.in_dim, conv_50_rcnn1_params.in_channels,
		conv_50_rcnn1_params.out_channels, conv_50_rcnn1_params.out_dim,
		conv_50_rcnn1_params.stride, 1, conv_50_rcnn1_params.padding, conv_50_rcnn1_params.kernel_size,
		conv_50_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_49_rcnn1_out, (elem_t*) conv_50_rcnn1_w, (acc_t*) conv_50_rcnn1_b, (elem_t*) conv_50_rcnn1_out,

		RELU, conv_50_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[50] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_51_rcnn1_params.I, conv_51_rcnn1_params.J, conv_51_rcnn1_params.K,
		conv_51_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_50_rcnn1_out, (elem_t*) conv_51_rcnn1_w, (acc_t*) conv_51_rcnn1_b, (elem_t*) conv_51_rcnn1_out,
		RELU, conv_51_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[51] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_52_rcnn1_params.I, conv_52_rcnn1_params.J, conv_52_rcnn1_params.K,
		conv_52_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_51_rcnn1_out, (elem_t*) conv_52_rcnn1_w, (acc_t*) conv_52_rcnn1_b, (elem_t*) conv_52_rcnn1_out,
		RELU, conv_52_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[52] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_53_rcnn1_params.batch_size, conv_53_rcnn1_params.in_dim, conv_53_rcnn1_params.in_channels,
		conv_53_rcnn1_params.out_channels, conv_53_rcnn1_params.out_dim,
		conv_53_rcnn1_params.stride, 1, conv_53_rcnn1_params.padding, conv_53_rcnn1_params.kernel_size,
		conv_53_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_52_rcnn1_out, (elem_t*) conv_53_rcnn1_w, (acc_t*) conv_53_rcnn1_b, (elem_t*) conv_53_rcnn1_out,

		RELU, conv_53_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[53] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_54_rcnn1_params.I, conv_54_rcnn1_params.J, conv_54_rcnn1_params.K,
		conv_54_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_53_rcnn1_out, (elem_t*) conv_54_rcnn1_w, (acc_t*) conv_54_rcnn1_b, (elem_t*) conv_54_rcnn1_out,
		RELU, conv_54_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[54] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_55_rcnn1_params.I, conv_55_rcnn1_params.J, conv_55_rcnn1_params.K,
		conv_55_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_54_rcnn1_out, (elem_t*) conv_55_rcnn1_w, (acc_t*) conv_55_rcnn1_b, (elem_t*) conv_55_rcnn1_out,
		RELU, conv_55_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[55] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_56_rcnn1_params.batch_size, conv_56_rcnn1_params.in_dim, conv_56_rcnn1_params.in_channels,
		conv_56_rcnn1_params.out_channels, conv_56_rcnn1_params.out_dim,
		conv_56_rcnn1_params.stride, 1, conv_56_rcnn1_params.padding, conv_56_rcnn1_params.kernel_size,
		conv_56_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_55_rcnn1_out, (elem_t*) conv_56_rcnn1_w, (acc_t*) conv_56_rcnn1_b, (elem_t*) conv_56_rcnn1_out,

		RELU, conv_56_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[56] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_57_rcnn1_params.I, conv_57_rcnn1_params.J, conv_57_rcnn1_params.K,
		conv_57_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_56_rcnn1_out, (elem_t*) conv_57_rcnn1_w, (acc_t*) conv_57_rcnn1_b, (elem_t*) conv_57_rcnn1_out,
		RELU, conv_57_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[57] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_58_rcnn1_params.I, conv_58_rcnn1_params.J, conv_58_rcnn1_params.K,
		conv_58_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_57_rcnn1_out, (elem_t*) conv_58_rcnn1_w, (acc_t*) conv_58_rcnn1_b, (elem_t*) conv_58_rcnn1_out,
		RELU, conv_58_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[58] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_59_rcnn1_params.batch_size, conv_59_rcnn1_params.in_dim, conv_59_rcnn1_params.in_channels,
		conv_59_rcnn1_params.out_channels, conv_59_rcnn1_params.out_dim,
		conv_59_rcnn1_params.stride, 1, conv_59_rcnn1_params.padding, conv_59_rcnn1_params.kernel_size,
		conv_59_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_58_rcnn1_out, (elem_t*) conv_59_rcnn1_w, (acc_t*) conv_59_rcnn1_b, (elem_t*) conv_59_rcnn1_out,

		RELU, conv_59_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[59] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_60_rcnn1_params.I, conv_60_rcnn1_params.J, conv_60_rcnn1_params.K,
		conv_60_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_59_rcnn1_out, (elem_t*) conv_60_rcnn1_w, (acc_t*) conv_60_rcnn1_b, (elem_t*) conv_60_rcnn1_out,
		RELU, conv_60_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[60] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_61_rcnn1_params.I, conv_61_rcnn1_params.J, conv_61_rcnn1_params.K,
		conv_61_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_60_rcnn1_out, (elem_t*) conv_61_rcnn1_w, (acc_t*) conv_61_rcnn1_b, (elem_t*) conv_61_rcnn1_out,
		RELU, conv_61_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[61] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_62_rcnn1_params.batch_size, conv_62_rcnn1_params.in_dim, conv_62_rcnn1_params.in_channels,
		conv_62_rcnn1_params.out_channels, conv_62_rcnn1_params.out_dim,
		conv_62_rcnn1_params.stride, 1, conv_62_rcnn1_params.padding, conv_62_rcnn1_params.kernel_size,
		conv_62_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_61_rcnn1_out, (elem_t*) conv_62_rcnn1_w, (acc_t*) conv_62_rcnn1_b, (elem_t*) conv_62_rcnn1_out,

		RELU, conv_62_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[62] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_63_rcnn1_params.I, conv_63_rcnn1_params.J, conv_63_rcnn1_params.K,
		conv_63_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_62_rcnn1_out, (elem_t*) conv_63_rcnn1_w, (acc_t*) conv_63_rcnn1_b, (elem_t*) conv_63_rcnn1_out,
		RELU, conv_63_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[63] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_64_rcnn1_params.I, conv_64_rcnn1_params.J, conv_64_rcnn1_params.K,
		conv_64_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_63_rcnn1_out, (elem_t*) conv_64_rcnn1_w, (acc_t*) conv_64_rcnn1_b, (elem_t*) conv_64_rcnn1_out,
		RELU, conv_64_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[64] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_65_rcnn1_params.batch_size, conv_65_rcnn1_params.in_dim, conv_65_rcnn1_params.in_channels,
		conv_65_rcnn1_params.out_channels, conv_65_rcnn1_params.out_dim,
		conv_65_rcnn1_params.stride, 1, conv_65_rcnn1_params.padding, conv_65_rcnn1_params.kernel_size,
		conv_65_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_64_rcnn1_out, (elem_t*) conv_65_rcnn1_w, (acc_t*) conv_65_rcnn1_b, (elem_t*) conv_65_rcnn1_out,

		RELU, conv_65_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[65] = end - start;
  }
  if(block == -1 || block == 2){

        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_66_rcnn1_params.I, conv_66_rcnn1_params.J, conv_66_rcnn1_params.K,
		conv_66_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_65_rcnn1_out, (elem_t*) conv_66_rcnn1_w, (acc_t*) conv_66_rcnn1_b, (elem_t*) conv_66_rcnn1_out,
		RELU, conv_66_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[66] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_67_rcnn1_params.I, conv_67_rcnn1_params.J, conv_67_rcnn1_params.K,
		conv_67_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_66_rcnn1_out, (elem_t*) conv_67_rcnn1_w, (acc_t*) conv_67_rcnn1_b, (elem_t*) conv_67_rcnn1_out,
		RELU, conv_67_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[67] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_68_rcnn1_params.batch_size, conv_68_rcnn1_params.in_dim, conv_68_rcnn1_params.in_channels,
		conv_68_rcnn1_params.out_channels, conv_68_rcnn1_params.out_dim,
		conv_68_rcnn1_params.stride, 1, conv_68_rcnn1_params.padding, conv_68_rcnn1_params.kernel_size,
		conv_68_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_67_rcnn1_out, (elem_t*) conv_68_rcnn1_w, (acc_t*) conv_68_rcnn1_b, (elem_t*) conv_68_rcnn1_out,

		RELU, conv_68_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[68] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_69_rcnn1_params.I, conv_69_rcnn1_params.J, conv_69_rcnn1_params.K,
		conv_69_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_68_rcnn1_out, (elem_t*) conv_69_rcnn1_w, (acc_t*) conv_69_rcnn1_b, (elem_t*) conv_69_rcnn1_out,
		RELU, conv_69_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[69] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_70_rcnn1_params.I, conv_70_rcnn1_params.J, conv_70_rcnn1_params.K,
		conv_70_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_69_rcnn1_out, (elem_t*) conv_70_rcnn1_w, (acc_t*) conv_70_rcnn1_b, (elem_t*) conv_70_rcnn1_out,
		RELU, conv_70_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[70] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_71_rcnn1_params.batch_size, conv_71_rcnn1_params.in_dim, conv_71_rcnn1_params.in_channels,
		conv_71_rcnn1_params.out_channels, conv_71_rcnn1_params.out_dim,
		conv_71_rcnn1_params.stride, 1, conv_71_rcnn1_params.padding, conv_71_rcnn1_params.kernel_size,
		conv_71_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_70_rcnn1_out, (elem_t*) conv_71_rcnn1_w, (acc_t*) conv_71_rcnn1_b, (elem_t*) conv_71_rcnn1_out,

		RELU, conv_71_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[71] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_72_rcnn1_params.I, conv_72_rcnn1_params.J, conv_72_rcnn1_params.K,
		conv_72_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_71_rcnn1_out, (elem_t*) conv_72_rcnn1_w, (acc_t*) conv_72_rcnn1_b, (elem_t*) conv_72_rcnn1_out,
		RELU, conv_72_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[72] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_73_rcnn1_params.I, conv_73_rcnn1_params.J, conv_73_rcnn1_params.K,
		conv_73_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_72_rcnn1_out, (elem_t*) conv_73_rcnn1_w, (acc_t*) conv_73_rcnn1_b, (elem_t*) conv_73_rcnn1_out,
		RELU, conv_73_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[73] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_74_rcnn1_params.batch_size, conv_74_rcnn1_params.in_dim, conv_74_rcnn1_params.in_channels,
		conv_74_rcnn1_params.out_channels, conv_74_rcnn1_params.out_dim,
		conv_74_rcnn1_params.stride, 1, conv_74_rcnn1_params.padding, conv_74_rcnn1_params.kernel_size,
		conv_74_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_73_rcnn1_out, (elem_t*) conv_74_rcnn1_w, (acc_t*) conv_74_rcnn1_b, (elem_t*) conv_74_rcnn1_out,

		RELU, conv_74_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[74] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_75_rcnn1_params.I, conv_75_rcnn1_params.J, conv_75_rcnn1_params.K,
		conv_75_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_74_rcnn1_out, (elem_t*) conv_75_rcnn1_w, (acc_t*) conv_75_rcnn1_b, (elem_t*) conv_75_rcnn1_out,
		RELU, conv_75_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[75] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_76_rcnn1_params.I, conv_76_rcnn1_params.J, conv_76_rcnn1_params.K,
		conv_76_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_75_rcnn1_out, (elem_t*) conv_76_rcnn1_w, (acc_t*) conv_76_rcnn1_b, (elem_t*) conv_76_rcnn1_out,
		RELU, conv_76_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[76] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_77_rcnn1_params.batch_size, conv_77_rcnn1_params.in_dim, conv_77_rcnn1_params.in_channels,
		conv_77_rcnn1_params.out_channels, conv_77_rcnn1_params.out_dim,
		conv_77_rcnn1_params.stride, 1, conv_77_rcnn1_params.padding, conv_77_rcnn1_params.kernel_size,
		conv_77_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_76_rcnn1_out, (elem_t*) conv_77_rcnn1_w, (acc_t*) conv_77_rcnn1_b, (elem_t*) conv_77_rcnn1_out,

		RELU, conv_77_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[77] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_78_rcnn1_params.I, conv_78_rcnn1_params.J, conv_78_rcnn1_params.K,
		conv_78_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_77_rcnn1_out, (elem_t*) conv_78_rcnn1_w, (acc_t*) conv_78_rcnn1_b, (elem_t*) conv_78_rcnn1_out,
		RELU, conv_78_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[78] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_79_rcnn1_params.I, conv_79_rcnn1_params.J, conv_79_rcnn1_params.K,
		conv_79_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_78_rcnn1_out, (elem_t*) conv_79_rcnn1_w, (acc_t*) conv_79_rcnn1_b, (elem_t*) conv_79_rcnn1_out,
		RELU, conv_79_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[79] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_80_rcnn1_params.batch_size, conv_80_rcnn1_params.in_dim, conv_80_rcnn1_params.in_channels,
		conv_80_rcnn1_params.out_channels, conv_80_rcnn1_params.out_dim,
		conv_80_rcnn1_params.stride, 1, conv_80_rcnn1_params.padding, conv_80_rcnn1_params.kernel_size,
		conv_80_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_79_rcnn1_out, (elem_t*) conv_80_rcnn1_w, (acc_t*) conv_80_rcnn1_b, (elem_t*) conv_80_rcnn1_out,

		RELU, conv_80_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[80] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_81_rcnn1_params.I, conv_81_rcnn1_params.J, conv_81_rcnn1_params.K,
		conv_81_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_80_rcnn1_out, (elem_t*) conv_81_rcnn1_w, (acc_t*) conv_81_rcnn1_b, (elem_t*) conv_81_rcnn1_out,
		RELU, conv_81_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[81] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_82_rcnn1_params.I, conv_82_rcnn1_params.J, conv_82_rcnn1_params.K,
		conv_82_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_81_rcnn1_out, (elem_t*) conv_82_rcnn1_w, (acc_t*) conv_82_rcnn1_b, (elem_t*) conv_82_rcnn1_out,
		RELU, conv_82_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[82] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_83_rcnn1_params.batch_size, conv_83_rcnn1_params.in_dim, conv_83_rcnn1_params.in_channels,
		conv_83_rcnn1_params.out_channels, conv_83_rcnn1_params.out_dim,
		conv_83_rcnn1_params.stride, 1, conv_83_rcnn1_params.padding, conv_83_rcnn1_params.kernel_size,
		conv_83_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_82_rcnn1_out, (elem_t*) conv_83_rcnn1_w, (acc_t*) conv_83_rcnn1_b, (elem_t*) conv_83_rcnn1_out,

		RELU, conv_83_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[83] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_84_rcnn1_params.I, conv_84_rcnn1_params.J, conv_84_rcnn1_params.K,
		conv_84_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_83_rcnn1_out, (elem_t*) conv_84_rcnn1_w, (acc_t*) conv_84_rcnn1_b, (elem_t*) conv_84_rcnn1_out,
		RELU, conv_84_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[84] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_85_rcnn1_params.I, conv_85_rcnn1_params.J, conv_85_rcnn1_params.K,
		conv_85_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_84_rcnn1_out, (elem_t*) conv_85_rcnn1_w, (acc_t*) conv_85_rcnn1_b, (elem_t*) conv_85_rcnn1_out,
		RELU, conv_85_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[85] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_86_rcnn1_params.batch_size, conv_86_rcnn1_params.in_dim, conv_86_rcnn1_params.in_channels,
		conv_86_rcnn1_params.out_channels, conv_86_rcnn1_params.out_dim,
		conv_86_rcnn1_params.stride, 1, conv_86_rcnn1_params.padding, conv_86_rcnn1_params.kernel_size,
		conv_86_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_85_rcnn1_out, (elem_t*) conv_86_rcnn1_w, (acc_t*) conv_86_rcnn1_b, (elem_t*) conv_86_rcnn1_out,

		RELU, conv_86_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[86] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_87_rcnn1_params.I, conv_87_rcnn1_params.J, conv_87_rcnn1_params.K,
		conv_87_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_86_rcnn1_out, (elem_t*) conv_87_rcnn1_w, (acc_t*) conv_87_rcnn1_b, (elem_t*) conv_87_rcnn1_out,
		RELU, conv_87_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[87] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_88_rcnn1_params.I, conv_88_rcnn1_params.J, conv_88_rcnn1_params.K,
		conv_88_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_87_rcnn1_out, (elem_t*) conv_88_rcnn1_w, (acc_t*) conv_88_rcnn1_b, (elem_t*) conv_88_rcnn1_out,
		RELU, conv_88_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[88] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_89_rcnn1_params.batch_size, conv_89_rcnn1_params.in_dim, conv_89_rcnn1_params.in_channels,
		conv_89_rcnn1_params.out_channels, conv_89_rcnn1_params.out_dim,
		conv_89_rcnn1_params.stride, 1, conv_89_rcnn1_params.padding, conv_89_rcnn1_params.kernel_size,
		conv_89_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_88_rcnn1_out, (elem_t*) conv_89_rcnn1_w, (acc_t*) conv_89_rcnn1_b, (elem_t*) conv_89_rcnn1_out,

		RELU, conv_89_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[89] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_90_rcnn1_params.I, conv_90_rcnn1_params.J, conv_90_rcnn1_params.K,
		conv_90_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_89_rcnn1_out, (elem_t*) conv_90_rcnn1_w, (acc_t*) conv_90_rcnn1_b, (elem_t*) conv_90_rcnn1_out,
		RELU, conv_90_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[90] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_91_rcnn1_params.I, conv_91_rcnn1_params.J, conv_91_rcnn1_params.K,
		conv_91_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_90_rcnn1_out, (elem_t*) conv_91_rcnn1_w, (acc_t*) conv_91_rcnn1_b, (elem_t*) conv_91_rcnn1_out,
		RELU, conv_91_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[91] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_92_rcnn1_params.batch_size, conv_92_rcnn1_params.in_dim, conv_92_rcnn1_params.in_channels,
		conv_92_rcnn1_params.out_channels, conv_92_rcnn1_params.out_dim,
		conv_92_rcnn1_params.stride, 1, conv_92_rcnn1_params.padding, conv_92_rcnn1_params.kernel_size,
		conv_92_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_91_rcnn1_out, (elem_t*) conv_92_rcnn1_w, (acc_t*) conv_92_rcnn1_b, (elem_t*) conv_92_rcnn1_out,

		RELU, conv_92_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[92] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_93_rcnn1_params.I, conv_93_rcnn1_params.J, conv_93_rcnn1_params.K,
		conv_93_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_92_rcnn1_out, (elem_t*) conv_93_rcnn1_w, (acc_t*) conv_93_rcnn1_b, (elem_t*) conv_93_rcnn1_out,
		RELU, conv_93_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[93] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_94_rcnn1_params.I, conv_94_rcnn1_params.J, conv_94_rcnn1_params.K,
		conv_94_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_93_rcnn1_out, (elem_t*) conv_94_rcnn1_w, (acc_t*) conv_94_rcnn1_b, (elem_t*) conv_94_rcnn1_out,
		RELU, conv_94_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[94] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_95_rcnn1_params.batch_size, conv_95_rcnn1_params.in_dim, conv_95_rcnn1_params.in_channels,
		conv_95_rcnn1_params.out_channels, conv_95_rcnn1_params.out_dim,
		conv_95_rcnn1_params.stride, 1, conv_95_rcnn1_params.padding, conv_95_rcnn1_params.kernel_size,
		conv_95_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_94_rcnn1_out, (elem_t*) conv_95_rcnn1_w, (acc_t*) conv_95_rcnn1_b, (elem_t*) conv_95_rcnn1_out,

		RELU, conv_95_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[95] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_96_rcnn1_params.I, conv_96_rcnn1_params.J, conv_96_rcnn1_params.K,
		conv_96_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_95_rcnn1_out, (elem_t*) conv_96_rcnn1_w, (acc_t*) conv_96_rcnn1_b, (elem_t*) conv_96_rcnn1_out,
		RELU, conv_96_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[96] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_97_rcnn1_params.I, conv_97_rcnn1_params.J, conv_97_rcnn1_params.K,
		conv_97_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_96_rcnn1_out, (elem_t*) conv_97_rcnn1_w, (acc_t*) conv_97_rcnn1_b, (elem_t*) conv_97_rcnn1_out,
		RELU, conv_97_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[97] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_98_rcnn1_params.I, conv_98_rcnn1_params.J, conv_98_rcnn1_params.K,
		conv_98_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_97_rcnn1_out, (elem_t*) conv_98_rcnn1_w, (acc_t*) conv_98_rcnn1_b, (elem_t*) conv_98_rcnn1_out,
		RELU, conv_98_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[98] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_99_rcnn1_params.batch_size, conv_99_rcnn1_params.in_dim, conv_99_rcnn1_params.in_channels,
		conv_99_rcnn1_params.out_channels, conv_99_rcnn1_params.out_dim,
		conv_99_rcnn1_params.stride, 1, conv_99_rcnn1_params.padding, conv_99_rcnn1_params.kernel_size,
		conv_99_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_98_rcnn1_out, (elem_t*) conv_99_rcnn1_w, (acc_t*) conv_99_rcnn1_b, (elem_t*) conv_99_rcnn1_out,

		RELU, conv_99_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[99] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_100_rcnn1_params.I, conv_100_rcnn1_params.J, conv_100_rcnn1_params.K,
		conv_100_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_99_rcnn1_out, (elem_t*) conv_100_rcnn1_w, (acc_t*) conv_100_rcnn1_b, (elem_t*) conv_100_rcnn1_out,
		RELU, conv_100_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[100] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_101_rcnn1_params.I, conv_101_rcnn1_params.J, conv_101_rcnn1_params.K,
		conv_101_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_100_rcnn1_out, (elem_t*) conv_101_rcnn1_w, (acc_t*) conv_101_rcnn1_b, (elem_t*) conv_101_rcnn1_out,
		RELU, conv_101_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[101] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_102_rcnn1_params.batch_size, conv_102_rcnn1_params.in_dim, conv_102_rcnn1_params.in_channels,
		conv_102_rcnn1_params.out_channels, conv_102_rcnn1_params.out_dim,
		conv_102_rcnn1_params.stride, 1, conv_102_rcnn1_params.padding, conv_102_rcnn1_params.kernel_size,
		conv_102_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_101_rcnn1_out, (elem_t*) conv_102_rcnn1_w, (acc_t*) conv_102_rcnn1_b, (elem_t*) conv_102_rcnn1_out,

		RELU, conv_102_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[102] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_103_rcnn1_params.I, conv_103_rcnn1_params.J, conv_103_rcnn1_params.K,
		conv_103_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_102_rcnn1_out, (elem_t*) conv_103_rcnn1_w, (acc_t*) conv_103_rcnn1_b, (elem_t*) conv_103_rcnn1_out,
		RELU, conv_103_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[103] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_104_rcnn1_params.I, conv_104_rcnn1_params.J, conv_104_rcnn1_params.K,
		conv_104_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_103_rcnn1_out, (elem_t*) conv_104_rcnn1_w, (acc_t*) conv_104_rcnn1_b, (elem_t*) conv_104_rcnn1_out,
		RELU, conv_104_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[104] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_105_rcnn1_params.batch_size, conv_105_rcnn1_params.in_dim, conv_105_rcnn1_params.in_channels,
		conv_105_rcnn1_params.out_channels, conv_105_rcnn1_params.out_dim,
		conv_105_rcnn1_params.stride, 1, conv_105_rcnn1_params.padding, conv_105_rcnn1_params.kernel_size,
		conv_105_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_104_rcnn1_out, (elem_t*) conv_105_rcnn1_w, (acc_t*) conv_105_rcnn1_b, (elem_t*) conv_105_rcnn1_out,

		RELU, conv_105_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[105] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_106_rcnn1_params.I, conv_106_rcnn1_params.J, conv_106_rcnn1_params.K,
		conv_106_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_105_rcnn1_out, (elem_t*) conv_106_rcnn1_w, (acc_t*) conv_106_rcnn1_b, (elem_t*) conv_106_rcnn1_out,
		RELU, conv_106_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[106] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_107_rcnn1_params.I, conv_107_rcnn1_params.J, conv_107_rcnn1_params.K,
		conv_107_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_106_rcnn1_out, (elem_t*) conv_107_rcnn1_w, (acc_t*) conv_107_rcnn1_b, (elem_t*) conv_107_rcnn1_out,
		RELU, conv_107_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[107] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_108_rcnn1_params.batch_size, conv_108_rcnn1_params.in_dim, conv_108_rcnn1_params.in_channels,
		conv_108_rcnn1_params.out_channels, conv_108_rcnn1_params.out_dim,
		conv_108_rcnn1_params.stride, 1, conv_108_rcnn1_params.padding, conv_108_rcnn1_params.kernel_size,
		conv_108_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_107_rcnn1_out, (elem_t*) conv_108_rcnn1_w, (acc_t*) conv_108_rcnn1_b, (elem_t*) conv_108_rcnn1_out,

		RELU, conv_108_rcnn1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[108] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_109_rcnn1_params.I, conv_109_rcnn1_params.J, conv_109_rcnn1_params.K,
		conv_109_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_108_rcnn1_out, (elem_t*) conv_109_rcnn1_w, (acc_t*) conv_109_rcnn1_b, (elem_t*) conv_109_rcnn1_out,
		RELU, conv_109_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[109] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_110_rcnn1_params.I, conv_110_rcnn1_params.J, conv_110_rcnn1_params.K,
		conv_110_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_109_rcnn1_out, (elem_t*) conv_110_rcnn1_w, (acc_t*) conv_110_rcnn1_b, (elem_t*) conv_110_rcnn1_out,
		RELU, conv_110_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[110] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_111_rcnn1_params.I, conv_111_rcnn1_params.J, conv_111_rcnn1_params.K,
		conv_111_rcnn1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_110_rcnn1_out, (elem_t*) conv_111_rcnn1_w, (acc_t*) conv_111_rcnn1_b, (elem_t*) conv_111_rcnn1_out,
		RELU, conv_111_rcnn1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[111] = end - start;
  }
    for(int i = 0; i < (112+1); i++)
    {
      if(i < 112)
        cycles[i] = conv_cycles[i];
      else
      {
        if(i == (112)) cycles[i] = total_conv_cycles + total_fc_cycles + total_resadd_cycles + other_cycles;
      }
    }
    return cycles;

}
