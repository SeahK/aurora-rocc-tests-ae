#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "handnet_params.h"
#include "images_xr.h"

uint64_t* handnet_function_1(int block, bool weight_direct_dram, int num_array, int cid)
{
  uint64_t start, end;
  uint64_t total_fc_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
  uint64_t conv_cycles[80];
  static uint64_t cycles[81];
  bool input_direct_dram = false; bool output_direct_dram = false; bool bias_direct_dram = false; 

  for(int i = 0; i < 3; i ++){
    if(block == -1 || block == i){
        start = read_cycles();
	tiled_opcode_conv_default(
		conv_0_hand1_params.batch_size, conv_0_hand1_params.in_dim, conv_0_hand1_params.in_channels,
		conv_0_hand1_params.out_channels, conv_0_hand1_params.out_dim,
		conv_0_hand1_params.stride, 1, conv_0_hand1_params.padding, conv_0_hand1_params.kernel_size,
		conv_0_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) image_handnet, (elem_t*) conv_0_hand1_w, (acc_t*) conv_0_hand1_b, (elem_t*) conv_0_hand1_out,

		RELU, conv_0_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[0] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_1_hand1_params.I, conv_1_hand1_params.J, conv_1_hand1_params.K,
		conv_1_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_0_hand1_out, (elem_t*) conv_1_hand1_w, (acc_t*) conv_1_hand1_b, (elem_t*) conv_1_hand1_out,
		RELU, conv_1_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[1] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_2_hand1_params.batch_size, conv_2_hand1_params.in_dim, conv_2_hand1_params.in_channels,
		conv_2_hand1_params.out_channels, conv_2_hand1_params.out_dim,
		conv_2_hand1_params.stride, 1, conv_2_hand1_params.padding, conv_2_hand1_params.kernel_size,
		conv_2_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_1_hand1_out, (elem_t*) conv_2_hand1_w, (acc_t*) conv_2_hand1_b, (elem_t*) conv_2_hand1_out,

		RELU, conv_2_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[2] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_3_hand1_params.I, conv_3_hand1_params.J, conv_3_hand1_params.K,
		conv_3_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_2_hand1_out, (elem_t*) conv_3_hand1_w, (acc_t*) conv_3_hand1_b, (elem_t*) conv_3_hand1_out,
		RELU, conv_3_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[3] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_4_hand1_params.I, conv_4_hand1_params.J, conv_4_hand1_params.K,
		conv_4_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_3_hand1_out, (elem_t*) conv_4_hand1_w, (acc_t*) conv_4_hand1_b, (elem_t*) conv_4_hand1_out,
		RELU, conv_4_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[4] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_5_hand1_params.I, conv_5_hand1_params.J, conv_5_hand1_params.K,
		conv_5_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_4_hand1_out, (elem_t*) conv_5_hand1_w, (acc_t*) conv_5_hand1_b, (elem_t*) conv_5_hand1_out,
		RELU, conv_5_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[5] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_6_hand1_params.batch_size, conv_6_hand1_params.in_dim, conv_6_hand1_params.in_channels,
		conv_6_hand1_params.out_channels, conv_6_hand1_params.out_dim,
		conv_6_hand1_params.stride, 1, conv_6_hand1_params.padding, conv_6_hand1_params.kernel_size,
		conv_6_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_5_hand1_out, (elem_t*) conv_6_hand1_w, (acc_t*) conv_6_hand1_b, (elem_t*) conv_6_hand1_out,

		RELU, conv_6_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[6] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_7_hand1_params.I, conv_7_hand1_params.J, conv_7_hand1_params.K,
		conv_7_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_6_hand1_out, (elem_t*) conv_7_hand1_w, (acc_t*) conv_7_hand1_b, (elem_t*) conv_7_hand1_out,
		RELU, conv_7_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[7] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_8_hand1_params.I, conv_8_hand1_params.J, conv_8_hand1_params.K,
		conv_8_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_7_hand1_out, (elem_t*) conv_8_hand1_w, (acc_t*) conv_8_hand1_b, (elem_t*) conv_8_hand1_out,
		RELU, conv_8_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[8] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_9_hand1_params.batch_size, conv_9_hand1_params.in_dim, conv_9_hand1_params.in_channels,
		conv_9_hand1_params.out_channels, conv_9_hand1_params.out_dim,
		conv_9_hand1_params.stride, 1, conv_9_hand1_params.padding, conv_9_hand1_params.kernel_size,
		conv_9_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_8_hand1_out, (elem_t*) conv_9_hand1_w, (acc_t*) conv_9_hand1_b, (elem_t*) conv_9_hand1_out,

		RELU, conv_9_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[9] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_10_hand1_params.I, conv_10_hand1_params.J, conv_10_hand1_params.K,
		conv_10_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_9_hand1_out, (elem_t*) conv_10_hand1_w, (acc_t*) conv_10_hand1_b, (elem_t*) conv_10_hand1_out,
		RELU, conv_10_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[10] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_11_hand1_params.I, conv_11_hand1_params.J, conv_11_hand1_params.K,
		conv_11_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_10_hand1_out, (elem_t*) conv_11_hand1_w, (acc_t*) conv_11_hand1_b, (elem_t*) conv_11_hand1_out,
		RELU, conv_11_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[11] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_12_hand1_params.I, conv_12_hand1_params.J, conv_12_hand1_params.K,
		conv_12_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_11_hand1_out, (elem_t*) conv_12_hand1_w, (acc_t*) conv_12_hand1_b, (elem_t*) conv_12_hand1_out,
		RELU, conv_12_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[12] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_13_hand1_params.batch_size, conv_13_hand1_params.in_dim, conv_13_hand1_params.in_channels,
		conv_13_hand1_params.out_channels, conv_13_hand1_params.out_dim,
		conv_13_hand1_params.stride, 1, conv_13_hand1_params.padding, conv_13_hand1_params.kernel_size,
		conv_13_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_12_hand1_out, (elem_t*) conv_13_hand1_w, (acc_t*) conv_13_hand1_b, (elem_t*) conv_13_hand1_out,

		RELU, conv_13_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[13] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_14_hand1_params.I, conv_14_hand1_params.J, conv_14_hand1_params.K,
		conv_14_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_13_hand1_out, (elem_t*) conv_14_hand1_w, (acc_t*) conv_14_hand1_b, (elem_t*) conv_14_hand1_out,
		RELU, conv_14_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[14] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_15_hand1_params.I, conv_15_hand1_params.J, conv_15_hand1_params.K,
		conv_15_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_14_hand1_out, (elem_t*) conv_15_hand1_w, (acc_t*) conv_15_hand1_b, (elem_t*) conv_15_hand1_out,
		RELU, conv_15_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[15] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_16_hand1_params.batch_size, conv_16_hand1_params.in_dim, conv_16_hand1_params.in_channels,
		conv_16_hand1_params.out_channels, conv_16_hand1_params.out_dim,
		conv_16_hand1_params.stride, 1, conv_16_hand1_params.padding, conv_16_hand1_params.kernel_size,
		conv_16_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_15_hand1_out, (elem_t*) conv_16_hand1_w, (acc_t*) conv_16_hand1_b, (elem_t*) conv_16_hand1_out,

		RELU, conv_16_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[16] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_17_hand1_params.I, conv_17_hand1_params.J, conv_17_hand1_params.K,
		conv_17_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_16_hand1_out, (elem_t*) conv_17_hand1_w, (acc_t*) conv_17_hand1_b, (elem_t*) conv_17_hand1_out,
		RELU, conv_17_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[17] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_18_hand1_params.I, conv_18_hand1_params.J, conv_18_hand1_params.K,
		conv_18_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_17_hand1_out, (elem_t*) conv_18_hand1_w, (acc_t*) conv_18_hand1_b, (elem_t*) conv_18_hand1_out,
		RELU, conv_18_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[18] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_19_hand1_params.batch_size, conv_19_hand1_params.in_dim, conv_19_hand1_params.in_channels,
		conv_19_hand1_params.out_channels, conv_19_hand1_params.out_dim,
		conv_19_hand1_params.stride, 1, conv_19_hand1_params.padding, conv_19_hand1_params.kernel_size,
		conv_19_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_18_hand1_out, (elem_t*) conv_19_hand1_w, (acc_t*) conv_19_hand1_b, (elem_t*) conv_19_hand1_out,

		RELU, conv_19_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[19] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_20_hand1_params.I, conv_20_hand1_params.J, conv_20_hand1_params.K,
		conv_20_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_19_hand1_out, (elem_t*) conv_20_hand1_w, (acc_t*) conv_20_hand1_b, (elem_t*) conv_20_hand1_out,
		RELU, conv_20_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[20] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_21_hand1_params.I, conv_21_hand1_params.J, conv_21_hand1_params.K,
		conv_21_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_20_hand1_out, (elem_t*) conv_21_hand1_w, (acc_t*) conv_21_hand1_b, (elem_t*) conv_21_hand1_out,
		RELU, conv_21_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[21] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_22_hand1_params.batch_size, conv_22_hand1_params.in_dim, conv_22_hand1_params.in_channels,
		conv_22_hand1_params.out_channels, conv_22_hand1_params.out_dim,
		conv_22_hand1_params.stride, 1, conv_22_hand1_params.padding, conv_22_hand1_params.kernel_size,
		conv_22_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_21_hand1_out, (elem_t*) conv_22_hand1_w, (acc_t*) conv_22_hand1_b, (elem_t*) conv_22_hand1_out,

		RELU, conv_22_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[22] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_23_hand1_params.I, conv_23_hand1_params.J, conv_23_hand1_params.K,
		conv_23_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_22_hand1_out, (elem_t*) conv_23_hand1_w, (acc_t*) conv_23_hand1_b, (elem_t*) conv_23_hand1_out,
		RELU, conv_23_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[23] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_24_hand1_params.I, conv_24_hand1_params.J, conv_24_hand1_params.K,
		conv_24_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_23_hand1_out, (elem_t*) conv_24_hand1_w, (acc_t*) conv_24_hand1_b, (elem_t*) conv_24_hand1_out,
		RELU, conv_24_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[24] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_25_hand1_params.batch_size, conv_25_hand1_params.in_dim, conv_25_hand1_params.in_channels,
		conv_25_hand1_params.out_channels, conv_25_hand1_params.out_dim,
		conv_25_hand1_params.stride, 1, conv_25_hand1_params.padding, conv_25_hand1_params.kernel_size,
		conv_25_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_24_hand1_out, (elem_t*) conv_25_hand1_w, (acc_t*) conv_25_hand1_b, (elem_t*) conv_25_hand1_out,

		RELU, conv_25_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[25] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_26_hand1_params.I, conv_26_hand1_params.J, conv_26_hand1_params.K,
		conv_26_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_25_hand1_out, (elem_t*) conv_26_hand1_w, (acc_t*) conv_26_hand1_b, (elem_t*) conv_26_hand1_out,
		RELU, conv_26_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[26] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_27_hand1_params.I, conv_27_hand1_params.J, conv_27_hand1_params.K,
		conv_27_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_26_hand1_out, (elem_t*) conv_27_hand1_w, (acc_t*) conv_27_hand1_b, (elem_t*) conv_27_hand1_out,
		RELU, conv_27_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[27] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_28_hand1_params.batch_size, conv_28_hand1_params.in_dim, conv_28_hand1_params.in_channels,
		conv_28_hand1_params.out_channels, conv_28_hand1_params.out_dim,
		conv_28_hand1_params.stride, 1, conv_28_hand1_params.padding, conv_28_hand1_params.kernel_size,
		conv_28_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_27_hand1_out, (elem_t*) conv_28_hand1_w, (acc_t*) conv_28_hand1_b, (elem_t*) conv_28_hand1_out,

		RELU, conv_28_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[28] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_29_hand1_params.I, conv_29_hand1_params.J, conv_29_hand1_params.K,
		conv_29_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_28_hand1_out, (elem_t*) conv_29_hand1_w, (acc_t*) conv_29_hand1_b, (elem_t*) conv_29_hand1_out,
		RELU, conv_29_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[29] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_30_hand1_params.I, conv_30_hand1_params.J, conv_30_hand1_params.K,
		conv_30_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_29_hand1_out, (elem_t*) conv_30_hand1_w, (acc_t*) conv_30_hand1_b, (elem_t*) conv_30_hand1_out,
		RELU, conv_30_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[30] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_31_hand1_params.batch_size, conv_31_hand1_params.in_dim, conv_31_hand1_params.in_channels,
		conv_31_hand1_params.out_channels, conv_31_hand1_params.out_dim,
		conv_31_hand1_params.stride, 1, conv_31_hand1_params.padding, conv_31_hand1_params.kernel_size,
		conv_31_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_30_hand1_out, (elem_t*) conv_31_hand1_w, (acc_t*) conv_31_hand1_b, (elem_t*) conv_31_hand1_out,

		RELU, conv_31_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[31] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_32_hand1_params.I, conv_32_hand1_params.J, conv_32_hand1_params.K,
		conv_32_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_31_hand1_out, (elem_t*) conv_32_hand1_w, (acc_t*) conv_32_hand1_b, (elem_t*) conv_32_hand1_out,
		RELU, conv_32_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[32] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_33_hand1_params.I, conv_33_hand1_params.J, conv_33_hand1_params.K,
		conv_33_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_32_hand1_out, (elem_t*) conv_33_hand1_w, (acc_t*) conv_33_hand1_b, (elem_t*) conv_33_hand1_out,
		RELU, conv_33_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[33] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_34_hand1_params.batch_size, conv_34_hand1_params.in_dim, conv_34_hand1_params.in_channels,
		conv_34_hand1_params.out_channels, conv_34_hand1_params.out_dim,
		conv_34_hand1_params.stride, 1, conv_34_hand1_params.padding, conv_34_hand1_params.kernel_size,
		conv_34_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_33_hand1_out, (elem_t*) conv_34_hand1_w, (acc_t*) conv_34_hand1_b, (elem_t*) conv_34_hand1_out,

		RELU, conv_34_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[34] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_35_hand1_params.I, conv_35_hand1_params.J, conv_35_hand1_params.K,
		conv_35_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_34_hand1_out, (elem_t*) conv_35_hand1_w, (acc_t*) conv_35_hand1_b, (elem_t*) conv_35_hand1_out,
		RELU, conv_35_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[35] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_36_hand1_params.I, conv_36_hand1_params.J, conv_36_hand1_params.K,
		conv_36_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_35_hand1_out, (elem_t*) conv_36_hand1_w, (acc_t*) conv_36_hand1_b, (elem_t*) conv_36_hand1_out,
		RELU, conv_36_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[36] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_37_hand1_params.batch_size, conv_37_hand1_params.in_dim, conv_37_hand1_params.in_channels,
		conv_37_hand1_params.out_channels, conv_37_hand1_params.out_dim,
		conv_37_hand1_params.stride, 1, conv_37_hand1_params.padding, conv_37_hand1_params.kernel_size,
		conv_37_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_36_hand1_out, (elem_t*) conv_37_hand1_w, (acc_t*) conv_37_hand1_b, (elem_t*) conv_37_hand1_out,

		RELU, conv_37_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[37] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_38_hand1_params.I, conv_38_hand1_params.J, conv_38_hand1_params.K,
		conv_38_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_37_hand1_out, (elem_t*) conv_38_hand1_w, (acc_t*) conv_38_hand1_b, (elem_t*) conv_38_hand1_out,
		RELU, conv_38_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[38] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_39_hand1_params.I, conv_39_hand1_params.J, conv_39_hand1_params.K,
		conv_39_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_38_hand1_out, (elem_t*) conv_39_hand1_w, (acc_t*) conv_39_hand1_b, (elem_t*) conv_39_hand1_out,
		RELU, conv_39_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[39] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_40_hand1_params.batch_size, conv_40_hand1_params.in_dim, conv_40_hand1_params.in_channels,
		conv_40_hand1_params.out_channels, conv_40_hand1_params.out_dim,
		conv_40_hand1_params.stride, 1, conv_40_hand1_params.padding, conv_40_hand1_params.kernel_size,
		conv_40_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_39_hand1_out, (elem_t*) conv_40_hand1_w, (acc_t*) conv_40_hand1_b, (elem_t*) conv_40_hand1_out,

		RELU, conv_40_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[40] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_41_hand1_params.I, conv_41_hand1_params.J, conv_41_hand1_params.K,
		conv_41_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_40_hand1_out, (elem_t*) conv_41_hand1_w, (acc_t*) conv_41_hand1_b, (elem_t*) conv_41_hand1_out,
		RELU, conv_41_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[41] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_42_hand1_params.I, conv_42_hand1_params.J, conv_42_hand1_params.K,
		conv_42_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_41_hand1_out, (elem_t*) conv_42_hand1_w, (acc_t*) conv_42_hand1_b, (elem_t*) conv_42_hand1_out,
		RELU, conv_42_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[42] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_43_hand1_params.batch_size, conv_43_hand1_params.in_dim, conv_43_hand1_params.in_channels,
		conv_43_hand1_params.out_channels, conv_43_hand1_params.out_dim,
		conv_43_hand1_params.stride, 1, conv_43_hand1_params.padding, conv_43_hand1_params.kernel_size,
		conv_43_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_42_hand1_out, (elem_t*) conv_43_hand1_w, (acc_t*) conv_43_hand1_b, (elem_t*) conv_43_hand1_out,

		RELU, conv_43_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[43] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_44_hand1_params.I, conv_44_hand1_params.J, conv_44_hand1_params.K,
		conv_44_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_43_hand1_out, (elem_t*) conv_44_hand1_w, (acc_t*) conv_44_hand1_b, (elem_t*) conv_44_hand1_out,
		RELU, conv_44_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[44] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_45_hand1_params.I, conv_45_hand1_params.J, conv_45_hand1_params.K,
		conv_45_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_44_hand1_out, (elem_t*) conv_45_hand1_w, (acc_t*) conv_45_hand1_b, (elem_t*) conv_45_hand1_out,
		RELU, conv_45_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[45] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_46_hand1_params.batch_size, conv_46_hand1_params.in_dim, conv_46_hand1_params.in_channels,
		conv_46_hand1_params.out_channels, conv_46_hand1_params.out_dim,
		conv_46_hand1_params.stride, 1, conv_46_hand1_params.padding, conv_46_hand1_params.kernel_size,
		conv_46_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_45_hand1_out, (elem_t*) conv_46_hand1_w, (acc_t*) conv_46_hand1_b, (elem_t*) conv_46_hand1_out,

		RELU, conv_46_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[46] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_47_hand1_params.I, conv_47_hand1_params.J, conv_47_hand1_params.K,
		conv_47_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_46_hand1_out, (elem_t*) conv_47_hand1_w, (acc_t*) conv_47_hand1_b, (elem_t*) conv_47_hand1_out,
		RELU, conv_47_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[47] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_48_hand1_params.I, conv_48_hand1_params.J, conv_48_hand1_params.K,
		conv_48_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_47_hand1_out, (elem_t*) conv_48_hand1_w, (acc_t*) conv_48_hand1_b, (elem_t*) conv_48_hand1_out,
		RELU, conv_48_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[48] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_49_hand1_params.batch_size, conv_49_hand1_params.in_dim, conv_49_hand1_params.in_channels,
		conv_49_hand1_params.out_channels, conv_49_hand1_params.out_dim,
		conv_49_hand1_params.stride, 1, conv_49_hand1_params.padding, conv_49_hand1_params.kernel_size,
		conv_49_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_48_hand1_out, (elem_t*) conv_49_hand1_w, (acc_t*) conv_49_hand1_b, (elem_t*) conv_49_hand1_out,

		RELU, conv_49_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[49] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_50_hand1_params.I, conv_50_hand1_params.J, conv_50_hand1_params.K,
		conv_50_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_49_hand1_out, (elem_t*) conv_50_hand1_w, (acc_t*) conv_50_hand1_b, (elem_t*) conv_50_hand1_out,
		RELU, conv_50_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[50] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_51_hand1_params.I, conv_51_hand1_params.J, conv_51_hand1_params.K,
		conv_51_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_50_hand1_out, (elem_t*) conv_51_hand1_w, (acc_t*) conv_51_hand1_b, (elem_t*) conv_51_hand1_out,
		RELU, conv_51_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[51] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_52_hand1_params.batch_size, conv_52_hand1_params.in_dim, conv_52_hand1_params.in_channels,
		conv_52_hand1_params.out_channels, conv_52_hand1_params.out_dim,
		conv_52_hand1_params.stride, 1, conv_52_hand1_params.padding, conv_52_hand1_params.kernel_size,
		conv_52_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_51_hand1_out, (elem_t*) conv_52_hand1_w, (acc_t*) conv_52_hand1_b, (elem_t*) conv_52_hand1_out,

		RELU, conv_52_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[52] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_53_hand1_params.I, conv_53_hand1_params.J, conv_53_hand1_params.K,
		conv_53_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_52_hand1_out, (elem_t*) conv_53_hand1_w, (acc_t*) conv_53_hand1_b, (elem_t*) conv_53_hand1_out,
		RELU, conv_53_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[53] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_54_hand1_params.I, conv_54_hand1_params.J, conv_54_hand1_params.K,
		conv_54_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_53_hand1_out, (elem_t*) conv_54_hand1_w, (acc_t*) conv_54_hand1_b, (elem_t*) conv_54_hand1_out,
		RELU, conv_54_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[54] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_55_hand1_params.batch_size, conv_55_hand1_params.in_dim, conv_55_hand1_params.in_channels,
		conv_55_hand1_params.out_channels, conv_55_hand1_params.out_dim,
		conv_55_hand1_params.stride, 1, conv_55_hand1_params.padding, conv_55_hand1_params.kernel_size,
		conv_55_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_54_hand1_out, (elem_t*) conv_55_hand1_w, (acc_t*) conv_55_hand1_b, (elem_t*) conv_55_hand1_out,

		RELU, conv_55_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[55] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_56_hand1_params.I, conv_56_hand1_params.J, conv_56_hand1_params.K,
		conv_56_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_55_hand1_out, (elem_t*) conv_56_hand1_w, (acc_t*) conv_56_hand1_b, (elem_t*) conv_56_hand1_out,
		RELU, conv_56_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[56] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_57_hand1_params.I, conv_57_hand1_params.J, conv_57_hand1_params.K,
		conv_57_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_56_hand1_out, (elem_t*) conv_57_hand1_w, (acc_t*) conv_57_hand1_b, (elem_t*) conv_57_hand1_out,
		RELU, conv_57_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[57] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_58_hand1_params.batch_size, conv_58_hand1_params.in_dim, conv_58_hand1_params.in_channels,
		conv_58_hand1_params.out_channels, conv_58_hand1_params.out_dim,
		conv_58_hand1_params.stride, 1, conv_58_hand1_params.padding, conv_58_hand1_params.kernel_size,
		conv_58_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_57_hand1_out, (elem_t*) conv_58_hand1_w, (acc_t*) conv_58_hand1_b, (elem_t*) conv_58_hand1_out,

		RELU, conv_58_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[58] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_59_hand1_params.I, conv_59_hand1_params.J, conv_59_hand1_params.K,
		conv_59_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_58_hand1_out, (elem_t*) conv_59_hand1_w, (acc_t*) conv_59_hand1_b, (elem_t*) conv_59_hand1_out,
		RELU, conv_59_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[59] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_60_hand1_params.I, conv_60_hand1_params.J, conv_60_hand1_params.K,
		conv_60_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_59_hand1_out, (elem_t*) conv_60_hand1_w, (acc_t*) conv_60_hand1_b, (elem_t*) conv_60_hand1_out,
		RELU, conv_60_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[60] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_61_hand1_params.batch_size, conv_61_hand1_params.in_dim, conv_61_hand1_params.in_channels,
		conv_61_hand1_params.out_channels, conv_61_hand1_params.out_dim,
		conv_61_hand1_params.stride, 1, conv_61_hand1_params.padding, conv_61_hand1_params.kernel_size,
		conv_61_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_60_hand1_out, (elem_t*) conv_61_hand1_w, (acc_t*) conv_61_hand1_b, (elem_t*) conv_61_hand1_out,

		RELU, conv_61_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[61] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_62_hand1_params.I, conv_62_hand1_params.J, conv_62_hand1_params.K,
		conv_62_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_61_hand1_out, (elem_t*) conv_62_hand1_w, (acc_t*) conv_62_hand1_b, (elem_t*) conv_62_hand1_out,
		RELU, conv_62_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[62] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_63_hand1_params.I, conv_63_hand1_params.J, conv_63_hand1_params.K,
		conv_63_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_62_hand1_out, (elem_t*) conv_63_hand1_w, (acc_t*) conv_63_hand1_b, (elem_t*) conv_63_hand1_out,
		RELU, conv_63_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[63] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_64_hand1_params.batch_size, conv_64_hand1_params.in_dim, conv_64_hand1_params.in_channels,
		conv_64_hand1_params.out_channels, conv_64_hand1_params.out_dim,
		conv_64_hand1_params.stride, 1, conv_64_hand1_params.padding, conv_64_hand1_params.kernel_size,
		conv_64_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_63_hand1_out, (elem_t*) conv_64_hand1_w, (acc_t*) conv_64_hand1_b, (elem_t*) conv_64_hand1_out,

		RELU, conv_64_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[64] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_65_hand1_params.I, conv_65_hand1_params.J, conv_65_hand1_params.K,
		conv_65_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_64_hand1_out, (elem_t*) conv_65_hand1_w, (acc_t*) conv_65_hand1_b, (elem_t*) conv_65_hand1_out,
		RELU, conv_65_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[65] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_66_hand1_params.I, conv_66_hand1_params.J, conv_66_hand1_params.K,
		conv_66_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_65_hand1_out, (elem_t*) conv_66_hand1_w, (acc_t*) conv_66_hand1_b, (elem_t*) conv_66_hand1_out,
		RELU, conv_66_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[66] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_67_hand1_params.batch_size, conv_67_hand1_params.in_dim, conv_67_hand1_params.in_channels,
		conv_67_hand1_params.out_channels, conv_67_hand1_params.out_dim,
		conv_67_hand1_params.stride, 1, conv_67_hand1_params.padding, conv_67_hand1_params.kernel_size,
		conv_67_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_66_hand1_out, (elem_t*) conv_67_hand1_w, (acc_t*) conv_67_hand1_b, (elem_t*) conv_67_hand1_out,

		RELU, conv_67_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[67] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_68_hand1_params.I, conv_68_hand1_params.J, conv_68_hand1_params.K,
		conv_68_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_67_hand1_out, (elem_t*) conv_68_hand1_w, (acc_t*) conv_68_hand1_b, (elem_t*) conv_68_hand1_out,
		RELU, conv_68_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[68] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_69_hand1_params.I, conv_69_hand1_params.J, conv_69_hand1_params.K,
		conv_69_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_68_hand1_out, (elem_t*) conv_69_hand1_w, (acc_t*) conv_69_hand1_b, (elem_t*) conv_69_hand1_out,
		RELU, conv_69_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[69] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_70_hand1_params.batch_size, conv_70_hand1_params.in_dim, conv_70_hand1_params.in_channels,
		conv_70_hand1_params.out_channels, conv_70_hand1_params.out_dim,
		conv_70_hand1_params.stride, 1, conv_70_hand1_params.padding, conv_70_hand1_params.kernel_size,
		conv_70_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_69_hand1_out, (elem_t*) conv_70_hand1_w, (acc_t*) conv_70_hand1_b, (elem_t*) conv_70_hand1_out,

		RELU, conv_70_hand1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[70] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_71_hand1_params.I, conv_71_hand1_params.J, conv_71_hand1_params.K,
		conv_71_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_70_hand1_out, (elem_t*) conv_71_hand1_w, (acc_t*) conv_71_hand1_b, (elem_t*) conv_71_hand1_out,
		RELU, conv_71_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[71] = end - start;
    }
  }
  if(block == -1 || block == 3){
        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_72_hand1_params.I, conv_72_hand1_params.J, conv_72_hand1_params.K,
		conv_72_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_71_hand1_out, (elem_t*) conv_72_hand1_w, (acc_t*) conv_72_hand1_b, (elem_t*) conv_72_hand1_out,
		RELU, conv_72_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[72] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_73_hand1_params.I, conv_73_hand1_params.J, conv_73_hand1_params.K,
		conv_73_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_72_hand1_out, (elem_t*) conv_73_hand1_w, (acc_t*) conv_73_hand1_b, (elem_t*) conv_73_hand1_out,
		RELU, conv_73_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[73] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_74_hand1_params.I, conv_74_hand1_params.J, conv_74_hand1_params.K,
		conv_74_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_73_hand1_out, (elem_t*) conv_74_hand1_w, (acc_t*) conv_74_hand1_b, (elem_t*) conv_74_hand1_out,
		RELU, conv_74_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[74] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_75_hand1_params.I, conv_75_hand1_params.J, conv_75_hand1_params.K,
		conv_75_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_74_hand1_out, (elem_t*) conv_75_hand1_w, (acc_t*) conv_75_hand1_b, (elem_t*) conv_75_hand1_out,
		RELU, conv_75_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[75] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_76_hand1_params.I, conv_76_hand1_params.J, conv_76_hand1_params.K,
		conv_76_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_75_hand1_out, (elem_t*) conv_76_hand1_w, (acc_t*) conv_76_hand1_b, (elem_t*) conv_76_hand1_out,
		RELU, conv_76_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[76] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_77_hand1_params.I, conv_77_hand1_params.J, conv_77_hand1_params.K,
		conv_77_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_76_hand1_out, (elem_t*) conv_77_hand1_w, (acc_t*) conv_77_hand1_b, (elem_t*) conv_77_hand1_out,
		RELU, conv_77_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[77] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_78_hand1_params.I, conv_78_hand1_params.J, conv_78_hand1_params.K,
		conv_78_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_77_hand1_out, (elem_t*) conv_78_hand1_w, (acc_t*) conv_78_hand1_b, (elem_t*) conv_78_hand1_out,
		RELU, conv_78_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[78] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_79_hand1_params.I, conv_79_hand1_params.J, conv_79_hand1_params.K,
		conv_79_hand1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_78_hand1_out, (elem_t*) conv_79_hand1_w, (acc_t*) conv_79_hand1_b, (elem_t*) conv_79_hand1_out,
		RELU, conv_79_hand1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[79] = end - start;
  }
    for(int i = 0; i < (80+1); i++)
    {
      if(i < 80)
        cycles[i] = conv_cycles[i];
      else
      {
        if(i == (80)) cycles[i] = total_conv_cycles + total_fc_cycles + total_resadd_cycles + other_cycles;
      }
    }
    return cycles;

}
