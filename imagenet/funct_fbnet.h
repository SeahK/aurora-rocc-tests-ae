#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "fbnet_params.h"
#include "images_xr.h"

uint64_t* fbnet_function_1(int block, bool weight_direct_dram, int num_array, int cid)
{
  uint64_t start, end;
  uint64_t total_fc_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
  uint64_t conv_cycles[63];
  static uint64_t cycles[64];
  bool input_direct_dram = false; bool output_direct_dram = false; bool bias_direct_dram = false; 

  if(block == -1 || block == 0){
        start = read_cycles();

	tiled_opcode_conv_default(
		conv_0_fb1_params.batch_size, conv_0_fb1_params.in_dim, conv_0_fb1_params.in_channels,
		conv_0_fb1_params.out_channels, conv_0_fb1_params.out_dim,
		conv_0_fb1_params.stride, 1, conv_0_fb1_params.padding, conv_0_fb1_params.kernel_size,
		conv_0_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) image_fbnet, (elem_t*) conv_0_fb1_w, (acc_t*) conv_0_fb1_b, (elem_t*) conv_0_fb1_out,

		RELU, conv_0_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[0] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_1_fb1_params.batch_size, conv_1_fb1_params.in_dim, conv_1_fb1_params.in_channels,
		conv_1_fb1_params.out_channels, conv_1_fb1_params.out_dim,
		conv_1_fb1_params.stride, 1, conv_1_fb1_params.padding, conv_1_fb1_params.kernel_size,
		conv_1_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_0_fb1_out, (elem_t*) conv_1_fb1_w, (acc_t*) conv_1_fb1_b, (elem_t*) conv_1_fb1_out,

		RELU, conv_1_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[1] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_2_fb1_params.I, conv_2_fb1_params.J, conv_2_fb1_params.K,
		conv_2_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_1_fb1_out, (elem_t*) conv_2_fb1_w, (acc_t*) conv_2_fb1_b, (elem_t*) conv_2_fb1_out,
		RELU, conv_2_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[2] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_3_fb1_params.I, conv_3_fb1_params.J, conv_3_fb1_params.K,
		conv_3_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_2_fb1_out, (elem_t*) conv_3_fb1_w, (acc_t*) conv_3_fb1_b, (elem_t*) conv_3_fb1_out,
		RELU, conv_3_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[3] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_4_fb1_params.batch_size, conv_4_fb1_params.in_dim, conv_4_fb1_params.in_channels,
		conv_4_fb1_params.out_channels, conv_4_fb1_params.out_dim,
		conv_4_fb1_params.stride, 1, conv_4_fb1_params.padding, conv_4_fb1_params.kernel_size,
		conv_4_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_3_fb1_out, (elem_t*) conv_4_fb1_w, (acc_t*) conv_4_fb1_b, (elem_t*) conv_4_fb1_out,

		RELU, conv_4_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[4] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_5_fb1_params.I, conv_5_fb1_params.J, conv_5_fb1_params.K,
		conv_5_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_4_fb1_out, (elem_t*) conv_5_fb1_w, (acc_t*) conv_5_fb1_b, (elem_t*) conv_5_fb1_out,
		RELU, conv_5_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[5] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_6_fb1_params.batch_size, conv_6_fb1_params.in_dim, conv_6_fb1_params.in_channels,
		conv_6_fb1_params.out_channels, conv_6_fb1_params.out_dim,
		conv_6_fb1_params.stride, 1, conv_6_fb1_params.padding, conv_6_fb1_params.kernel_size,
		conv_6_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_5_fb1_out, (elem_t*) conv_6_fb1_w, (acc_t*) conv_6_fb1_b, (elem_t*) conv_6_fb1_out,

		RELU, conv_6_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[6] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_7_fb1_params.I, conv_7_fb1_params.J, conv_7_fb1_params.K,
		conv_7_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_6_fb1_out, (elem_t*) conv_7_fb1_w, (acc_t*) conv_7_fb1_b, (elem_t*) conv_7_fb1_out,
		RELU, conv_7_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[7] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_8_fb1_params.batch_size, conv_8_fb1_params.in_dim, conv_8_fb1_params.in_channels,
		conv_8_fb1_params.out_channels, conv_8_fb1_params.out_dim,
		conv_8_fb1_params.stride, 1, conv_8_fb1_params.padding, conv_8_fb1_params.kernel_size,
		conv_8_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_7_fb1_out, (elem_t*) conv_8_fb1_w, (acc_t*) conv_8_fb1_b, (elem_t*) conv_8_fb1_out,

		RELU, conv_8_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[8] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_9_fb1_params.I, conv_9_fb1_params.J, conv_9_fb1_params.K,
		conv_9_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_8_fb1_out, (elem_t*) conv_9_fb1_w, (acc_t*) conv_9_fb1_b, (elem_t*) conv_9_fb1_out,
		RELU, conv_9_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[9] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_10_fb1_params.I, conv_10_fb1_params.J, conv_10_fb1_params.K,
		conv_10_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_9_fb1_out, (elem_t*) conv_10_fb1_w, (acc_t*) conv_10_fb1_b, (elem_t*) conv_10_fb1_out,
		RELU, conv_10_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[10] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_11_fb1_params.batch_size, conv_11_fb1_params.in_dim, conv_11_fb1_params.in_channels,
		conv_11_fb1_params.out_channels, conv_11_fb1_params.out_dim,
		conv_11_fb1_params.stride, 1, conv_11_fb1_params.padding, conv_11_fb1_params.kernel_size,
		conv_11_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_10_fb1_out, (elem_t*) conv_11_fb1_w, (acc_t*) conv_11_fb1_b, (elem_t*) conv_11_fb1_out,

		RELU, conv_11_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[11] = end - start;
  }
  if(block == -1 || block == 1){

        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_12_fb1_params.I, conv_12_fb1_params.J, conv_12_fb1_params.K,
		conv_12_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_11_fb1_out, (elem_t*) conv_12_fb1_w, (acc_t*) conv_12_fb1_b, (elem_t*) conv_12_fb1_out,
		RELU, conv_12_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[12] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_13_fb1_params.I, conv_13_fb1_params.J, conv_13_fb1_params.K,
		conv_13_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_12_fb1_out, (elem_t*) conv_13_fb1_w, (acc_t*) conv_13_fb1_b, (elem_t*) conv_13_fb1_out,
		RELU, conv_13_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[13] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_14_fb1_params.batch_size, conv_14_fb1_params.in_dim, conv_14_fb1_params.in_channels,
		conv_14_fb1_params.out_channels, conv_14_fb1_params.out_dim,
		conv_14_fb1_params.stride, 1, conv_14_fb1_params.padding, conv_14_fb1_params.kernel_size,
		conv_14_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_13_fb1_out, (elem_t*) conv_14_fb1_w, (acc_t*) conv_14_fb1_b, (elem_t*) conv_14_fb1_out,

		RELU, conv_14_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[14] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_15_fb1_params.I, conv_15_fb1_params.J, conv_15_fb1_params.K,
		conv_15_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_14_fb1_out, (elem_t*) conv_15_fb1_w, (acc_t*) conv_15_fb1_b, (elem_t*) conv_15_fb1_out,
		RELU, conv_15_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[15] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_16_fb1_params.I, conv_16_fb1_params.J, conv_16_fb1_params.K,
		conv_16_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_15_fb1_out, (elem_t*) conv_16_fb1_w, (acc_t*) conv_16_fb1_b, (elem_t*) conv_16_fb1_out,
		RELU, conv_16_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[16] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_17_fb1_params.batch_size, conv_17_fb1_params.in_dim, conv_17_fb1_params.in_channels,
		conv_17_fb1_params.out_channels, conv_17_fb1_params.out_dim,
		conv_17_fb1_params.stride, 1, conv_17_fb1_params.padding, conv_17_fb1_params.kernel_size,
		conv_17_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_16_fb1_out, (elem_t*) conv_17_fb1_w, (acc_t*) conv_17_fb1_b, (elem_t*) conv_17_fb1_out,

		RELU, conv_17_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[17] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_18_fb1_params.I, conv_18_fb1_params.J, conv_18_fb1_params.K,
		conv_18_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_17_fb1_out, (elem_t*) conv_18_fb1_w, (acc_t*) conv_18_fb1_b, (elem_t*) conv_18_fb1_out,
		RELU, conv_18_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[18] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_19_fb1_params.I, conv_19_fb1_params.J, conv_19_fb1_params.K,
		conv_19_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_18_fb1_out, (elem_t*) conv_19_fb1_w, (acc_t*) conv_19_fb1_b, (elem_t*) conv_19_fb1_out,
		RELU, conv_19_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[19] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_20_fb1_params.batch_size, conv_20_fb1_params.in_dim, conv_20_fb1_params.in_channels,
		conv_20_fb1_params.out_channels, conv_20_fb1_params.out_dim,
		conv_20_fb1_params.stride, 1, conv_20_fb1_params.padding, conv_20_fb1_params.kernel_size,
		conv_20_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_19_fb1_out, (elem_t*) conv_20_fb1_w, (acc_t*) conv_20_fb1_b, (elem_t*) conv_20_fb1_out,

		RELU, conv_20_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[20] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_21_fb1_params.I, conv_21_fb1_params.J, conv_21_fb1_params.K,
		conv_21_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_20_fb1_out, (elem_t*) conv_21_fb1_w, (acc_t*) conv_21_fb1_b, (elem_t*) conv_21_fb1_out,
		RELU, conv_21_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[21] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_22_fb1_params.I, conv_22_fb1_params.J, conv_22_fb1_params.K,
		conv_22_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_21_fb1_out, (elem_t*) conv_22_fb1_w, (acc_t*) conv_22_fb1_b, (elem_t*) conv_22_fb1_out,
		RELU, conv_22_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[22] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_23_fb1_params.batch_size, conv_23_fb1_params.in_dim, conv_23_fb1_params.in_channels,
		conv_23_fb1_params.out_channels, conv_23_fb1_params.out_dim,
		conv_23_fb1_params.stride, 1, conv_23_fb1_params.padding, conv_23_fb1_params.kernel_size,
		conv_23_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_22_fb1_out, (elem_t*) conv_23_fb1_w, (acc_t*) conv_23_fb1_b, (elem_t*) conv_23_fb1_out,

		RELU, conv_23_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[23] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_24_fb1_params.I, conv_24_fb1_params.J, conv_24_fb1_params.K,
		conv_24_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_23_fb1_out, (elem_t*) conv_24_fb1_w, (acc_t*) conv_24_fb1_b, (elem_t*) conv_24_fb1_out,
		RELU, conv_24_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[24] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_25_fb1_params.I, conv_25_fb1_params.J, conv_25_fb1_params.K,
		conv_25_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_24_fb1_out, (elem_t*) conv_25_fb1_w, (acc_t*) conv_25_fb1_b, (elem_t*) conv_25_fb1_out,
		RELU, conv_25_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[25] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_26_fb1_params.batch_size, conv_26_fb1_params.in_dim, conv_26_fb1_params.in_channels,
		conv_26_fb1_params.out_channels, conv_26_fb1_params.out_dim,
		conv_26_fb1_params.stride, 1, conv_26_fb1_params.padding, conv_26_fb1_params.kernel_size,
		conv_26_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_25_fb1_out, (elem_t*) conv_26_fb1_w, (acc_t*) conv_26_fb1_b, (elem_t*) conv_26_fb1_out,

		RELU, conv_26_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[26] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_27_fb1_params.I, conv_27_fb1_params.J, conv_27_fb1_params.K,
		conv_27_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_26_fb1_out, (elem_t*) conv_27_fb1_w, (acc_t*) conv_27_fb1_b, (elem_t*) conv_27_fb1_out,
		RELU, conv_27_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[27] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_28_fb1_params.I, conv_28_fb1_params.J, conv_28_fb1_params.K,
		conv_28_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_27_fb1_out, (elem_t*) conv_28_fb1_w, (acc_t*) conv_28_fb1_b, (elem_t*) conv_28_fb1_out,
		RELU, conv_28_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[28] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_29_fb1_params.batch_size, conv_29_fb1_params.in_dim, conv_29_fb1_params.in_channels,
		conv_29_fb1_params.out_channels, conv_29_fb1_params.out_dim,
		conv_29_fb1_params.stride, 1, conv_29_fb1_params.padding, conv_29_fb1_params.kernel_size,
		conv_29_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_28_fb1_out, (elem_t*) conv_29_fb1_w, (acc_t*) conv_29_fb1_b, (elem_t*) conv_29_fb1_out,

		RELU, conv_29_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[29] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_30_fb1_params.I, conv_30_fb1_params.J, conv_30_fb1_params.K,
		conv_30_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_29_fb1_out, (elem_t*) conv_30_fb1_w, (acc_t*) conv_30_fb1_b, (elem_t*) conv_30_fb1_out,
		RELU, conv_30_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[30] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_31_fb1_params.I, conv_31_fb1_params.J, conv_31_fb1_params.K,
		conv_31_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_30_fb1_out, (elem_t*) conv_31_fb1_w, (acc_t*) conv_31_fb1_b, (elem_t*) conv_31_fb1_out,
		RELU, conv_31_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[31] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_32_fb1_params.batch_size, conv_32_fb1_params.in_dim, conv_32_fb1_params.in_channels,
		conv_32_fb1_params.out_channels, conv_32_fb1_params.out_dim,
		conv_32_fb1_params.stride, 1, conv_32_fb1_params.padding, conv_32_fb1_params.kernel_size,
		conv_32_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_31_fb1_out, (elem_t*) conv_32_fb1_w, (acc_t*) conv_32_fb1_b, (elem_t*) conv_32_fb1_out,

		RELU, conv_32_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[32] = end - start;
  }
  if(block == -1 || block == 2){

        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_33_fb1_params.I, conv_33_fb1_params.J, conv_33_fb1_params.K,
		conv_33_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_32_fb1_out, (elem_t*) conv_33_fb1_w, (acc_t*) conv_33_fb1_b, (elem_t*) conv_33_fb1_out,
		RELU, conv_33_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[33] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_34_fb1_params.I, conv_34_fb1_params.J, conv_34_fb1_params.K,
		conv_34_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_33_fb1_out, (elem_t*) conv_34_fb1_w, (acc_t*) conv_34_fb1_b, (elem_t*) conv_34_fb1_out,
		RELU, conv_34_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[34] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_35_fb1_params.batch_size, conv_35_fb1_params.in_dim, conv_35_fb1_params.in_channels,
		conv_35_fb1_params.out_channels, conv_35_fb1_params.out_dim,
		conv_35_fb1_params.stride, 1, conv_35_fb1_params.padding, conv_35_fb1_params.kernel_size,
		conv_35_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_34_fb1_out, (elem_t*) conv_35_fb1_w, (acc_t*) conv_35_fb1_b, (elem_t*) conv_35_fb1_out,

		RELU, conv_35_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[35] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_36_fb1_params.I, conv_36_fb1_params.J, conv_36_fb1_params.K,
		conv_36_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_35_fb1_out, (elem_t*) conv_36_fb1_w, (acc_t*) conv_36_fb1_b, (elem_t*) conv_36_fb1_out,
		RELU, conv_36_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[36] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_37_fb1_params.I, conv_37_fb1_params.J, conv_37_fb1_params.K,
		conv_37_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_36_fb1_out, (elem_t*) conv_37_fb1_w, (acc_t*) conv_37_fb1_b, (elem_t*) conv_37_fb1_out,
		RELU, conv_37_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[37] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_38_fb1_params.batch_size, conv_38_fb1_params.in_dim, conv_38_fb1_params.in_channels,
		conv_38_fb1_params.out_channels, conv_38_fb1_params.out_dim,
		conv_38_fb1_params.stride, 1, conv_38_fb1_params.padding, conv_38_fb1_params.kernel_size,
		conv_38_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_37_fb1_out, (elem_t*) conv_38_fb1_w, (acc_t*) conv_38_fb1_b, (elem_t*) conv_38_fb1_out,

		RELU, conv_38_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[38] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_39_fb1_params.I, conv_39_fb1_params.J, conv_39_fb1_params.K,
		conv_39_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_38_fb1_out, (elem_t*) conv_39_fb1_w, (acc_t*) conv_39_fb1_b, (elem_t*) conv_39_fb1_out,
		RELU, conv_39_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[39] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_40_fb1_params.I, conv_40_fb1_params.J, conv_40_fb1_params.K,
		conv_40_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_39_fb1_out, (elem_t*) conv_40_fb1_w, (acc_t*) conv_40_fb1_b, (elem_t*) conv_40_fb1_out,
		RELU, conv_40_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[40] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_41_fb1_params.batch_size, conv_41_fb1_params.in_dim, conv_41_fb1_params.in_channels,
		conv_41_fb1_params.out_channels, conv_41_fb1_params.out_dim,
		conv_41_fb1_params.stride, 1, conv_41_fb1_params.padding, conv_41_fb1_params.kernel_size,
		conv_41_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_40_fb1_out, (elem_t*) conv_41_fb1_w, (acc_t*) conv_41_fb1_b, (elem_t*) conv_41_fb1_out,

		RELU, conv_41_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[41] = end - start;
  }
  if(block == -1 || block == 3){

        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_42_fb1_params.I, conv_42_fb1_params.J, conv_42_fb1_params.K,
		conv_42_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_41_fb1_out, (elem_t*) conv_42_fb1_w, (acc_t*) conv_42_fb1_b, (elem_t*) conv_42_fb1_out,
		RELU, conv_42_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[42] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_43_fb1_params.I, conv_43_fb1_params.J, conv_43_fb1_params.K,
		conv_43_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_42_fb1_out, (elem_t*) conv_43_fb1_w, (acc_t*) conv_43_fb1_b, (elem_t*) conv_43_fb1_out,
		RELU, conv_43_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[43] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_44_fb1_params.batch_size, conv_44_fb1_params.in_dim, conv_44_fb1_params.in_channels,
		conv_44_fb1_params.out_channels, conv_44_fb1_params.out_dim,
		conv_44_fb1_params.stride, 1, conv_44_fb1_params.padding, conv_44_fb1_params.kernel_size,
		conv_44_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_43_fb1_out, (elem_t*) conv_44_fb1_w, (acc_t*) conv_44_fb1_b, (elem_t*) conv_44_fb1_out,

		RELU, conv_44_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[44] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_45_fb1_params.I, conv_45_fb1_params.J, conv_45_fb1_params.K,
		conv_45_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_44_fb1_out, (elem_t*) conv_45_fb1_w, (acc_t*) conv_45_fb1_b, (elem_t*) conv_45_fb1_out,
		RELU, conv_45_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[45] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_46_fb1_params.I, conv_46_fb1_params.J, conv_46_fb1_params.K,
		conv_46_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_45_fb1_out, (elem_t*) conv_46_fb1_w, (acc_t*) conv_46_fb1_b, (elem_t*) conv_46_fb1_out,
		RELU, conv_46_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[46] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_47_fb1_params.batch_size, conv_47_fb1_params.in_dim, conv_47_fb1_params.in_channels,
		conv_47_fb1_params.out_channels, conv_47_fb1_params.out_dim,
		conv_47_fb1_params.stride, 1, conv_47_fb1_params.padding, conv_47_fb1_params.kernel_size,
		conv_47_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_46_fb1_out, (elem_t*) conv_47_fb1_w, (acc_t*) conv_47_fb1_b, (elem_t*) conv_47_fb1_out,

		RELU, conv_47_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[47] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_48_fb1_params.I, conv_48_fb1_params.J, conv_48_fb1_params.K,
		conv_48_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_47_fb1_out, (elem_t*) conv_48_fb1_w, (acc_t*) conv_48_fb1_b, (elem_t*) conv_48_fb1_out,
		RELU, conv_48_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[48] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_49_fb1_params.I, conv_49_fb1_params.J, conv_49_fb1_params.K,
		conv_49_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_48_fb1_out, (elem_t*) conv_49_fb1_w, (acc_t*) conv_49_fb1_b, (elem_t*) conv_49_fb1_out,
		RELU, conv_49_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[49] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_50_fb1_params.batch_size, conv_50_fb1_params.in_dim, conv_50_fb1_params.in_channels,
		conv_50_fb1_params.out_channels, conv_50_fb1_params.out_dim,
		conv_50_fb1_params.stride, 1, conv_50_fb1_params.padding, conv_50_fb1_params.kernel_size,
		conv_50_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_49_fb1_out, (elem_t*) conv_50_fb1_w, (acc_t*) conv_50_fb1_b, (elem_t*) conv_50_fb1_out,

		RELU, conv_50_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[50] = end - start;
  }
  if(block == -1 || block == 4){

        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_51_fb1_params.I, conv_51_fb1_params.J, conv_51_fb1_params.K,
		conv_51_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_50_fb1_out, (elem_t*) conv_51_fb1_w, (acc_t*) conv_51_fb1_b, (elem_t*) conv_51_fb1_out,
		RELU, conv_51_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[51] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_52_fb1_params.I, conv_52_fb1_params.J, conv_52_fb1_params.K,
		conv_52_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_51_fb1_out, (elem_t*) conv_52_fb1_w, (acc_t*) conv_52_fb1_b, (elem_t*) conv_52_fb1_out,
		RELU, conv_52_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[52] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_53_fb1_params.batch_size, conv_53_fb1_params.in_dim, conv_53_fb1_params.in_channels,
		conv_53_fb1_params.out_channels, conv_53_fb1_params.out_dim,
		conv_53_fb1_params.stride, 1, conv_53_fb1_params.padding, conv_53_fb1_params.kernel_size,
		conv_53_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_52_fb1_out, (elem_t*) conv_53_fb1_w, (acc_t*) conv_53_fb1_b, (elem_t*) conv_53_fb1_out,

		RELU, conv_53_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[53] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_54_fb1_params.I, conv_54_fb1_params.J, conv_54_fb1_params.K,
		conv_54_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_53_fb1_out, (elem_t*) conv_54_fb1_w, (acc_t*) conv_54_fb1_b, (elem_t*) conv_54_fb1_out,
		RELU, conv_54_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[54] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_55_fb1_params.I, conv_55_fb1_params.J, conv_55_fb1_params.K,
		conv_55_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_54_fb1_out, (elem_t*) conv_55_fb1_w, (acc_t*) conv_55_fb1_b, (elem_t*) conv_55_fb1_out,
		RELU, conv_55_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[55] = end - start;
  }
  if(block == -1 || block == 5){

        start = read_cycles();

	tiled_opcode_conv_default(
		conv_56_fb1_params.batch_size, conv_56_fb1_params.in_dim, conv_56_fb1_params.in_channels,
		conv_56_fb1_params.out_channels, conv_56_fb1_params.out_dim,
		conv_56_fb1_params.stride, 1, conv_56_fb1_params.padding, conv_56_fb1_params.kernel_size,
		conv_56_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_55_fb1_out, (elem_t*) conv_56_fb1_w, (acc_t*) conv_56_fb1_b, (elem_t*) conv_56_fb1_out,

		RELU, conv_56_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[56] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_57_fb1_params.I, conv_57_fb1_params.J, conv_57_fb1_params.K,
		conv_57_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_56_fb1_out, (elem_t*) conv_57_fb1_w, (acc_t*) conv_57_fb1_b, (elem_t*) conv_57_fb1_out,
		RELU, conv_57_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[57] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_58_fb1_params.I, conv_58_fb1_params.J, conv_58_fb1_params.K,
		conv_58_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_57_fb1_out, (elem_t*) conv_58_fb1_w, (acc_t*) conv_58_fb1_b, (elem_t*) conv_58_fb1_out,
		RELU, conv_58_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[58] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_59_fb1_params.batch_size, conv_59_fb1_params.in_dim, conv_59_fb1_params.in_channels,
		conv_59_fb1_params.out_channels, conv_59_fb1_params.out_dim,
		conv_59_fb1_params.stride, 1, conv_59_fb1_params.padding, conv_59_fb1_params.kernel_size,
		conv_59_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_58_fb1_out, (elem_t*) conv_59_fb1_w, (acc_t*) conv_59_fb1_b, (elem_t*) conv_59_fb1_out,

		RELU, conv_59_fb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[59] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_60_fb1_params.I, conv_60_fb1_params.J, conv_60_fb1_params.K,
		conv_60_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_59_fb1_out, (elem_t*) conv_60_fb1_w, (acc_t*) conv_60_fb1_b, (elem_t*) conv_60_fb1_out,
		RELU, conv_60_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[60] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_61_fb1_params.I, conv_61_fb1_params.J, conv_61_fb1_params.K,
		conv_61_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_60_fb1_out, (elem_t*) conv_61_fb1_w, (acc_t*) conv_61_fb1_b, (elem_t*) conv_61_fb1_out,
		RELU, conv_61_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[61] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_62_fb1_params.I, conv_62_fb1_params.J, conv_62_fb1_params.K,
		conv_62_fb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_61_fb1_out, (elem_t*) conv_62_fb1_w, (acc_t*) conv_62_fb1_b, (elem_t*) conv_62_fb1_out,
		RELU, conv_62_fb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[62] = end - start;
  }
    for(int i = 0; i < (63+1); i++)
    {
      if(i < 63)
        cycles[i] = conv_cycles[i];
      else
      {
        if(i == (63)) cycles[i] = total_conv_cycles + total_fc_cycles + total_resadd_cycles + other_cycles;
      }
    }
    return cycles;

}
