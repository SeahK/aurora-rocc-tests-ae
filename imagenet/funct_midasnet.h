#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "midasnet_params.h"
#include "images_xr.h"

uint64_t* midasnet_function_1(int block, bool weight_direct_dram, int num_array, int cid)
{
  uint64_t start, end;
  uint64_t total_fc_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
  uint64_t conv_cycles[45];
  static uint64_t cycles[46];
  bool input_direct_dram = false; bool output_direct_dram = false; bool bias_direct_dram = false; 

  if(block == -1 || block == 0){
        start = read_cycles();

	tiled_opcode_conv_default(
		conv_0_midas1_params.batch_size, conv_0_midas1_params.in_dim, conv_0_midas1_params.in_channels,
		conv_0_midas1_params.out_channels, conv_0_midas1_params.out_dim,
		conv_0_midas1_params.stride, 1, conv_0_midas1_params.padding, conv_0_midas1_params.kernel_size,
		conv_0_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) image_midasnet, (elem_t*) conv_0_midas1_w, (acc_t*) conv_0_midas1_b, (elem_t*) conv_0_midas1_out,

		RELU, conv_0_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[0] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_1_midas1_params.I, conv_1_midas1_params.J, conv_1_midas1_params.K,
		conv_1_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_0_midas1_out, (elem_t*) conv_1_midas1_w, (acc_t*) conv_1_midas1_b, (elem_t*) conv_1_midas1_out,
		RELU, conv_1_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[1] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_2_midas1_params.I, conv_2_midas1_params.J, conv_2_midas1_params.K,
		conv_2_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_1_midas1_out, (elem_t*) conv_2_midas1_w, (acc_t*) conv_2_midas1_b, (elem_t*) conv_2_midas1_out,
		RELU, conv_2_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[2] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_3_midas1_params.I, conv_3_midas1_params.J, conv_3_midas1_params.K,
		conv_3_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_2_midas1_out, (elem_t*) conv_3_midas1_w, (acc_t*) conv_3_midas1_b, (elem_t*) conv_3_midas1_out,
		RELU, conv_3_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[3] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_4_midas1_params.I, conv_4_midas1_params.J, conv_4_midas1_params.K,
		conv_4_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_3_midas1_out, (elem_t*) conv_4_midas1_w, (acc_t*) conv_4_midas1_b, (elem_t*) conv_4_midas1_out,
		RELU, conv_4_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[4] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_5_midas1_params.batch_size, conv_5_midas1_params.in_dim, conv_5_midas1_params.in_channels,
		conv_5_midas1_params.out_channels, conv_5_midas1_params.out_dim,
		conv_5_midas1_params.stride, 1, conv_5_midas1_params.padding, conv_5_midas1_params.kernel_size,
		conv_5_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_4_midas1_out, (elem_t*) conv_5_midas1_w, (acc_t*) conv_5_midas1_b, (elem_t*) conv_5_midas1_out,

		RELU, conv_5_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[5] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_6_midas1_params.I, conv_6_midas1_params.J, conv_6_midas1_params.K,
		conv_6_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_5_midas1_out, (elem_t*) conv_6_midas1_w, (acc_t*) conv_6_midas1_b, (elem_t*) conv_6_midas1_out,
		RELU, conv_6_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[6] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_7_midas1_params.I, conv_7_midas1_params.J, conv_7_midas1_params.K,
		conv_7_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_6_midas1_out, (elem_t*) conv_7_midas1_w, (acc_t*) conv_7_midas1_b, (elem_t*) conv_7_midas1_out,
		RELU, conv_7_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[7] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_8_midas1_params.batch_size, conv_8_midas1_params.in_dim, conv_8_midas1_params.in_channels,
		conv_8_midas1_params.out_channels, conv_8_midas1_params.out_dim,
		conv_8_midas1_params.stride, 1, conv_8_midas1_params.padding, conv_8_midas1_params.kernel_size,
		conv_8_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_7_midas1_out, (elem_t*) conv_8_midas1_w, (acc_t*) conv_8_midas1_b, (elem_t*) conv_8_midas1_out,

		RELU, conv_8_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[8] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_9_midas1_params.I, conv_9_midas1_params.J, conv_9_midas1_params.K,
		conv_9_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_8_midas1_out, (elem_t*) conv_9_midas1_w, (acc_t*) conv_9_midas1_b, (elem_t*) conv_9_midas1_out,
		RELU, conv_9_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[9] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_10_midas1_params.I, conv_10_midas1_params.J, conv_10_midas1_params.K,
		conv_10_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_9_midas1_out, (elem_t*) conv_10_midas1_w, (acc_t*) conv_10_midas1_b, (elem_t*) conv_10_midas1_out,
		RELU, conv_10_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[10] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_11_midas1_params.I, conv_11_midas1_params.J, conv_11_midas1_params.K,
		conv_11_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_10_midas1_out, (elem_t*) conv_11_midas1_w, (acc_t*) conv_11_midas1_b, (elem_t*) conv_11_midas1_out,
		RELU, conv_11_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[11] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_12_midas1_params.I, conv_12_midas1_params.J, conv_12_midas1_params.K,
		conv_12_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_11_midas1_out, (elem_t*) conv_12_midas1_w, (acc_t*) conv_12_midas1_b, (elem_t*) conv_12_midas1_out,
		RELU, conv_12_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[12] = end - start;
  }
  if(block == -1 || block  == 1){

        start = read_cycles();

	tiled_opcode_conv_default(
		conv_13_midas1_params.batch_size, conv_13_midas1_params.in_dim, conv_13_midas1_params.in_channels,
		conv_13_midas1_params.out_channels, conv_13_midas1_params.out_dim,
		conv_13_midas1_params.stride, 1, conv_13_midas1_params.padding, conv_13_midas1_params.kernel_size,
		conv_13_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_12_midas1_out, (elem_t*) conv_13_midas1_w, (acc_t*) conv_13_midas1_b, (elem_t*) conv_13_midas1_out,

		RELU, conv_13_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[13] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_14_midas1_params.I, conv_14_midas1_params.J, conv_14_midas1_params.K,
		conv_14_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_13_midas1_out, (elem_t*) conv_14_midas1_w, (acc_t*) conv_14_midas1_b, (elem_t*) conv_14_midas1_out,
		RELU, conv_14_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[14] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_15_midas1_params.I, conv_15_midas1_params.J, conv_15_midas1_params.K,
		conv_15_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_14_midas1_out, (elem_t*) conv_15_midas1_w, (acc_t*) conv_15_midas1_b, (elem_t*) conv_15_midas1_out,
		RELU, conv_15_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[15] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_16_midas1_params.batch_size, conv_16_midas1_params.in_dim, conv_16_midas1_params.in_channels,
		conv_16_midas1_params.out_channels, conv_16_midas1_params.out_dim,
		conv_16_midas1_params.stride, 1, conv_16_midas1_params.padding, conv_16_midas1_params.kernel_size,
		conv_16_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_15_midas1_out, (elem_t*) conv_16_midas1_w, (acc_t*) conv_16_midas1_b, (elem_t*) conv_16_midas1_out,

		RELU, conv_16_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[16] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_17_midas1_params.I, conv_17_midas1_params.J, conv_17_midas1_params.K,
		conv_17_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_16_midas1_out, (elem_t*) conv_17_midas1_w, (acc_t*) conv_17_midas1_b, (elem_t*) conv_17_midas1_out,
		RELU, conv_17_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[17] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_18_midas1_params.I, conv_18_midas1_params.J, conv_18_midas1_params.K,
		conv_18_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_17_midas1_out, (elem_t*) conv_18_midas1_w, (acc_t*) conv_18_midas1_b, (elem_t*) conv_18_midas1_out,
		RELU, conv_18_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[18] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_19_midas1_params.I, conv_19_midas1_params.J, conv_19_midas1_params.K,
		conv_19_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_18_midas1_out, (elem_t*) conv_19_midas1_w, (acc_t*) conv_19_midas1_b, (elem_t*) conv_19_midas1_out,
		RELU, conv_19_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[19] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_20_midas1_params.I, conv_20_midas1_params.J, conv_20_midas1_params.K,
		conv_20_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_19_midas1_out, (elem_t*) conv_20_midas1_w, (acc_t*) conv_20_midas1_b, (elem_t*) conv_20_midas1_out,
		RELU, conv_20_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[20] = end - start;
  }
  if(block == -1 || block == 2){

        start = read_cycles();

	tiled_opcode_conv_default(
		conv_21_midas1_params.batch_size, conv_21_midas1_params.in_dim, conv_21_midas1_params.in_channels,
		conv_21_midas1_params.out_channels, conv_21_midas1_params.out_dim,
		conv_21_midas1_params.stride, 1, conv_21_midas1_params.padding, conv_21_midas1_params.kernel_size,
		conv_21_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_20_midas1_out, (elem_t*) conv_21_midas1_w, (acc_t*) conv_21_midas1_b, (elem_t*) conv_21_midas1_out,

		RELU, conv_21_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[21] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_22_midas1_params.I, conv_22_midas1_params.J, conv_22_midas1_params.K,
		conv_22_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_21_midas1_out, (elem_t*) conv_22_midas1_w, (acc_t*) conv_22_midas1_b, (elem_t*) conv_22_midas1_out,
		RELU, conv_22_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[22] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_23_midas1_params.I, conv_23_midas1_params.J, conv_23_midas1_params.K,
		conv_23_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_22_midas1_out, (elem_t*) conv_23_midas1_w, (acc_t*) conv_23_midas1_b, (elem_t*) conv_23_midas1_out,
		RELU, conv_23_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[23] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_24_midas1_params.batch_size, conv_24_midas1_params.in_dim, conv_24_midas1_params.in_channels,
		conv_24_midas1_params.out_channels, conv_24_midas1_params.out_dim,
		conv_24_midas1_params.stride, 1, conv_24_midas1_params.padding, conv_24_midas1_params.kernel_size,
		conv_24_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_23_midas1_out, (elem_t*) conv_24_midas1_w, (acc_t*) conv_24_midas1_b, (elem_t*) conv_24_midas1_out,

		RELU, conv_24_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[24] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_25_midas1_params.I, conv_25_midas1_params.J, conv_25_midas1_params.K,
		conv_25_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_24_midas1_out, (elem_t*) conv_25_midas1_w, (acc_t*) conv_25_midas1_b, (elem_t*) conv_25_midas1_out,
		RELU, conv_25_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[25] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_26_midas1_params.I, conv_26_midas1_params.J, conv_26_midas1_params.K,
		conv_26_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_25_midas1_out, (elem_t*) conv_26_midas1_w, (acc_t*) conv_26_midas1_b, (elem_t*) conv_26_midas1_out,
		RELU, conv_26_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[26] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_27_midas1_params.batch_size, conv_27_midas1_params.in_dim, conv_27_midas1_params.in_channels,
		conv_27_midas1_params.out_channels, conv_27_midas1_params.out_dim,
		conv_27_midas1_params.stride, 1, conv_27_midas1_params.padding, conv_27_midas1_params.kernel_size,
		conv_27_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_26_midas1_out, (elem_t*) conv_27_midas1_w, (acc_t*) conv_27_midas1_b, (elem_t*) conv_27_midas1_out,

		RELU, conv_27_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[27] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_28_midas1_params.I, conv_28_midas1_params.J, conv_28_midas1_params.K,
		conv_28_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_27_midas1_out, (elem_t*) conv_28_midas1_w, (acc_t*) conv_28_midas1_b, (elem_t*) conv_28_midas1_out,
		RELU, conv_28_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[28] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_29_midas1_params.I, conv_29_midas1_params.J, conv_29_midas1_params.K,
		conv_29_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_28_midas1_out, (elem_t*) conv_29_midas1_w, (acc_t*) conv_29_midas1_b, (elem_t*) conv_29_midas1_out,
		RELU, conv_29_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[29] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_30_midas1_params.batch_size, conv_30_midas1_params.in_dim, conv_30_midas1_params.in_channels,
		conv_30_midas1_params.out_channels, conv_30_midas1_params.out_dim,
		conv_30_midas1_params.stride, 1, conv_30_midas1_params.padding, conv_30_midas1_params.kernel_size,
		conv_30_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_29_midas1_out, (elem_t*) conv_30_midas1_w, (acc_t*) conv_30_midas1_b, (elem_t*) conv_30_midas1_out,

		RELU, conv_30_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[30] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_31_midas1_params.I, conv_31_midas1_params.J, conv_31_midas1_params.K,
		conv_31_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_30_midas1_out, (elem_t*) conv_31_midas1_w, (acc_t*) conv_31_midas1_b, (elem_t*) conv_31_midas1_out,
		RELU, conv_31_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[31] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_32_midas1_params.I, conv_32_midas1_params.J, conv_32_midas1_params.K,
		conv_32_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_31_midas1_out, (elem_t*) conv_32_midas1_w, (acc_t*) conv_32_midas1_b, (elem_t*) conv_32_midas1_out,
		RELU, conv_32_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[32] = end - start;
  }
  if(block == -1 || block == 3){

        start = read_cycles();

	tiled_opcode_conv_default(
		conv_33_midas1_params.batch_size, conv_33_midas1_params.in_dim, conv_33_midas1_params.in_channels,
		conv_33_midas1_params.out_channels, conv_33_midas1_params.out_dim,
		conv_33_midas1_params.stride, 1, conv_33_midas1_params.padding, conv_33_midas1_params.kernel_size,
		conv_33_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_32_midas1_out, (elem_t*) conv_33_midas1_w, (acc_t*) conv_33_midas1_b, (elem_t*) conv_33_midas1_out,

		RELU, conv_33_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[33] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_34_midas1_params.I, conv_34_midas1_params.J, conv_34_midas1_params.K,
		conv_34_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_33_midas1_out, (elem_t*) conv_34_midas1_w, (acc_t*) conv_34_midas1_b, (elem_t*) conv_34_midas1_out,
		RELU, conv_34_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[34] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_35_midas1_params.I, conv_35_midas1_params.J, conv_35_midas1_params.K,
		conv_35_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_34_midas1_out, (elem_t*) conv_35_midas1_w, (acc_t*) conv_35_midas1_b, (elem_t*) conv_35_midas1_out,
		RELU, conv_35_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[35] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_36_midas1_params.batch_size, conv_36_midas1_params.in_dim, conv_36_midas1_params.in_channels,
		conv_36_midas1_params.out_channels, conv_36_midas1_params.out_dim,
		conv_36_midas1_params.stride, 1, conv_36_midas1_params.padding, conv_36_midas1_params.kernel_size,
		conv_36_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_35_midas1_out, (elem_t*) conv_36_midas1_w, (acc_t*) conv_36_midas1_b, (elem_t*) conv_36_midas1_out,

		RELU, conv_36_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[36] = end - start;
  }
  if(block == -1 || block == 4){

        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_37_midas1_params.I, conv_37_midas1_params.J, conv_37_midas1_params.K,
		conv_37_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_36_midas1_out, (elem_t*) conv_37_midas1_w, (acc_t*) conv_37_midas1_b, (elem_t*) conv_37_midas1_out,
		RELU, conv_37_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[37] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_38_midas1_params.I, conv_38_midas1_params.J, conv_38_midas1_params.K,
		conv_38_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_37_midas1_out, (elem_t*) conv_38_midas1_w, (acc_t*) conv_38_midas1_b, (elem_t*) conv_38_midas1_out,
		RELU, conv_38_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[38] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_39_midas1_params.batch_size, conv_39_midas1_params.in_dim, conv_39_midas1_params.in_channels,
		conv_39_midas1_params.out_channels, conv_39_midas1_params.out_dim,
		conv_39_midas1_params.stride, 1, conv_39_midas1_params.padding, conv_39_midas1_params.kernel_size,
		conv_39_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_38_midas1_out, (elem_t*) conv_39_midas1_w, (acc_t*) conv_39_midas1_b, (elem_t*) conv_39_midas1_out,

		RELU, conv_39_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[39] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_40_midas1_params.I, conv_40_midas1_params.J, conv_40_midas1_params.K,
		conv_40_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_39_midas1_out, (elem_t*) conv_40_midas1_w, (acc_t*) conv_40_midas1_b, (elem_t*) conv_40_midas1_out,
		RELU, conv_40_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[40] = end - start;
  }
  if(block == -1 || block == 5){
   // for(int i = 0; i < 2; i++){
        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_41_midas1_params.I, conv_41_midas1_params.J, conv_41_midas1_params.K,
		conv_41_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_40_midas1_out, (elem_t*) conv_41_midas1_w, (acc_t*) conv_41_midas1_b, (elem_t*) conv_41_midas1_out,
		RELU, conv_41_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[41] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_42_midas1_params.batch_size, conv_42_midas1_params.in_dim, conv_42_midas1_params.in_channels,
		conv_42_midas1_params.out_channels, conv_42_midas1_params.out_dim,
		conv_42_midas1_params.stride, 1, conv_42_midas1_params.padding, conv_42_midas1_params.kernel_size,
		conv_42_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_41_midas1_out, (elem_t*) conv_42_midas1_w, (acc_t*) conv_42_midas1_b, (elem_t*) conv_42_midas1_out,

		RELU, conv_42_midas1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[42] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_43_midas1_params.I, conv_43_midas1_params.J, conv_43_midas1_params.K,
		conv_43_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_42_midas1_out, (elem_t*) conv_43_midas1_w, (acc_t*) conv_43_midas1_b, (elem_t*) conv_43_midas1_out,
		RELU, conv_43_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[43] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_44_midas1_params.I, conv_44_midas1_params.J, conv_44_midas1_params.K,
		conv_44_midas1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_43_midas1_out, (elem_t*) conv_44_midas1_w, (acc_t*) conv_44_midas1_b, (elem_t*) conv_44_midas1_out,
		RELU, conv_44_midas1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[44] = end - start;
  //  }
  }
    for(int i = 0; i < (45+1); i++)
    {
      if(i < 45)
        cycles[i] = conv_cycles[i];
      else
      {
        if(i == (45)) cycles[i] = total_conv_cycles + total_fc_cycles + total_resadd_cycles + other_cycles;
      }
    }
    return cycles;

}
