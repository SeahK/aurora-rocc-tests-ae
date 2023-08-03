#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "ritnet_params.h"
#include "images_xr.h"

uint64_t* ritnet_function_1(int block, bool weight_direct_dram, int num_array, int cid)
{
  uint64_t start, end;
  uint64_t total_fc_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
  uint64_t conv_cycles[24];
  static uint64_t cycles[25];
  bool input_direct_dram = false; bool output_direct_dram = false; bool bias_direct_dram = false; 

  for (int i = 0; i < 2; i ++){
    if(block == -1 || block == i){
        start = read_cycles();

	tiled_opcode_conv_default(
		conv_0_rit1_params.batch_size, conv_0_rit1_params.in_dim, conv_0_rit1_params.in_channels,
		conv_0_rit1_params.out_channels, conv_0_rit1_params.out_dim,
		conv_0_rit1_params.stride, 1, conv_0_rit1_params.padding, conv_0_rit1_params.kernel_size,
		conv_0_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) image_ritnet, (elem_t*) conv_0_rit1_w, (acc_t*) conv_0_rit1_b, (elem_t*) conv_0_rit1_out,

		RELU, conv_0_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[0] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_1_rit1_params.I, conv_1_rit1_params.J, conv_1_rit1_params.K,
		conv_1_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_0_rit1_out, (elem_t*) conv_1_rit1_w, (acc_t*) conv_1_rit1_b, (elem_t*) conv_1_rit1_out,
		RELU, conv_1_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[1] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_2_rit1_params.batch_size, conv_2_rit1_params.in_dim, conv_2_rit1_params.in_channels,
		conv_2_rit1_params.out_channels, conv_2_rit1_params.out_dim,
		conv_2_rit1_params.stride, 1, conv_2_rit1_params.padding, conv_2_rit1_params.kernel_size,
		conv_2_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_1_rit1_out, (elem_t*) conv_2_rit1_w, (acc_t*) conv_2_rit1_b, (elem_t*) conv_2_rit1_out,

		RELU, conv_2_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[2] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_3_rit1_params.I, conv_3_rit1_params.J, conv_3_rit1_params.K,
		conv_3_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_2_rit1_out, (elem_t*) conv_3_rit1_w, (acc_t*) conv_3_rit1_b, (elem_t*) conv_3_rit1_out,
		RELU, conv_3_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[3] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_4_rit1_params.batch_size, conv_4_rit1_params.in_dim, conv_4_rit1_params.in_channels,
		conv_4_rit1_params.out_channels, conv_4_rit1_params.out_dim,
		conv_4_rit1_params.stride, 1, conv_4_rit1_params.padding, conv_4_rit1_params.kernel_size,
		conv_4_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_3_rit1_out, (elem_t*) conv_4_rit1_w, (acc_t*) conv_4_rit1_b, (elem_t*) conv_4_rit1_out,

		RELU, conv_4_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[4] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_5_rit1_params.batch_size, conv_5_rit1_params.in_dim, conv_5_rit1_params.in_channels,
		conv_5_rit1_params.out_channels, conv_5_rit1_params.out_dim,
		conv_5_rit1_params.stride, 1, conv_5_rit1_params.padding, conv_5_rit1_params.kernel_size,
		conv_5_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_4_rit1_out, (elem_t*) conv_5_rit1_w, (acc_t*) conv_5_rit1_b, (elem_t*) conv_5_rit1_out,

		RELU, conv_5_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[5] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_6_rit1_params.I, conv_6_rit1_params.J, conv_6_rit1_params.K,
		conv_6_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_5_rit1_out, (elem_t*) conv_6_rit1_w, (acc_t*) conv_6_rit1_b, (elem_t*) conv_6_rit1_out,
		RELU, conv_6_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[6] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_7_rit1_params.batch_size, conv_7_rit1_params.in_dim, conv_7_rit1_params.in_channels,
		conv_7_rit1_params.out_channels, conv_7_rit1_params.out_dim,
		conv_7_rit1_params.stride, 1, conv_7_rit1_params.padding, conv_7_rit1_params.kernel_size,
		conv_7_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_6_rit1_out, (elem_t*) conv_7_rit1_w, (acc_t*) conv_7_rit1_b, (elem_t*) conv_7_rit1_out,

		RELU, conv_7_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[7] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_8_rit1_params.I, conv_8_rit1_params.J, conv_8_rit1_params.K,
		conv_8_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_7_rit1_out, (elem_t*) conv_8_rit1_w, (acc_t*) conv_8_rit1_b, (elem_t*) conv_8_rit1_out,
		RELU, conv_8_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[8] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_9_rit1_params.batch_size, conv_9_rit1_params.in_dim, conv_9_rit1_params.in_channels,
		conv_9_rit1_params.out_channels, conv_9_rit1_params.out_dim,
		conv_9_rit1_params.stride, 1, conv_9_rit1_params.padding, conv_9_rit1_params.kernel_size,
		conv_9_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_8_rit1_out, (elem_t*) conv_9_rit1_w, (acc_t*) conv_9_rit1_b, (elem_t*) conv_9_rit1_out,

		RELU, conv_9_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[9] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_10_rit1_params.batch_size, conv_10_rit1_params.in_dim, conv_10_rit1_params.in_channels,
		conv_10_rit1_params.out_channels, conv_10_rit1_params.out_dim,
		conv_10_rit1_params.stride, 1, conv_10_rit1_params.padding, conv_10_rit1_params.kernel_size,
		conv_10_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_9_rit1_out, (elem_t*) conv_10_rit1_w, (acc_t*) conv_10_rit1_b, (elem_t*) conv_10_rit1_out,

		RELU, conv_10_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[10] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_11_rit1_params.I, conv_11_rit1_params.J, conv_11_rit1_params.K,
		conv_11_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_10_rit1_out, (elem_t*) conv_11_rit1_w, (acc_t*) conv_11_rit1_b, (elem_t*) conv_11_rit1_out,
		RELU, conv_11_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[11] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_12_rit1_params.batch_size, conv_12_rit1_params.in_dim, conv_12_rit1_params.in_channels,
		conv_12_rit1_params.out_channels, conv_12_rit1_params.out_dim,
		conv_12_rit1_params.stride, 1, conv_12_rit1_params.padding, conv_12_rit1_params.kernel_size,
		conv_12_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_11_rit1_out, (elem_t*) conv_12_rit1_w, (acc_t*) conv_12_rit1_b, (elem_t*) conv_12_rit1_out,

		RELU, conv_12_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[12] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_13_rit1_params.I, conv_13_rit1_params.J, conv_13_rit1_params.K,
		conv_13_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_12_rit1_out, (elem_t*) conv_13_rit1_w, (acc_t*) conv_13_rit1_b, (elem_t*) conv_13_rit1_out,
		RELU, conv_13_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[13] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_14_rit1_params.batch_size, conv_14_rit1_params.in_dim, conv_14_rit1_params.in_channels,
		conv_14_rit1_params.out_channels, conv_14_rit1_params.out_dim,
		conv_14_rit1_params.stride, 1, conv_14_rit1_params.padding, conv_14_rit1_params.kernel_size,
		conv_14_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_13_rit1_out, (elem_t*) conv_14_rit1_w, (acc_t*) conv_14_rit1_b, (elem_t*) conv_14_rit1_out,

		RELU, conv_14_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[14] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_15_rit1_params.batch_size, conv_15_rit1_params.in_dim, conv_15_rit1_params.in_channels,
		conv_15_rit1_params.out_channels, conv_15_rit1_params.out_dim,
		conv_15_rit1_params.stride, 1, conv_15_rit1_params.padding, conv_15_rit1_params.kernel_size,
		conv_15_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_14_rit1_out, (elem_t*) conv_15_rit1_w, (acc_t*) conv_15_rit1_b, (elem_t*) conv_15_rit1_out,

		RELU, conv_15_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[15] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_16_rit1_params.I, conv_16_rit1_params.J, conv_16_rit1_params.K,
		conv_16_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_15_rit1_out, (elem_t*) conv_16_rit1_w, (acc_t*) conv_16_rit1_b, (elem_t*) conv_16_rit1_out,
		RELU, conv_16_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[16] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_17_rit1_params.batch_size, conv_17_rit1_params.in_dim, conv_17_rit1_params.in_channels,
		conv_17_rit1_params.out_channels, conv_17_rit1_params.out_dim,
		conv_17_rit1_params.stride, 1, conv_17_rit1_params.padding, conv_17_rit1_params.kernel_size,
		conv_17_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_16_rit1_out, (elem_t*) conv_17_rit1_w, (acc_t*) conv_17_rit1_b, (elem_t*) conv_17_rit1_out,

		RELU, conv_17_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[17] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_18_rit1_params.I, conv_18_rit1_params.J, conv_18_rit1_params.K,
		conv_18_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_17_rit1_out, (elem_t*) conv_18_rit1_w, (acc_t*) conv_18_rit1_b, (elem_t*) conv_18_rit1_out,
		RELU, conv_18_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[18] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_19_rit1_params.batch_size, conv_19_rit1_params.in_dim, conv_19_rit1_params.in_channels,
		conv_19_rit1_params.out_channels, conv_19_rit1_params.out_dim,
		conv_19_rit1_params.stride, 1, conv_19_rit1_params.padding, conv_19_rit1_params.kernel_size,
		conv_19_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_18_rit1_out, (elem_t*) conv_19_rit1_w, (acc_t*) conv_19_rit1_b, (elem_t*) conv_19_rit1_out,

		RELU, conv_19_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[19] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_20_rit1_params.batch_size, conv_20_rit1_params.in_dim, conv_20_rit1_params.in_channels,
		conv_20_rit1_params.out_channels, conv_20_rit1_params.out_dim,
		conv_20_rit1_params.stride, 1, conv_20_rit1_params.padding, conv_20_rit1_params.kernel_size,
		conv_20_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_19_rit1_out, (elem_t*) conv_20_rit1_w, (acc_t*) conv_20_rit1_b, (elem_t*) conv_20_rit1_out,

		RELU, conv_20_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[20] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_21_rit1_params.I, conv_21_rit1_params.J, conv_21_rit1_params.K,
		conv_21_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_20_rit1_out, (elem_t*) conv_21_rit1_w, (acc_t*) conv_21_rit1_b, (elem_t*) conv_21_rit1_out,
		RELU, conv_21_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[21] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_22_rit1_params.batch_size, conv_22_rit1_params.in_dim, conv_22_rit1_params.in_channels,
		conv_22_rit1_params.out_channels, conv_22_rit1_params.out_dim,
		conv_22_rit1_params.stride, 1, conv_22_rit1_params.padding, conv_22_rit1_params.kernel_size,
		conv_22_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_21_rit1_out, (elem_t*) conv_22_rit1_w, (acc_t*) conv_22_rit1_b, (elem_t*) conv_22_rit1_out,

		RELU, conv_22_rit1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[22] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_23_rit1_params.I, conv_23_rit1_params.J, conv_23_rit1_params.K,
		conv_23_rit1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_22_rit1_out, (elem_t*) conv_23_rit1_w, (acc_t*) conv_23_rit1_b, (elem_t*) conv_23_rit1_out,
		RELU, conv_23_rit1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[23] = end - start;
    }
  }
    for(int i = 0; i < (24+1); i++)
    {
      if(i < 24)
        cycles[i] = conv_cycles[i];
      else
      {
        if(i == (24)) cycles[i] = total_conv_cycles + total_fc_cycles + total_resadd_cycles + other_cycles;
      }
    }
    return cycles;

}
