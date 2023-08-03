#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "rgbnet_params.h"
#include "images_xr.h"

uint64_t* rgbnet_function_1(int block, bool weight_direct_dram, int num_array, int cid)
{
  uint64_t start, end;
  uint64_t total_fc_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
  uint64_t conv_cycles[18];
  static uint64_t cycles[19];
  bool input_direct_dram = false; bool output_direct_dram = false; bool bias_direct_dram = false; 

        start = read_cycles();

	tiled_opcode_conv_default(
		conv_0_rgb1_params.batch_size, conv_0_rgb1_params.in_dim, conv_0_rgb1_params.in_channels,
		conv_0_rgb1_params.out_channels, conv_0_rgb1_params.out_dim,
		conv_0_rgb1_params.stride, 1, conv_0_rgb1_params.padding, conv_0_rgb1_params.kernel_size,
		conv_0_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) image_rgbnet, (elem_t*) conv_0_rgb1_w, (acc_t*) conv_0_rgb1_b, (elem_t*) conv_0_rgb1_out,

		RELU, conv_0_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[0] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_1_rgb1_params.batch_size, conv_1_rgb1_params.in_dim, conv_1_rgb1_params.in_channels,
		conv_1_rgb1_params.out_channels, conv_1_rgb1_params.out_dim,
		conv_1_rgb1_params.stride, 1, conv_1_rgb1_params.padding, conv_1_rgb1_params.kernel_size,
		conv_1_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_0_rgb1_out, (elem_t*) conv_1_rgb1_w, (acc_t*) conv_1_rgb1_b, (elem_t*) conv_1_rgb1_out,

		RELU, conv_1_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[1] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_2_rgb1_params.batch_size, conv_2_rgb1_params.in_dim, conv_2_rgb1_params.in_channels,
		conv_2_rgb1_params.out_channels, conv_2_rgb1_params.out_dim,
		conv_2_rgb1_params.stride, 1, conv_2_rgb1_params.padding, conv_2_rgb1_params.kernel_size,
		conv_2_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_1_rgb1_out, (elem_t*) conv_2_rgb1_w, (acc_t*) conv_2_rgb1_b, (elem_t*) conv_2_rgb1_out,

		RELU, conv_2_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[2] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_3_rgb1_params.batch_size, conv_3_rgb1_params.in_dim, conv_3_rgb1_params.in_channels,
		conv_3_rgb1_params.out_channels, conv_3_rgb1_params.out_dim,
		conv_3_rgb1_params.stride, 1, conv_3_rgb1_params.padding, conv_3_rgb1_params.kernel_size,
		conv_3_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_2_rgb1_out, (elem_t*) conv_3_rgb1_w, (acc_t*) conv_3_rgb1_b, (elem_t*) conv_3_rgb1_out,

		RELU, conv_3_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[3] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_4_rgb1_params.batch_size, conv_4_rgb1_params.in_dim, conv_4_rgb1_params.in_channels,
		conv_4_rgb1_params.out_channels, conv_4_rgb1_params.out_dim,
		conv_4_rgb1_params.stride, 1, conv_4_rgb1_params.padding, conv_4_rgb1_params.kernel_size,
		conv_4_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_3_rgb1_out, (elem_t*) conv_4_rgb1_w, (acc_t*) conv_4_rgb1_b, (elem_t*) conv_4_rgb1_out,

		RELU, conv_4_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[4] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_5_rgb1_params.batch_size, conv_5_rgb1_params.in_dim, conv_5_rgb1_params.in_channels,
		conv_5_rgb1_params.out_channels, conv_5_rgb1_params.out_dim,
		conv_5_rgb1_params.stride, 1, conv_5_rgb1_params.padding, conv_5_rgb1_params.kernel_size,
		conv_5_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_4_rgb1_out, (elem_t*) conv_5_rgb1_w, (acc_t*) conv_5_rgb1_b, (elem_t*) conv_5_rgb1_out,

		RELU, conv_5_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[5] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_6_rgb1_params.batch_size, conv_6_rgb1_params.in_dim, conv_6_rgb1_params.in_channels,
		conv_6_rgb1_params.out_channels, conv_6_rgb1_params.out_dim,
		conv_6_rgb1_params.stride, 1, conv_6_rgb1_params.padding, conv_6_rgb1_params.kernel_size,
		conv_6_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_5_rgb1_out, (elem_t*) conv_6_rgb1_w, (acc_t*) conv_6_rgb1_b, (elem_t*) conv_6_rgb1_out,

		RELU, conv_6_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[6] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_7_rgb1_params.batch_size, conv_7_rgb1_params.in_dim, conv_7_rgb1_params.in_channels,
		conv_7_rgb1_params.out_channels, conv_7_rgb1_params.out_dim,
		conv_7_rgb1_params.stride, 1, conv_7_rgb1_params.padding, conv_7_rgb1_params.kernel_size,
		conv_7_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_6_rgb1_out, (elem_t*) conv_7_rgb1_w, (acc_t*) conv_7_rgb1_b, (elem_t*) conv_7_rgb1_out,

		RELU, conv_7_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[7] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_8_rgb1_params.batch_size, conv_8_rgb1_params.in_dim, conv_8_rgb1_params.in_channels,
		conv_8_rgb1_params.out_channels, conv_8_rgb1_params.out_dim,
		conv_8_rgb1_params.stride, 1, conv_8_rgb1_params.padding, conv_8_rgb1_params.kernel_size,
		conv_8_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_7_rgb1_out, (elem_t*) conv_8_rgb1_w, (acc_t*) conv_8_rgb1_b, (elem_t*) conv_8_rgb1_out,

		RELU, conv_8_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[8] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_9_rgb1_params.batch_size, conv_9_rgb1_params.in_dim, conv_9_rgb1_params.in_channels,
		conv_9_rgb1_params.out_channels, conv_9_rgb1_params.out_dim,
		conv_9_rgb1_params.stride, 1, conv_9_rgb1_params.padding, conv_9_rgb1_params.kernel_size,
		conv_9_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_8_rgb1_out, (elem_t*) conv_9_rgb1_w, (acc_t*) conv_9_rgb1_b, (elem_t*) conv_9_rgb1_out,

		RELU, conv_9_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[9] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_10_rgb1_params.batch_size, conv_10_rgb1_params.in_dim, conv_10_rgb1_params.in_channels,
		conv_10_rgb1_params.out_channels, conv_10_rgb1_params.out_dim,
		conv_10_rgb1_params.stride, 1, conv_10_rgb1_params.padding, conv_10_rgb1_params.kernel_size,
		conv_10_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_9_rgb1_out, (elem_t*) conv_10_rgb1_w, (acc_t*) conv_10_rgb1_b, (elem_t*) conv_10_rgb1_out,

		RELU, conv_10_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[10] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_11_rgb1_params.batch_size, conv_11_rgb1_params.in_dim, conv_11_rgb1_params.in_channels,
		conv_11_rgb1_params.out_channels, conv_11_rgb1_params.out_dim,
		conv_11_rgb1_params.stride, 1, conv_11_rgb1_params.padding, conv_11_rgb1_params.kernel_size,
		conv_11_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_10_rgb1_out, (elem_t*) conv_11_rgb1_w, (acc_t*) conv_11_rgb1_b, (elem_t*) conv_11_rgb1_out,

		RELU, conv_11_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[11] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_12_rgb1_params.batch_size, conv_12_rgb1_params.in_dim, conv_12_rgb1_params.in_channels,
		conv_12_rgb1_params.out_channels, conv_12_rgb1_params.out_dim,
		conv_12_rgb1_params.stride, 1, conv_12_rgb1_params.padding, conv_12_rgb1_params.kernel_size,
		conv_12_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_11_rgb1_out, (elem_t*) conv_12_rgb1_w, (acc_t*) conv_12_rgb1_b, (elem_t*) conv_12_rgb1_out,

		RELU, conv_12_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[12] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_13_rgb1_params.batch_size, conv_13_rgb1_params.in_dim, conv_13_rgb1_params.in_channels,
		conv_13_rgb1_params.out_channels, conv_13_rgb1_params.out_dim,
		conv_13_rgb1_params.stride, 1, conv_13_rgb1_params.padding, conv_13_rgb1_params.kernel_size,
		conv_13_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_12_rgb1_out, (elem_t*) conv_13_rgb1_w, (acc_t*) conv_13_rgb1_b, (elem_t*) conv_13_rgb1_out,

		RELU, conv_13_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[13] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_14_rgb1_params.batch_size, conv_14_rgb1_params.in_dim, conv_14_rgb1_params.in_channels,
		conv_14_rgb1_params.out_channels, conv_14_rgb1_params.out_dim,
		conv_14_rgb1_params.stride, 1, conv_14_rgb1_params.padding, conv_14_rgb1_params.kernel_size,
		conv_14_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_13_rgb1_out, (elem_t*) conv_14_rgb1_w, (acc_t*) conv_14_rgb1_b, (elem_t*) conv_14_rgb1_out,

		RELU, conv_14_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[14] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_15_rgb1_params.batch_size, conv_15_rgb1_params.in_dim, conv_15_rgb1_params.in_channels,
		conv_15_rgb1_params.out_channels, conv_15_rgb1_params.out_dim,
		conv_15_rgb1_params.stride, 1, conv_15_rgb1_params.padding, conv_15_rgb1_params.kernel_size,
		conv_15_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_14_rgb1_out, (elem_t*) conv_15_rgb1_w, (acc_t*) conv_15_rgb1_b, (elem_t*) conv_15_rgb1_out,

		RELU, conv_15_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[15] = end - start;


        start = read_cycles();

	tiled_opcode_conv_default(
		conv_16_rgb1_params.batch_size, conv_16_rgb1_params.in_dim, conv_16_rgb1_params.in_channels,
		conv_16_rgb1_params.out_channels, conv_16_rgb1_params.out_dim,
		conv_16_rgb1_params.stride, 1, conv_16_rgb1_params.padding, conv_16_rgb1_params.kernel_size,
		conv_16_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

		(elem_t*) conv_15_rgb1_out, (elem_t*) conv_16_rgb1_w, (acc_t*) conv_16_rgb1_b, (elem_t*) conv_16_rgb1_out,

		RELU, conv_16_rgb1_params.output_scale, 0,
        1, 1, 0, false,
        num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[16] = end - start;


        start = read_cycles();

	tiled_opcode_matmul_nn_default(conv_17_rgb1_params.I, conv_17_rgb1_params.J, conv_17_rgb1_params.K,
		conv_17_rgb1_params.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
		(elem_t*) conv_16_rgb1_out, (elem_t*) conv_17_rgb1_w, (acc_t*) conv_17_rgb1_b, (elem_t*) conv_17_rgb1_out,
		RELU, conv_17_rgb1_params.output_scale, 0, true,
		WS, num_array, cid);

	end = read_cycles();
	total_conv_cycles += end - start;
	conv_cycles[17] = end - start;

    for(int i = 0; i < (18+1); i++)
    {
      if(i < 18)
        cycles[i] = conv_cycles[i];
      else
      {
        if(i == (18)) cycles[i] = total_conv_cycles + total_fc_cycles + total_resadd_cycles + other_cycles;
      }
    }
    return cycles;

}
