#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#ifndef DEBUG
#include "squeezenet_params_1.h"
#include "images.h"

#define THREAD_SYNC 0

uint64_t* squeezenet_function_1(bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
#define num_cycle (26+1+3)
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif        

  static uint64_t cycles[num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[26];
    //uint64_t conv_cycles[15];
    uint64_t pool_cycles[1];
    // conv_1

    start = read_cycles();
    tiled_opcode_conv_default(
        conv_1_params_squeeze1.batch_size, conv_1_params_squeeze1.in_dim, conv_1_params_squeeze1.in_channels,
        conv_1_params_squeeze1.out_channels, conv_1_params_squeeze1.out_dim,
        conv_1_params_squeeze1.stride, 1, conv_1_params_squeeze1.padding, conv_1_params_squeeze1.kernel_size,
        conv_1_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)image0, (elem_t*)conv_1_w_squeeze1, (acc_t*)conv_1_b_squeeze1, (elem_t*)conv_1_out_squeeze1,

        RELU, conv_1_params_squeeze1.output_scale, 0,
        1, 0, 0, false,
	//conv_1_params_squeeze1.pool_size, conv_1_params_squeeze1.pool_stride, conv_1_params_squeeze1.pool_padding, false,
	num_array, cid);
        //WS, 2* orow_divide, batch_divide,  cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
//    printf("conv cycles 0: %llu\n", conv_cycles[0]);

#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif        

/*
    // conv_1
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_1_params_squeeze1.batch_size, conv_1_params_squeeze1.in_dim, conv_1_params_squeeze1.in_channels,
        conv_1_params_squeeze1.out_channels, conv_1_params_squeeze1.out_dim,
        conv_1_params_squeeze1.stride, 1, conv_1_params_squeeze1.padding, conv_1_params_squeeze1.kernel_size,
        conv_1_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)image0, (elem_t*)conv_1_w_squeeze1, (acc_t*)conv_1_b_squeeze1, (elem_t*)conv_1_out_squeeze1,
        RELU, conv_1_params_squeeze1.output_scale, 0,
        1, 1, 0, false,
	//conv_1_params_squeeze1.pool_size, conv_1_params_squeeze1.pool_stride, conv_1_params_squeeze1.pool_padding, false,
        WS, orow_divide * 2, batch_divide,  orow_divide + cid, target_util);
    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif   
*//*
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params_squeeze1.batch_size,
        conv_1_params_squeeze1.out_channels, conv_1_params_squeeze1.out_dim, conv_1_params_squeeze1.out_dim_pooled,
        conv_1_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        conv_1_params_squeeze1.pool_size, conv_1_params_squeeze1.pool_stride, conv_1_params_squeeze1.pool_padding,
        (elem_t*)conv_1_out_squeeze1, (elem_t*)conv_1_out_squeeze1_pooled,
	num_array, cid);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif         
   */     
    // conv_2
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_2_params_squeeze1.I, conv_2_params_squeeze1.J, conv_2_params_squeeze1.K, conv_2_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_1_out_squeeze1_pooled, (elem_t*)conv_2_w_squeeze1, (acc_t*)conv_2_b_squeeze1, (elem_t*)conv_2_out_squeeze1,
        RELU, conv_2_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
 
    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_3_params_squeeze1.I, conv_3_params_squeeze1.J, conv_3_params_squeeze1.K, conv_3_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_2_out_squeeze1, (elem_t*)conv_3_w_squeeze1, (acc_t*)conv_3_b_squeeze1, (elem_t*)conv_4_out_squeeze1,
        RELU, conv_3_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_4
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_4_params_squeeze1.batch_size, conv_4_params_squeeze1.in_dim, conv_4_params_squeeze1.in_channels,
        conv_4_params_squeeze1.out_channels, conv_4_params_squeeze1.out_dim,
        conv_4_params_squeeze1.stride, 1, conv_4_params_squeeze1.padding, conv_4_params_squeeze1.kernel_size, conv_4_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_2_out_squeeze1, (elem_t*)conv_4_w_squeeze1, (acc_t*)conv_4_b_squeeze1, (elem_t*)conv_4_out_squeeze1 + conv_4_params_squeeze1.out_channels,

        RELU, conv_4_params_squeeze1.output_scale, 0,
        conv_4_params_squeeze1.pool_size, 0, conv_4_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_5
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_5_params_squeeze1.I, conv_5_params_squeeze1.J, conv_5_params_squeeze1.K, conv_5_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_4_out_squeeze1, (elem_t*)conv_5_w_squeeze1, (acc_t*)conv_5_b_squeeze1, (elem_t*)conv_5_out_squeeze1,
        RELU, conv_5_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_6
     start = read_cycles();
    tiled_opcode_conv_default(
        conv_6_params_squeeze1.batch_size, conv_6_params_squeeze1.in_dim, conv_6_params_squeeze1.in_channels,
        conv_6_params_squeeze1.out_channels, conv_6_params_squeeze1.out_dim,
        conv_6_params_squeeze1.stride, 1, conv_6_params_squeeze1.padding, conv_6_params_squeeze1.kernel_size, conv_6_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_squeeze1, (elem_t*)conv_6_w_squeeze1, (acc_t*)conv_6_b_squeeze1, (elem_t*)conv_7_out_squeeze1_pooled,

        RELU, conv_6_params_squeeze1.output_scale, 0,
        conv_6_params_squeeze1.pool_size, conv_6_params_squeeze1.pool_stride, conv_6_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_7
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_7_params_squeeze1.batch_size, conv_7_params_squeeze1.in_dim, conv_7_params_squeeze1.in_channels,
        conv_7_params_squeeze1.out_channels, conv_7_params_squeeze1.out_dim,
        conv_7_params_squeeze1.stride, 1, conv_7_params_squeeze1.padding, conv_7_params_squeeze1.kernel_size, conv_7_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_squeeze1, (elem_t*)conv_7_w_squeeze1, (acc_t*)conv_7_b_squeeze1, (elem_t*)conv_7_out_squeeze1_pooled + conv_7_params_squeeze1.out_channels,

        RELU, conv_7_params_squeeze1.output_scale, 0,
        conv_7_params_squeeze1.pool_size, conv_7_params_squeeze1.pool_stride, conv_7_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_8
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_8_params_squeeze1.I, conv_8_params_squeeze1.J, conv_8_params_squeeze1.K, conv_8_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_7_out_squeeze1_pooled, (elem_t*)conv_8_w_squeeze1, (acc_t*)conv_8_b_squeeze1, (elem_t*)conv_8_out_squeeze1,
        RELU, conv_8_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_9_params_squeeze1.I, conv_9_params_squeeze1.J, conv_9_params_squeeze1.K, conv_9_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_8_out_squeeze1, (elem_t*)conv_9_w_squeeze1, (acc_t*)conv_9_b_squeeze1, (elem_t*)conv_10_out_squeeze1,
        RELU, conv_9_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_10
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_10_params_squeeze1.batch_size, conv_10_params_squeeze1.in_dim, conv_10_params_squeeze1.in_channels,
        conv_10_params_squeeze1.out_channels, conv_10_params_squeeze1.out_dim,
        conv_10_params_squeeze1.stride, 1, conv_10_params_squeeze1.padding, conv_10_params_squeeze1.kernel_size, conv_10_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_8_out_squeeze1, (elem_t*)conv_10_w_squeeze1, (acc_t*)conv_10_b_squeeze1, (elem_t*)conv_10_out_squeeze1 + conv_10_params_squeeze1.out_channels,

        RELU, conv_10_params_squeeze1.output_scale, 0,
        conv_10_params_squeeze1.pool_size, 0, conv_10_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_11
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_11_params_squeeze1.I, conv_11_params_squeeze1.J, conv_11_params_squeeze1.K, conv_11_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_10_out_squeeze1, (elem_t*)conv_11_w_squeeze1, (acc_t*)conv_11_b_squeeze1, (elem_t*)conv_11_out_squeeze1,
        RELU, conv_11_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_12
     start = read_cycles();
    tiled_opcode_conv_default(
        conv_12_params_squeeze1.batch_size, conv_12_params_squeeze1.in_dim, conv_12_params_squeeze1.in_channels,
        conv_12_params_squeeze1.out_channels, conv_12_params_squeeze1.out_dim,
        conv_12_params_squeeze1.stride, 1, conv_12_params_squeeze1.padding, conv_12_params_squeeze1.kernel_size, conv_12_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_11_out_squeeze1, (elem_t*)conv_12_w_squeeze1, (acc_t*)conv_12_b_squeeze1, (elem_t*)conv_13_out_squeeze1_pooled,

        RELU, conv_12_params_squeeze1.output_scale, 0,
        conv_12_params_squeeze1.pool_size, conv_12_params_squeeze1.pool_stride, conv_12_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_13
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_13_params_squeeze1.batch_size, conv_13_params_squeeze1.in_dim, conv_13_params_squeeze1.in_channels,
        conv_13_params_squeeze1.out_channels, conv_13_params_squeeze1.out_dim,
        conv_13_params_squeeze1.stride, 1, conv_13_params_squeeze1.padding, conv_13_params_squeeze1.kernel_size, conv_13_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_11_out_squeeze1, (elem_t*)conv_13_w_squeeze1, (acc_t*)conv_13_b_squeeze1, (elem_t*)conv_13_out_squeeze1_pooled + conv_13_params_squeeze1.out_channels,

        RELU, conv_13_params_squeeze1.output_scale, 0,
        conv_13_params_squeeze1.pool_size, conv_13_params_squeeze1.pool_stride, conv_13_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_14
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_14_params_squeeze1.I, conv_14_params_squeeze1.J, conv_14_params_squeeze1.K, conv_14_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_13_out_squeeze1_pooled, (elem_t*)conv_14_w_squeeze1, (acc_t*)conv_14_b_squeeze1, (elem_t*)conv_14_out_squeeze1,
        RELU, conv_14_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_15_params_squeeze1.I, conv_15_params_squeeze1.J, conv_15_params_squeeze1.K, conv_15_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_14_out_squeeze1, (elem_t*)conv_15_w_squeeze1, (acc_t*)conv_15_b_squeeze1, (elem_t*)conv_16_out_squeeze1,
        RELU, conv_15_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_16
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_16_params_squeeze1.batch_size, 13, 48,
        192, 13,
        1, 1, 1, 3, conv_16_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_14_out_squeeze1, (elem_t*)conv_16_w_squeeze1, (acc_t*)conv_16_b_squeeze1, (elem_t*)conv_16_out_squeeze1 + 192,

        RELU, conv_16_params_squeeze1.output_scale, 0,
        conv_16_params_squeeze1.pool_size, 0, conv_16_params_squeeze1.pool_padding, false,

        num_array, cid);

#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_17
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_17_params_squeeze1.I, conv_17_params_squeeze1.J, conv_17_params_squeeze1.K, conv_17_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_16_out_squeeze1, (elem_t*)conv_17_w_squeeze1, (acc_t*)conv_17_b_squeeze1, (elem_t*)conv_17_out_squeeze1,
        RELU, conv_17_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_18
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_18_params_squeeze1.I, conv_18_params_squeeze1.J, conv_18_params_squeeze1.K, conv_18_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_17_out_squeeze1, (elem_t*)conv_18_w_squeeze1, (acc_t*)conv_18_b_squeeze1, (elem_t*)conv_19_out_squeeze1,
        RELU, conv_18_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_19
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_19_params_squeeze1.batch_size, conv_19_params_squeeze1.in_dim, conv_19_params_squeeze1.in_channels,
        conv_19_params_squeeze1.out_channels, conv_19_params_squeeze1.out_dim,
        conv_19_params_squeeze1.stride, 1, conv_19_params_squeeze1.padding, conv_19_params_squeeze1.kernel_size, conv_19_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_17_out_squeeze1, (elem_t*)conv_19_w_squeeze1, (acc_t*)conv_19_b_squeeze1, (elem_t*)conv_19_out_squeeze1 + conv_19_params_squeeze1.out_channels,

        RELU, conv_19_params_squeeze1.output_scale, 0,
        conv_19_params_squeeze1.pool_size, 0, conv_19_params_squeeze1.pool_padding, false,

        num_array, cid);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  



    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_20
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_20_params_squeeze1.I, conv_20_params_squeeze1.J, conv_20_params_squeeze1.K, conv_20_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_19_out_squeeze1, (elem_t*)conv_20_w_squeeze1, (acc_t*)conv_20_b_squeeze1, (elem_t*)conv_20_out_squeeze1,
        RELU, conv_20_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_21
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_21_params_squeeze1.I, conv_21_params_squeeze1.J, conv_21_params_squeeze1.K, conv_21_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_20_out_squeeze1, (elem_t*)conv_21_w_squeeze1, (acc_t*)conv_21_b_squeeze1, (elem_t*)conv_22_out_squeeze1,
        RELU, conv_21_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_22
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_22_params_squeeze1.batch_size, conv_22_params_squeeze1.in_dim, conv_22_params_squeeze1.in_channels,
        conv_22_params_squeeze1.out_channels, conv_22_params_squeeze1.out_dim,
        conv_22_params_squeeze1.stride, 1, conv_22_params_squeeze1.padding, conv_22_params_squeeze1.kernel_size, conv_22_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_20_out_squeeze1, (elem_t*)conv_22_w_squeeze1, (acc_t*)conv_22_b_squeeze1, (elem_t*)conv_22_out_squeeze1 + conv_22_params_squeeze1.out_channels,

        RELU, conv_22_params_squeeze1.output_scale, 0,
        conv_22_params_squeeze1.pool_size, 0, conv_22_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_23
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_23_params_squeeze1.I, conv_23_params_squeeze1.J, conv_23_params_squeeze1.K, conv_23_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_22_out_squeeze1, (elem_t*)conv_23_w_squeeze1, (acc_t*)conv_23_b_squeeze1, (elem_t*)conv_23_out_squeeze1,
        RELU, conv_23_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_24
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_24_params_squeeze1.I, conv_24_params_squeeze1.J, conv_24_params_squeeze1.K, conv_24_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_23_out_squeeze1, (elem_t*)conv_24_w_squeeze1, (acc_t*)conv_24_b_squeeze1, (elem_t*)conv_25_out_squeeze1,
        RELU, conv_24_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_25
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_25_params_squeeze1.batch_size, conv_25_params_squeeze1.in_dim, conv_25_params_squeeze1.in_channels,
        conv_25_params_squeeze1.out_channels, conv_25_params_squeeze1.out_dim,
        conv_25_params_squeeze1.stride, 1, conv_25_params_squeeze1.padding, conv_25_params_squeeze1.kernel_size, conv_25_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_23_out_squeeze1, (elem_t*)conv_25_w_squeeze1, (acc_t*)conv_25_b_squeeze1, (elem_t*)conv_25_out_squeeze1 + conv_25_params_squeeze1.out_channels,

        RELU, conv_25_params_squeeze1.output_scale, 0,
        conv_25_params_squeeze1.pool_size, 0, conv_25_params_squeeze1.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_26
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_26_params_squeeze1.I, conv_26_params_squeeze1.J, conv_26_params_squeeze1.K, conv_26_params_squeeze1.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_25_out_squeeze1, (elem_t*)conv_26_w_squeeze1, (acc_t*)conv_26_b_squeeze1, (elem_t*)conv_26_out_squeeze1,
        RELU, conv_26_params_squeeze1.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
  

    for(int i = 0; i < num_cycle; i++){
      if(i < 26){
        cycles[i] = conv_cycles[i];
      }
      else if (i < 27){
        cycles[i] = pool_cycles[i - 26];
      }
      else{
        if(i == 27) cycles[i] = total_conv_cycles;
        if(i == 28) cycles[i] = total_pool_cycles;
        if(i == 29) cycles[i] = total_conv_cycles + total_pool_cycles;
      }
    }
    return cycles;
#undef num_cycle
}

uint64_t* squeezenet_function_11(bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
#define num_cycle (26+1+3)
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif        

  static uint64_t cycles[num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[26];
    //uint64_t conv_cycles[15];
    uint64_t pool_cycles[1];
    // conv_1

    start = read_cycles();
    tiled_opcode_conv_default(
        conv_1_params_squeeze11.batch_size, conv_1_params_squeeze11.in_dim, conv_1_params_squeeze11.in_channels,
        conv_1_params_squeeze11.out_channels, conv_1_params_squeeze11.out_dim,
        conv_1_params_squeeze11.stride, 1, conv_1_params_squeeze11.padding, conv_1_params_squeeze11.kernel_size,
        conv_1_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)image00, (elem_t*)conv_1_w_squeeze11, (acc_t*)conv_1_b_squeeze11, (elem_t*)conv_1_out_squeeze11,

        RELU, conv_1_params_squeeze11.output_scale, 0,
        1, 0, 0, false,
	//conv_1_params_squeeze11.pool_size, conv_1_params_squeeze11.pool_stride, conv_1_params_squeeze11.pool_padding, false,
	num_array, cid);
        //WS, 2* orow_divide, batch_divide,  cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
//    printf("conv cycles 0: %llu\n", conv_cycles[0]);

#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif        

/*
    // conv_1
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_1_params_squeeze11.batch_size, conv_1_params_squeeze11.in_dim, conv_1_params_squeeze11.in_channels,
        conv_1_params_squeeze11.out_channels, conv_1_params_squeeze11.out_dim,
        conv_1_params_squeeze11.stride, 1, conv_1_params_squeeze11.padding, conv_1_params_squeeze11.kernel_size,
        conv_1_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)image0, (elem_t*)conv_1_w_squeeze11, (acc_t*)conv_1_b_squeeze11, (elem_t*)conv_1_out_squeeze11,
        RELU, conv_1_params_squeeze11.output_scale, 0,
        1, 1, 0, false,
	//conv_1_params_squeeze11.pool_size, conv_1_params_squeeze11.pool_stride, conv_1_params_squeeze11.pool_padding, false,
        WS, orow_divide * 2, batch_divide,  orow_divide + cid, target_util);
    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif   
*//*
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params_squeeze11.batch_size,
        conv_1_params_squeeze11.out_channels, conv_1_params_squeeze11.out_dim, conv_1_params_squeeze11.out_dim_pooled,
        conv_1_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        conv_1_params_squeeze11.pool_size, conv_1_params_squeeze11.pool_stride, conv_1_params_squeeze11.pool_padding,
        (elem_t*)conv_1_out_squeeze11, (elem_t*)conv_1_out_squeeze11_pooled,
	num_array, cid);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif         
   */     
    // conv_2
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_2_params_squeeze11.I, conv_2_params_squeeze11.J, conv_2_params_squeeze11.K, conv_2_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_1_out_squeeze11_pooled, (elem_t*)conv_2_w_squeeze11, (acc_t*)conv_2_b_squeeze11, (elem_t*)conv_2_out_squeeze11,
        RELU, conv_2_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
 
    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_3_params_squeeze11.I, conv_3_params_squeeze11.J, conv_3_params_squeeze11.K, conv_3_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_2_out_squeeze11, (elem_t*)conv_3_w_squeeze11, (acc_t*)conv_3_b_squeeze11, (elem_t*)conv_4_out_squeeze11,
        RELU, conv_3_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_4
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_4_params_squeeze11.batch_size, conv_4_params_squeeze11.in_dim, conv_4_params_squeeze11.in_channels,
        conv_4_params_squeeze11.out_channels, conv_4_params_squeeze11.out_dim,
        conv_4_params_squeeze11.stride, 1, conv_4_params_squeeze11.padding, conv_4_params_squeeze11.kernel_size, conv_4_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_2_out_squeeze11, (elem_t*)conv_4_w_squeeze11, (acc_t*)conv_4_b_squeeze11, (elem_t*)conv_4_out_squeeze11 + conv_4_params_squeeze11.out_channels,

        RELU, conv_4_params_squeeze11.output_scale, 0,
        conv_4_params_squeeze11.pool_size, 0, conv_4_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_5
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_5_params_squeeze11.I, conv_5_params_squeeze11.J, conv_5_params_squeeze11.K, conv_5_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_4_out_squeeze11, (elem_t*)conv_5_w_squeeze11, (acc_t*)conv_5_b_squeeze11, (elem_t*)conv_5_out_squeeze11,
        RELU, conv_5_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_6
     start = read_cycles();
    tiled_opcode_conv_default(
        conv_6_params_squeeze11.batch_size, conv_6_params_squeeze11.in_dim, conv_6_params_squeeze11.in_channels,
        conv_6_params_squeeze11.out_channels, conv_6_params_squeeze11.out_dim,
        conv_6_params_squeeze11.stride, 1, conv_6_params_squeeze11.padding, conv_6_params_squeeze11.kernel_size, conv_6_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_squeeze11, (elem_t*)conv_6_w_squeeze11, (acc_t*)conv_6_b_squeeze11, (elem_t*)conv_7_out_squeeze11_pooled,

        RELU, conv_6_params_squeeze11.output_scale, 0,
        conv_6_params_squeeze11.pool_size, conv_6_params_squeeze11.pool_stride, conv_6_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_7
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_7_params_squeeze11.batch_size, conv_7_params_squeeze11.in_dim, conv_7_params_squeeze11.in_channels,
        conv_7_params_squeeze11.out_channels, conv_7_params_squeeze11.out_dim,
        conv_7_params_squeeze11.stride, 1, conv_7_params_squeeze11.padding, conv_7_params_squeeze11.kernel_size, conv_7_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_squeeze11, (elem_t*)conv_7_w_squeeze11, (acc_t*)conv_7_b_squeeze11, (elem_t*)conv_7_out_squeeze11_pooled + conv_7_params_squeeze11.out_channels,

        RELU, conv_7_params_squeeze11.output_scale, 0,
        conv_7_params_squeeze11.pool_size, conv_7_params_squeeze11.pool_stride, conv_7_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_8
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_8_params_squeeze11.I, conv_8_params_squeeze11.J, conv_8_params_squeeze11.K, conv_8_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_7_out_squeeze11_pooled, (elem_t*)conv_8_w_squeeze11, (acc_t*)conv_8_b_squeeze11, (elem_t*)conv_8_out_squeeze11,
        RELU, conv_8_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_9_params_squeeze11.I, conv_9_params_squeeze11.J, conv_9_params_squeeze11.K, conv_9_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_8_out_squeeze11, (elem_t*)conv_9_w_squeeze11, (acc_t*)conv_9_b_squeeze11, (elem_t*)conv_10_out_squeeze11,
        RELU, conv_9_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_10
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_10_params_squeeze11.batch_size, conv_10_params_squeeze11.in_dim, conv_10_params_squeeze11.in_channels,
        conv_10_params_squeeze11.out_channels, conv_10_params_squeeze11.out_dim,
        conv_10_params_squeeze11.stride, 1, conv_10_params_squeeze11.padding, conv_10_params_squeeze11.kernel_size, conv_10_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_8_out_squeeze11, (elem_t*)conv_10_w_squeeze11, (acc_t*)conv_10_b_squeeze11, (elem_t*)conv_10_out_squeeze11 + conv_10_params_squeeze11.out_channels,

        RELU, conv_10_params_squeeze11.output_scale, 0,
        conv_10_params_squeeze11.pool_size, 0, conv_10_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_11
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_11_params_squeeze11.I, conv_11_params_squeeze11.J, conv_11_params_squeeze11.K, conv_11_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_10_out_squeeze11, (elem_t*)conv_11_w_squeeze11, (acc_t*)conv_11_b_squeeze11, (elem_t*)conv_11_out_squeeze11,
        RELU, conv_11_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_12
     start = read_cycles();
    tiled_opcode_conv_default(
        conv_12_params_squeeze11.batch_size, conv_12_params_squeeze11.in_dim, conv_12_params_squeeze11.in_channels,
        conv_12_params_squeeze11.out_channels, conv_12_params_squeeze11.out_dim,
        conv_12_params_squeeze11.stride, 1, conv_12_params_squeeze11.padding, conv_12_params_squeeze11.kernel_size, conv_12_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_11_out_squeeze11, (elem_t*)conv_12_w_squeeze11, (acc_t*)conv_12_b_squeeze11, (elem_t*)conv_13_out_squeeze11_pooled,

        RELU, conv_12_params_squeeze11.output_scale, 0,
        conv_12_params_squeeze11.pool_size, conv_12_params_squeeze11.pool_stride, conv_12_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_13
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_13_params_squeeze11.batch_size, conv_13_params_squeeze11.in_dim, conv_13_params_squeeze11.in_channels,
        conv_13_params_squeeze11.out_channels, conv_13_params_squeeze11.out_dim,
        conv_13_params_squeeze11.stride, 1, conv_13_params_squeeze11.padding, conv_13_params_squeeze11.kernel_size, conv_13_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_11_out_squeeze11, (elem_t*)conv_13_w_squeeze11, (acc_t*)conv_13_b_squeeze11, (elem_t*)conv_13_out_squeeze11_pooled + conv_13_params_squeeze11.out_channels,

        RELU, conv_13_params_squeeze11.output_scale, 0,
        conv_13_params_squeeze11.pool_size, conv_13_params_squeeze11.pool_stride, conv_13_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_14
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_14_params_squeeze11.I, conv_14_params_squeeze11.J, conv_14_params_squeeze11.K, conv_14_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_13_out_squeeze11_pooled, (elem_t*)conv_14_w_squeeze11, (acc_t*)conv_14_b_squeeze11, (elem_t*)conv_14_out_squeeze11,
        RELU, conv_14_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_15_params_squeeze11.I, conv_15_params_squeeze11.J, conv_15_params_squeeze11.K, conv_15_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_14_out_squeeze11, (elem_t*)conv_15_w_squeeze11, (acc_t*)conv_15_b_squeeze11, (elem_t*)conv_16_out_squeeze11,
        RELU, conv_15_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_16
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_16_params_squeeze11.batch_size, 13, 48,
        192, 13,
        1, 1, 1, 3, conv_16_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_14_out_squeeze11, (elem_t*)conv_16_w_squeeze11, (acc_t*)conv_16_b_squeeze11, (elem_t*)conv_16_out_squeeze11 + 192,

        RELU, conv_16_params_squeeze11.output_scale, 0,
        conv_16_params_squeeze11.pool_size, 0, conv_16_params_squeeze11.pool_padding, false,

        num_array, cid);

#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_17
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_17_params_squeeze11.I, conv_17_params_squeeze11.J, conv_17_params_squeeze11.K, conv_17_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_16_out_squeeze11, (elem_t*)conv_17_w_squeeze11, (acc_t*)conv_17_b_squeeze11, (elem_t*)conv_17_out_squeeze11,
        RELU, conv_17_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_18
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_18_params_squeeze11.I, conv_18_params_squeeze11.J, conv_18_params_squeeze11.K, conv_18_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_17_out_squeeze11, (elem_t*)conv_18_w_squeeze11, (acc_t*)conv_18_b_squeeze11, (elem_t*)conv_19_out_squeeze11,
        RELU, conv_18_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_19
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_19_params_squeeze11.batch_size, conv_19_params_squeeze11.in_dim, conv_19_params_squeeze11.in_channels,
        conv_19_params_squeeze11.out_channels, conv_19_params_squeeze11.out_dim,
        conv_19_params_squeeze11.stride, 1, conv_19_params_squeeze11.padding, conv_19_params_squeeze11.kernel_size, conv_19_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_17_out_squeeze11, (elem_t*)conv_19_w_squeeze11, (acc_t*)conv_19_b_squeeze11, (elem_t*)conv_19_out_squeeze11 + conv_19_params_squeeze11.out_channels,

        RELU, conv_19_params_squeeze11.output_scale, 0,
        conv_19_params_squeeze11.pool_size, 0, conv_19_params_squeeze11.pool_padding, false,

        num_array, cid);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  



    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_20
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_20_params_squeeze11.I, conv_20_params_squeeze11.J, conv_20_params_squeeze11.K, conv_20_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_19_out_squeeze11, (elem_t*)conv_20_w_squeeze11, (acc_t*)conv_20_b_squeeze11, (elem_t*)conv_20_out_squeeze11,
        RELU, conv_20_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_21
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_21_params_squeeze11.I, conv_21_params_squeeze11.J, conv_21_params_squeeze11.K, conv_21_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_20_out_squeeze11, (elem_t*)conv_21_w_squeeze11, (acc_t*)conv_21_b_squeeze11, (elem_t*)conv_22_out_squeeze11,
        RELU, conv_21_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_22
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_22_params_squeeze11.batch_size, conv_22_params_squeeze11.in_dim, conv_22_params_squeeze11.in_channels,
        conv_22_params_squeeze11.out_channels, conv_22_params_squeeze11.out_dim,
        conv_22_params_squeeze11.stride, 1, conv_22_params_squeeze11.padding, conv_22_params_squeeze11.kernel_size, conv_22_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_20_out_squeeze11, (elem_t*)conv_22_w_squeeze11, (acc_t*)conv_22_b_squeeze11, (elem_t*)conv_22_out_squeeze11 + conv_22_params_squeeze11.out_channels,

        RELU, conv_22_params_squeeze11.output_scale, 0,
        conv_22_params_squeeze11.pool_size, 0, conv_22_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_23
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_23_params_squeeze11.I, conv_23_params_squeeze11.J, conv_23_params_squeeze11.K, conv_23_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_22_out_squeeze11, (elem_t*)conv_23_w_squeeze11, (acc_t*)conv_23_b_squeeze11, (elem_t*)conv_23_out_squeeze11,
        RELU, conv_23_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_24
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_24_params_squeeze11.I, conv_24_params_squeeze11.J, conv_24_params_squeeze11.K, conv_24_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_23_out_squeeze11, (elem_t*)conv_24_w_squeeze11, (acc_t*)conv_24_b_squeeze11, (elem_t*)conv_25_out_squeeze11,
        RELU, conv_24_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_25
     start = read_cycles();

    tiled_opcode_conv_default(
        conv_25_params_squeeze11.batch_size, conv_25_params_squeeze11.in_dim, conv_25_params_squeeze11.in_channels,
        conv_25_params_squeeze11.out_channels, conv_25_params_squeeze11.out_dim,
        conv_25_params_squeeze11.stride, 1, conv_25_params_squeeze11.padding, conv_25_params_squeeze11.kernel_size, conv_25_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_23_out_squeeze11, (elem_t*)conv_25_w_squeeze11, (acc_t*)conv_25_b_squeeze11, (elem_t*)conv_25_out_squeeze11 + conv_25_params_squeeze11.out_channels,

        RELU, conv_25_params_squeeze11.output_scale, 0,
        conv_25_params_squeeze11.pool_size, 0, conv_25_params_squeeze11.pool_padding, false,

        num_array, cid);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_26
    start = read_cycles();
    tiled_opcode_matmul_nn_default(conv_26_params_squeeze11.I, conv_26_params_squeeze11.J, conv_26_params_squeeze11.K, conv_26_params_squeeze11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)conv_25_out_squeeze11, (elem_t*)conv_26_w_squeeze11, (acc_t*)conv_26_b_squeeze11, (elem_t*)conv_26_out_squeeze11,
        RELU, conv_26_params_squeeze11.output_scale, 0, true,
        WS,
        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
  

    for(int i = 0; i < num_cycle; i++){
      if(i < 26){
        cycles[i] = conv_cycles[i];
      }
      else if (i < 27){
        cycles[i] = pool_cycles[i - 26];
      }
      else{
        if(i == 27) cycles[i] = total_conv_cycles;
        if(i == 28) cycles[i] = total_pool_cycles;
        if(i == 29) cycles[i] = total_conv_cycles + total_pool_cycles;
      }
    }
    return cycles;
#undef num_cycle
}

#else

uint64_t* squeezenet_function_1(bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
    
    int round = 2;
    int total_single_time = 3000000;
    dummy_workload(cid, round, total_single_time, num_array);
}
uint64_t* squeezenet_function_11(bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
    
    int round = 2;
    int total_single_time = 3000000;
    dummy_workload(cid, round, total_single_time, num_array);
}
#endif
