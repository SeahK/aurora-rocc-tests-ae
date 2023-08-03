
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#ifndef DEBUG
#include "res18net_params_1.h"
#include "images.h"

#define THREAD_SYNC 0

uint64_t* res18net_function_1(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
#define num_cycle (21+8+3)

  static uint64_t cycles[num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[21];
    uint64_t resadd_cycles[8];
    //uint64_t pool_cycles[3];

    if(block == -1 || block == 0){
    // conv_1
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_1_params_r18es.batch_size, conv_1_params_r18es.in_dim, conv_1_params_r18es.in_channels,
        conv_1_params_r18es.out_channels, conv_1_params_r18es.out_dim,
        conv_1_params_r18es.stride, 1, conv_1_params_r18es.padding, conv_1_params_r18es.kernel_size,
        conv_1_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)image5, (elem_t*)conv_1_w_r18es, (acc_t*)conv_1_b_r18es, (elem_t*)conv_1_out_r18es,

        RELU, conv_1_params_r18es.output_scale, 0,
        1, 1, 0, false,
        //conv_1_params_r18es.pool_size, conv_1_params_r18es.pool_stride, conv_1_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_2
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_2_params_r18es.batch_size, conv_2_params_r18es.in_dim, conv_2_params_r18es.in_channels,
        conv_2_params_r18es.out_channels, conv_2_params_r18es.out_dim,
        conv_2_params_r18es.stride, 1, conv_2_params_r18es.padding, conv_2_params_r18es.kernel_size,
        conv_2_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_1_out_r18es, (elem_t*)conv_2_w_r18es, (acc_t*)conv_2_b_r18es, (elem_t*)conv_2_out_r18es,

        RELU, conv_2_params_r18es.output_scale, 0,
        conv_2_params_r18es.pool_size, 0, conv_2_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_3_params_r18es.batch_size, conv_3_params_r18es.in_dim, conv_3_params_r18es.in_channels,
        conv_3_params_r18es.out_channels, conv_3_params_r18es.out_dim,
        conv_3_params_r18es.stride, 1, conv_3_params_r18es.padding, conv_3_params_r18es.kernel_size,
        conv_3_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_2_out_r18es, (elem_t*)conv_3_w_r18es, (acc_t*)conv_3_b_r18es, (elem_t*)conv_3_out_r18es,

        NO_ACTIVATION, conv_3_params_r18es.output_scale, 0,
        conv_3_params_r18es.pool_size, 0, conv_3_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
     
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_3_params_r18es.I, conv_3_params_r18es.J,
        conv_3_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_3_out_r18es,
        (elem_t*)conv_1_out_r18es,
        (elem_t*)conv_3_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
    
    // conv_4
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_4_params_r18es.batch_size, conv_4_params_r18es.in_dim, conv_4_params_r18es.in_channels,
        conv_4_params_r18es.out_channels, conv_4_params_r18es.out_dim,
        conv_4_params_r18es.stride, 1, conv_4_params_r18es.padding, conv_4_params_r18es.kernel_size,
        conv_4_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_3_out_r18es, (elem_t*)conv_4_w_r18es, (acc_t*)conv_4_b_r18es, (elem_t*)conv_4_out_r18es,

        RELU, conv_4_params_r18es.output_scale, 0,
        conv_4_params_r18es.pool_size, 0, conv_4_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_5
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_5_params_r18es.batch_size, conv_5_params_r18es.in_dim, conv_5_params_r18es.in_channels,
        conv_5_params_r18es.out_channels, conv_5_params_r18es.out_dim,
        conv_5_params_r18es.stride, 1, conv_5_params_r18es.padding, conv_5_params_r18es.kernel_size,
        conv_5_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_4_out_r18es, (elem_t*)conv_5_w_r18es, (acc_t*)conv_5_b_r18es, (elem_t*)conv_5_out_r18es,

        NO_ACTIVATION, conv_5_params_r18es.output_scale, 0,
        conv_5_params_r18es.pool_size, 0, conv_5_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
   
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_5_params_r18es.I, conv_5_params_r18es.J,
        conv_5_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_5_out_r18es,
        (elem_t*)conv_3_out_r18es,
        (elem_t*)conv_5_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
       
    // conv_6
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_6_params_r18es.batch_size, conv_6_params_r18es.in_dim, conv_6_params_r18es.in_channels,
        conv_6_params_r18es.out_channels, conv_6_params_r18es.out_dim,
        conv_6_params_r18es.stride, 1, conv_6_params_r18es.padding, conv_6_params_r18es.kernel_size,
        conv_6_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_r18es, (elem_t*)conv_6_w_r18es, (acc_t*)conv_6_b_r18es, (elem_t*)conv_6_out_r18es,

        RELU, conv_6_params_r18es.output_scale, 0,
        conv_6_params_r18es.pool_size, 0, conv_6_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_7
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_7_params_r18es.batch_size, conv_7_params_r18es.in_dim, conv_7_params_r18es.in_channels,
        conv_7_params_r18es.out_channels, conv_7_params_r18es.out_dim,
        conv_7_params_r18es.stride, 1, conv_7_params_r18es.padding, conv_7_params_r18es.kernel_size,
        conv_7_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_6_out_r18es, (elem_t*)conv_7_w_r18es, (acc_t*)conv_7_b_r18es, (elem_t*)conv_7_out_r18es,

        NO_ACTIVATION, conv_7_params_r18es.output_scale, 0,
        conv_7_params_r18es.pool_size, 0, conv_7_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
    }
    if (block == -1 || block == 1){
        
    // Downsampling conv_5_out_r18es
    // conv_8
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_8_params_r18es.batch_size, conv_8_params_r18es.in_dim, conv_8_params_r18es.in_channels,
        conv_8_params_r18es.out_channels, conv_8_params_r18es.out_dim,
        conv_8_params_r18es.stride, 1, conv_8_params_r18es.padding, conv_8_params_r18es.kernel_size,
        conv_8_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_r18es, (elem_t*)conv_8_w_r18es, (acc_t*)conv_8_b_r18es, (elem_t*)conv_8_out_r18es,

        NO_ACTIVATION, conv_8_params_r18es.output_scale, 0,
        conv_8_params_r18es.pool_size, 0, conv_8_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_7_params_r18es.I, conv_7_params_r18es.J,
        conv_7_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_8_out_r18es,
        (elem_t*)conv_7_out_r18es,
        (elem_t*)conv_8_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_9_params_r18es.batch_size, conv_9_params_r18es.in_dim, conv_9_params_r18es.in_channels,
        conv_9_params_r18es.out_channels, conv_9_params_r18es.out_dim,
        conv_9_params_r18es.stride, 1, conv_9_params_r18es.padding, conv_9_params_r18es.kernel_size,
        conv_9_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_8_out_r18es, (elem_t*)conv_9_w_r18es, (acc_t*)conv_9_b_r18es, (elem_t*)conv_9_out_r18es,

        RELU, conv_9_params_r18es.output_scale, 0,
        conv_9_params_r18es.pool_size, 0, conv_9_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
    // conv_10
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_10_params_r18es.batch_size, conv_10_params_r18es.in_dim, conv_10_params_r18es.in_channels,
        conv_10_params_r18es.out_channels, conv_10_params_r18es.out_dim,
        conv_10_params_r18es.stride, 1, conv_10_params_r18es.padding, conv_10_params_r18es.kernel_size,
        conv_10_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_9_out_r18es, (elem_t*)conv_10_w_r18es, (acc_t*)conv_10_b_r18es, (elem_t*)conv_10_out_r18es,

        NO_ACTIVATION, conv_10_params_r18es.output_scale, 0,
        conv_10_params_r18es.pool_size, 0, conv_10_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_10_params_r18es.I, conv_10_params_r18es.J,
        conv_10_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_10_out_r18es,
        (elem_t*)conv_8_out_r18es,
        (elem_t*)conv_10_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_11
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_11_params_r18es.batch_size, conv_11_params_r18es.in_dim, conv_11_params_r18es.in_channels,
        conv_11_params_r18es.out_channels, conv_11_params_r18es.out_dim,
        conv_11_params_r18es.stride, 1, conv_11_params_r18es.padding, conv_11_params_r18es.kernel_size,
        conv_11_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_10_out_r18es, (elem_t*)conv_11_w_r18es, (acc_t*)conv_11_b_r18es, (elem_t*)conv_11_out_r18es,

        RELU, conv_11_params_r18es.output_scale, 0,
        conv_11_params_r18es.pool_size, 0, conv_11_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_12
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_12_params_r18es.batch_size, conv_12_params_r18es.in_dim, conv_12_params_r18es.in_channels,
        conv_12_params_r18es.out_channels, conv_12_params_r18es.out_dim,
        conv_12_params_r18es.stride, 1, conv_12_params_r18es.padding, conv_12_params_r18es.kernel_size,
        conv_12_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_11_out_r18es, (elem_t*)conv_12_w_r18es, (acc_t*)conv_12_b_r18es, (elem_t*)conv_12_out_r18es,

        NO_ACTIVATION, conv_12_params_r18es.output_scale, 0,
        conv_12_params_r18es.pool_size, 0, conv_12_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Downsampling conv_10_out_r18es
    // conv_13
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_13_params_r18es.batch_size, conv_13_params_r18es.in_dim, conv_13_params_r18es.in_channels,
        conv_13_params_r18es.out_channels, conv_13_params_r18es.out_dim,
        conv_13_params_r18es.stride, 1, conv_13_params_r18es.padding, conv_13_params_r18es.kernel_size,
        conv_13_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_10_out_r18es, (elem_t*)conv_13_w_r18es, (acc_t*)conv_13_b_r18es, (elem_t*)conv_13_out_r18es,

        NO_ACTIVATION, conv_13_params_r18es.output_scale, 0,
        conv_13_params_r18es.pool_size, 0, conv_13_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_12_params_r18es.I, conv_12_params_r18es.J,
        conv_12_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_13_out_r18es,
        (elem_t*)conv_12_out_r18es,
        (elem_t*)conv_13_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_14
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_14_params_r18es.batch_size, conv_14_params_r18es.in_dim, conv_14_params_r18es.in_channels,
        conv_14_params_r18es.out_channels, conv_14_params_r18es.out_dim,
        conv_14_params_r18es.stride, 1, conv_14_params_r18es.padding, conv_14_params_r18es.kernel_size,
        conv_14_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_13_out_r18es, (elem_t*)conv_14_w_r18es, (acc_t*)conv_14_b_r18es, (elem_t*)conv_14_out_r18es,

        RELU, conv_14_params_r18es.output_scale, 0,
        conv_14_params_r18es.pool_size, 0, conv_14_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_15_params_r18es.batch_size, conv_15_params_r18es.in_dim, conv_15_params_r18es.in_channels,
        conv_15_params_r18es.out_channels, conv_15_params_r18es.out_dim,
        conv_15_params_r18es.stride, 1, conv_15_params_r18es.padding, conv_15_params_r18es.kernel_size,
        conv_15_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_14_out_r18es, (elem_t*)conv_15_w_r18es, (acc_t*)conv_15_b_r18es, (elem_t*)conv_15_out_r18es,

        NO_ACTIVATION, conv_15_params_r18es.output_scale, 0,
        conv_15_params_r18es.pool_size, 0, conv_15_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_15_params_r18es.I, conv_15_params_r18es.J,
        conv_15_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_15_out_r18es,
        (elem_t*)conv_13_out_r18es,
        (elem_t*)conv_15_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
    }
    if (block == -1 || block == 2){
        
    // conv_16
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_16_params_r18es.batch_size, conv_16_params_r18es.in_dim, conv_16_params_r18es.in_channels,
        conv_16_params_r18es.out_channels, conv_16_params_r18es.out_dim,
        conv_16_params_r18es.stride, 1, conv_16_params_r18es.padding, conv_16_params_r18es.kernel_size,
        conv_16_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_15_out_r18es, (elem_t*)conv_16_w_r18es, (acc_t*)conv_16_b_r18es, (elem_t*)conv_16_out_r18es,

        RELU, conv_16_params_r18es.output_scale, 0,
        conv_16_params_r18es.pool_size, 0, conv_16_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_17
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_17_params_r18es.batch_size, conv_17_params_r18es.in_dim, conv_17_params_r18es.in_channels,
        conv_17_params_r18es.out_channels, conv_17_params_r18es.out_dim,
        conv_17_params_r18es.stride, 1, conv_17_params_r18es.padding, conv_17_params_r18es.kernel_size,
        conv_17_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_16_out_r18es, (elem_t*)conv_17_w_r18es, (acc_t*)conv_17_b_r18es, (elem_t*)conv_17_out_r18es,

        NO_ACTIVATION, conv_17_params_r18es.output_scale, 0,
        conv_17_params_r18es.pool_size, 0, conv_17_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Downsampling conv_15_out_r18es
    // conv_18
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_18_params_r18es.batch_size, conv_18_params_r18es.in_dim, conv_18_params_r18es.in_channels,
        conv_18_params_r18es.out_channels, conv_18_params_r18es.out_dim,
        conv_18_params_r18es.stride, 1, conv_18_params_r18es.padding, conv_18_params_r18es.kernel_size,
        conv_18_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_15_out_r18es, (elem_t*)conv_18_w_r18es, (acc_t*)conv_18_b_r18es, (elem_t*)conv_18_out_r18es,

        NO_ACTIVATION, conv_18_params_r18es.output_scale, 0,
        conv_18_params_r18es.pool_size, 0, conv_18_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_17_params_r18es.I, conv_17_params_r18es.J,
        conv_17_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_18_out_r18es,
        (elem_t*)conv_17_out_r18es,
        (elem_t*)conv_18_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_19
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_19_params_r18es.batch_size, conv_19_params_r18es.in_dim, conv_19_params_r18es.in_channels,
        conv_19_params_r18es.out_channels, conv_19_params_r18es.out_dim,
        conv_19_params_r18es.stride, 1, conv_19_params_r18es.padding, conv_19_params_r18es.kernel_size,
        conv_19_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_18_out_r18es, (elem_t*)conv_19_w_r18es, (acc_t*)conv_19_b_r18es, (elem_t*)conv_19_out_r18es,

        RELU, conv_19_params_r18es.output_scale, 0,
        conv_19_params_r18es.pool_size, 0, conv_19_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // conv_20
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_20_params_r18es.batch_size, conv_20_params_r18es.in_dim, conv_20_params_r18es.in_channels,
        conv_20_params_r18es.out_channels, conv_20_params_r18es.out_dim,
        conv_20_params_r18es.stride, 1, conv_20_params_r18es.padding, conv_20_params_r18es.kernel_size,
        conv_20_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_19_out_r18es, (elem_t*)conv_20_w_r18es, (acc_t*)conv_20_b_r18es, (elem_t*)conv_20_out_r18es,

        NO_ACTIVATION, conv_20_params_r18es.output_scale, 0,
        conv_20_params_r18es.pool_size, 0, conv_20_params_r18es.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_20_params_r18es.I, conv_20_params_r18es.J,
        conv_20_params_r18es.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_20_out_r18es,
        (elem_t*)conv_18_out_r18es,
        (elem_t*)conv_20_out_r18es,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
        
    // Global averaging
    
    static elem_t average[1][512] row_align(MAX_BLOCK_LEN);
/*
    start = read_cycles();
    if(cid == 0)
        tiled_global_average_auto(conv_20_out_r18es, average, conv_20_params_r18es.batch_size,
            conv_20_params_r18es.out_channels, conv_20_params_r18es.out_dim, WS);
       

    end = read_cycles();
    other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif
*/
    // fc_21
    start = read_cycles();

    tiled_opcode_matmul_nn_default(fc_21_params_r18es.I, fc_21_params_r18es.J, fc_21_params_r18es.K, fc_21_params_r18es.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)average, (elem_t*)fc_21_w_r18es, (acc_t*)fc_21_b_r18es, (elem_t*)fc_21_out_r18es,
        NO_ACTIVATION, fc_21_params_r18es.output_scale, 0, false, WS, num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es);
#endif

    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 21){
        cycles[i] = conv_cycles[i];
      }
      else if (i < 29){
        cycles[i] = resadd_cycles[i - 21];
      }
      else{
        if(i == 29) cycles[i] = total_conv_cycles;
        if(i == 30) cycles[i] = total_resadd_cycles;
        if(i == 31) cycles[i] = total_conv_cycles + total_resadd_cycles;
      }
    }

    return cycles;
#undef num_cycle
}

uint64_t* res18net_function_11(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
#define num_cycle (21+8+3)

  static uint64_t cycles[num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_resadd_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[21];
    uint64_t resadd_cycles[8];
    //uint64_t pool_cycles[3];

    if(block == -1 || block == 0){
    // conv_1
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_1_params_r18es11.batch_size, conv_1_params_r18es11.in_dim, conv_1_params_r18es11.in_channels,
        conv_1_params_r18es11.out_channels, conv_1_params_r18es11.out_dim,
        conv_1_params_r18es11.stride, 1, conv_1_params_r18es11.padding, conv_1_params_r18es11.kernel_size,
        conv_1_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)image55, (elem_t*)conv_1_w_r18es11, (acc_t*)conv_1_b_r18es11, (elem_t*)conv_1_out_r18es11,

        RELU, conv_1_params_r18es11.output_scale, 0,
        1, 1, 0, false,
        //conv_1_params_r18es11.pool_size, conv_1_params_r18es11.pool_stride, conv_1_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_2
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_2_params_r18es11.batch_size, conv_2_params_r18es11.in_dim, conv_2_params_r18es11.in_channels,
        conv_2_params_r18es11.out_channels, conv_2_params_r18es11.out_dim,
        conv_2_params_r18es11.stride, 1, conv_2_params_r18es11.padding, conv_2_params_r18es11.kernel_size,
        conv_2_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_1_out_r18es11, (elem_t*)conv_2_w_r18es11, (acc_t*)conv_2_b_r18es11, (elem_t*)conv_2_out_r18es11,

        RELU, conv_2_params_r18es11.output_scale, 0,
        conv_2_params_r18es11.pool_size, 0, conv_2_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_3_params_r18es11.batch_size, conv_3_params_r18es11.in_dim, conv_3_params_r18es11.in_channels,
        conv_3_params_r18es11.out_channels, conv_3_params_r18es11.out_dim,
        conv_3_params_r18es11.stride, 1, conv_3_params_r18es11.padding, conv_3_params_r18es11.kernel_size,
        conv_3_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_2_out_r18es11, (elem_t*)conv_3_w_r18es11, (acc_t*)conv_3_b_r18es11, (elem_t*)conv_3_out_r18es11,

        NO_ACTIVATION, conv_3_params_r18es11.output_scale, 0,
        conv_3_params_r18es11.pool_size, 0, conv_3_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
     
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_3_params_r18es11.I, conv_3_params_r18es11.J,
        conv_3_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_3_out_r18es11,
        (elem_t*)conv_1_out_r18es11,
        (elem_t*)conv_3_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
    
    // conv_4
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_4_params_r18es11.batch_size, conv_4_params_r18es11.in_dim, conv_4_params_r18es11.in_channels,
        conv_4_params_r18es11.out_channels, conv_4_params_r18es11.out_dim,
        conv_4_params_r18es11.stride, 1, conv_4_params_r18es11.padding, conv_4_params_r18es11.kernel_size,
        conv_4_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_3_out_r18es11, (elem_t*)conv_4_w_r18es11, (acc_t*)conv_4_b_r18es11, (elem_t*)conv_4_out_r18es11,

        RELU, conv_4_params_r18es11.output_scale, 0,
        conv_4_params_r18es11.pool_size, 0, conv_4_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_5
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_5_params_r18es11.batch_size, conv_5_params_r18es11.in_dim, conv_5_params_r18es11.in_channels,
        conv_5_params_r18es11.out_channels, conv_5_params_r18es11.out_dim,
        conv_5_params_r18es11.stride, 1, conv_5_params_r18es11.padding, conv_5_params_r18es11.kernel_size,
        conv_5_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_4_out_r18es11, (elem_t*)conv_5_w_r18es11, (acc_t*)conv_5_b_r18es11, (elem_t*)conv_5_out_r18es11,

        NO_ACTIVATION, conv_5_params_r18es11.output_scale, 0,
        conv_5_params_r18es11.pool_size, 0, conv_5_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
   
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_5_params_r18es11.I, conv_5_params_r18es11.J,
        conv_5_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_5_out_r18es11,
        (elem_t*)conv_3_out_r18es11,
        (elem_t*)conv_5_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
       
    // conv_6
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_6_params_r18es11.batch_size, conv_6_params_r18es11.in_dim, conv_6_params_r18es11.in_channels,
        conv_6_params_r18es11.out_channels, conv_6_params_r18es11.out_dim,
        conv_6_params_r18es11.stride, 1, conv_6_params_r18es11.padding, conv_6_params_r18es11.kernel_size,
        conv_6_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_r18es11, (elem_t*)conv_6_w_r18es11, (acc_t*)conv_6_b_r18es11, (elem_t*)conv_6_out_r18es11,

        RELU, conv_6_params_r18es11.output_scale, 0,
        conv_6_params_r18es11.pool_size, 0, conv_6_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_7
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_7_params_r18es11.batch_size, conv_7_params_r18es11.in_dim, conv_7_params_r18es11.in_channels,
        conv_7_params_r18es11.out_channels, conv_7_params_r18es11.out_dim,
        conv_7_params_r18es11.stride, 1, conv_7_params_r18es11.padding, conv_7_params_r18es11.kernel_size,
        conv_7_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_6_out_r18es11, (elem_t*)conv_7_w_r18es11, (acc_t*)conv_7_b_r18es11, (elem_t*)conv_7_out_r18es11,

        NO_ACTIVATION, conv_7_params_r18es11.output_scale, 0,
        conv_7_params_r18es11.pool_size, 0, conv_7_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    }
    if(block == -1 || block == 1) {
    // Downsampling conv_5_out_r18es11
    // conv_8
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_8_params_r18es11.batch_size, conv_8_params_r18es11.in_dim, conv_8_params_r18es11.in_channels,
        conv_8_params_r18es11.out_channels, conv_8_params_r18es11.out_dim,
        conv_8_params_r18es11.stride, 1, conv_8_params_r18es11.padding, conv_8_params_r18es11.kernel_size,
        conv_8_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_5_out_r18es11, (elem_t*)conv_8_w_r18es11, (acc_t*)conv_8_b_r18es11, (elem_t*)conv_8_out_r18es11,

        NO_ACTIVATION, conv_8_params_r18es11.output_scale, 0,
        conv_8_params_r18es11.pool_size, 0, conv_8_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_7_params_r18es11.I, conv_7_params_r18es11.J,
        conv_7_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_8_out_r18es11,
        (elem_t*)conv_7_out_r18es11,
        (elem_t*)conv_8_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_9_params_r18es11.batch_size, conv_9_params_r18es11.in_dim, conv_9_params_r18es11.in_channels,
        conv_9_params_r18es11.out_channels, conv_9_params_r18es11.out_dim,
        conv_9_params_r18es11.stride, 1, conv_9_params_r18es11.padding, conv_9_params_r18es11.kernel_size,
        conv_9_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_8_out_r18es11, (elem_t*)conv_9_w_r18es11, (acc_t*)conv_9_b_r18es11, (elem_t*)conv_9_out_r18es11,

        RELU, conv_9_params_r18es11.output_scale, 0,
        conv_9_params_r18es11.pool_size, 0, conv_9_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
    // conv_10
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_10_params_r18es11.batch_size, conv_10_params_r18es11.in_dim, conv_10_params_r18es11.in_channels,
        conv_10_params_r18es11.out_channels, conv_10_params_r18es11.out_dim,
        conv_10_params_r18es11.stride, 1, conv_10_params_r18es11.padding, conv_10_params_r18es11.kernel_size,
        conv_10_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_9_out_r18es11, (elem_t*)conv_10_w_r18es11, (acc_t*)conv_10_b_r18es11, (elem_t*)conv_10_out_r18es11,

        NO_ACTIVATION, conv_10_params_r18es11.output_scale, 0,
        conv_10_params_r18es11.pool_size, 0, conv_10_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_10_params_r18es11.I, conv_10_params_r18es11.J,
        conv_10_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_10_out_r18es11,
        (elem_t*)conv_8_out_r18es11,
        (elem_t*)conv_10_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_11
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_11_params_r18es11.batch_size, conv_11_params_r18es11.in_dim, conv_11_params_r18es11.in_channels,
        conv_11_params_r18es11.out_channels, conv_11_params_r18es11.out_dim,
        conv_11_params_r18es11.stride, 1, conv_11_params_r18es11.padding, conv_11_params_r18es11.kernel_size,
        conv_11_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_10_out_r18es11, (elem_t*)conv_11_w_r18es11, (acc_t*)conv_11_b_r18es11, (elem_t*)conv_11_out_r18es11,

        RELU, conv_11_params_r18es11.output_scale, 0,
        conv_11_params_r18es11.pool_size, 0, conv_11_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_12
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_12_params_r18es11.batch_size, conv_12_params_r18es11.in_dim, conv_12_params_r18es11.in_channels,
        conv_12_params_r18es11.out_channels, conv_12_params_r18es11.out_dim,
        conv_12_params_r18es11.stride, 1, conv_12_params_r18es11.padding, conv_12_params_r18es11.kernel_size,
        conv_12_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_11_out_r18es11, (elem_t*)conv_12_w_r18es11, (acc_t*)conv_12_b_r18es11, (elem_t*)conv_12_out_r18es11,

        NO_ACTIVATION, conv_12_params_r18es11.output_scale, 0,
        conv_12_params_r18es11.pool_size, 0, conv_12_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Downsampling conv_10_out_r18es11
    // conv_13
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_13_params_r18es11.batch_size, conv_13_params_r18es11.in_dim, conv_13_params_r18es11.in_channels,
        conv_13_params_r18es11.out_channels, conv_13_params_r18es11.out_dim,
        conv_13_params_r18es11.stride, 1, conv_13_params_r18es11.padding, conv_13_params_r18es11.kernel_size,
        conv_13_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_10_out_r18es11, (elem_t*)conv_13_w_r18es11, (acc_t*)conv_13_b_r18es11, (elem_t*)conv_13_out_r18es11,

        NO_ACTIVATION, conv_13_params_r18es11.output_scale, 0,
        conv_13_params_r18es11.pool_size, 0, conv_13_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_12_params_r18es11.I, conv_12_params_r18es11.J,
        conv_12_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_13_out_r18es11,
        (elem_t*)conv_12_out_r18es11,
        (elem_t*)conv_13_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_14
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_14_params_r18es11.batch_size, conv_14_params_r18es11.in_dim, conv_14_params_r18es11.in_channels,
        conv_14_params_r18es11.out_channels, conv_14_params_r18es11.out_dim,
        conv_14_params_r18es11.stride, 1, conv_14_params_r18es11.padding, conv_14_params_r18es11.kernel_size,
        conv_14_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_13_out_r18es11, (elem_t*)conv_14_w_r18es11, (acc_t*)conv_14_b_r18es11, (elem_t*)conv_14_out_r18es11,

        RELU, conv_14_params_r18es11.output_scale, 0,
        conv_14_params_r18es11.pool_size, 0, conv_14_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_15_params_r18es11.batch_size, conv_15_params_r18es11.in_dim, conv_15_params_r18es11.in_channels,
        conv_15_params_r18es11.out_channels, conv_15_params_r18es11.out_dim,
        conv_15_params_r18es11.stride, 1, conv_15_params_r18es11.padding, conv_15_params_r18es11.kernel_size,
        conv_15_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_14_out_r18es11, (elem_t*)conv_15_w_r18es11, (acc_t*)conv_15_b_r18es11, (elem_t*)conv_15_out_r18es11,

        NO_ACTIVATION, conv_15_params_r18es11.output_scale, 0,
        conv_15_params_r18es11.pool_size, 0, conv_15_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_15_params_r18es11.I, conv_15_params_r18es11.J,
        conv_15_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_15_out_r18es11,
        (elem_t*)conv_13_out_r18es11,
        (elem_t*)conv_15_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    }
    if (block == -1 || block == 2){
    // conv_16
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_16_params_r18es11.batch_size, conv_16_params_r18es11.in_dim, conv_16_params_r18es11.in_channels,
        conv_16_params_r18es11.out_channels, conv_16_params_r18es11.out_dim,
        conv_16_params_r18es11.stride, 1, conv_16_params_r18es11.padding, conv_16_params_r18es11.kernel_size,
        conv_16_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_15_out_r18es11, (elem_t*)conv_16_w_r18es11, (acc_t*)conv_16_b_r18es11, (elem_t*)conv_16_out_r18es11,

        RELU, conv_16_params_r18es11.output_scale, 0,
        conv_16_params_r18es11.pool_size, 0, conv_16_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_17
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_17_params_r18es11.batch_size, conv_17_params_r18es11.in_dim, conv_17_params_r18es11.in_channels,
        conv_17_params_r18es11.out_channels, conv_17_params_r18es11.out_dim,
        conv_17_params_r18es11.stride, 1, conv_17_params_r18es11.padding, conv_17_params_r18es11.kernel_size,
        conv_17_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_16_out_r18es11, (elem_t*)conv_17_w_r18es11, (acc_t*)conv_17_b_r18es11, (elem_t*)conv_17_out_r18es11,

        NO_ACTIVATION, conv_17_params_r18es11.output_scale, 0,
        conv_17_params_r18es11.pool_size, 0, conv_17_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Downsampling conv_15_out_r18es11
    // conv_18
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_18_params_r18es11.batch_size, conv_18_params_r18es11.in_dim, conv_18_params_r18es11.in_channels,
        conv_18_params_r18es11.out_channels, conv_18_params_r18es11.out_dim,
        conv_18_params_r18es11.stride, 1, conv_18_params_r18es11.padding, conv_18_params_r18es11.kernel_size,
        conv_18_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_15_out_r18es11, (elem_t*)conv_18_w_r18es11, (acc_t*)conv_18_b_r18es11, (elem_t*)conv_18_out_r18es11,

        NO_ACTIVATION, conv_18_params_r18es11.output_scale, 0,
        conv_18_params_r18es11.pool_size, 0, conv_18_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_17_params_r18es11.I, conv_17_params_r18es11.J,
        conv_17_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_18_out_r18es11,
        (elem_t*)conv_17_out_r18es11,
        (elem_t*)conv_18_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_19
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_19_params_r18es11.batch_size, conv_19_params_r18es11.in_dim, conv_19_params_r18es11.in_channels,
        conv_19_params_r18es11.out_channels, conv_19_params_r18es11.out_dim,
        conv_19_params_r18es11.stride, 1, conv_19_params_r18es11.padding, conv_19_params_r18es11.kernel_size,
        conv_19_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_18_out_r18es11, (elem_t*)conv_19_w_r18es11, (acc_t*)conv_19_b_r18es11, (elem_t*)conv_19_out_r18es11,

        RELU, conv_19_params_r18es11.output_scale, 0,
        conv_19_params_r18es11.pool_size, 0, conv_19_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // conv_20
    start = read_cycles();
    tiled_opcode_conv_default(
        conv_20_params_r18es11.batch_size, conv_20_params_r18es11.in_dim, conv_20_params_r18es11.in_channels,
        conv_20_params_r18es11.out_channels, conv_20_params_r18es11.out_dim,
        conv_20_params_r18es11.stride, 1, conv_20_params_r18es11.padding, conv_20_params_r18es11.kernel_size,
        conv_20_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

        (elem_t*)conv_19_out_r18es11, (elem_t*)conv_20_w_r18es11, (acc_t*)conv_20_b_r18es11, (elem_t*)conv_20_out_r18es11,

        NO_ACTIVATION, conv_20_params_r18es11.output_scale, 0,
        conv_20_params_r18es11.pool_size, 0, conv_20_params_r18es11.pool_padding, false,

        num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_opcode_resadd_default(conv_20_params_r18es11.I, conv_20_params_r18es11.J,
        conv_20_params_r18es11.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, output_direct_dram, output_direct_dram, output_direct_dram,
        (elem_t*)conv_20_out_r18es11,
        (elem_t*)conv_18_out_r18es11,
        (elem_t*)conv_20_out_r18es11,
        true,
        num_array, cid);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
        
    // Global averaging
    
    static elem_t average[1][512] row_align(MAX_BLOCK_LEN);
/*
    start = read_cycles();
    if(cid == 0)
        tiled_global_average_auto(conv_20_out_r18es11, average, conv_20_params_r18es11.batch_size,
            conv_20_params_r18es11.out_channels, conv_20_params_r18es11.out_dim, WS);
       

    end = read_cycles();
    other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
*/
    // fc_21
    start = read_cycles();

    tiled_opcode_matmul_nn_default(fc_21_params_r18es11.I, fc_21_params_r18es11.J, fc_21_params_r18es11.K, fc_21_params_r18es11.out_stride, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
        (elem_t*)average, (elem_t*)fc_21_w_r18es11, (acc_t*)fc_21_b_r18es11, (elem_t*)fc_21_out_r18es11,
        NO_ACTIVATION, fc_21_params_r18es11.output_scale, 0, false, WS, num_array, cid);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_r18es11);
#endif
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 21){
        cycles[i] = conv_cycles[i];
      }
      else if (i < 29){
        cycles[i] = resadd_cycles[i - 21];
      }
      else{
        if(i == 29) cycles[i] = total_conv_cycles;
        if(i == 30) cycles[i] = total_resadd_cycles;
        if(i == 31) cycles[i] = total_conv_cycles + total_resadd_cycles;
      }
    }

    return cycles;
#undef num_cycle
}

#else
uint64_t* res18net_function_1(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
    int round = 5;
    int total_single_time = 10000000;
    if (block >= 0){
        round = 1;
        total_single_time /= 2;
    }
    dummy_workload(cid, round, total_single_time, num_array);
}

uint64_t* res18net_function_11(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
    int round = 5;
    int total_single_time = 10000000;
    if (block >= 0){
        round = 1;
        total_single_time /= 2;
    }
    dummy_workload(cid, round, total_single_time, num_array);
}
#endif
