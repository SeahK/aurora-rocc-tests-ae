
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#ifndef DEBUG
#include "alexnet_params_1.h"
#include "images.h"

#define THREAD_SYNC 0

uint64_t* alexnet_function_1(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
#define num_cycle (5+3+3+4)

  static uint64_t cycles[num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[5];
    uint64_t matmul_cycles[3];
    uint64_t pool_cycles[3];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif
    if(block == -1 || block == 0){
      // conv_1
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_1_params_alex1.batch_size, conv_1_params_alex1.in_dim, conv_1_params_alex1.in_channels,
          conv_1_params_alex1.out_channels, conv_1_params_alex1.out_dim,
          conv_1_params_alex1.stride, 1, conv_1_params_alex1.padding, conv_1_params_alex1.kernel_size,
          conv_1_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          
	  (elem_t*)image4, (elem_t*)conv_1_w_alex1, (acc_t*)conv_1_b_alex1, (elem_t*)conv_1_out_alex1,

          RELU, conv_1_params_alex1.output_scale, 0,
          1, 1, 0, false,
    //conv_1_params_alex1.pool_size, conv_1_params_alex1.pool_stride, conv_1_params_alex1.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif 
//printf("before pool 1\n");       
//if(cid == 0 || cid == 1)  
/* 
      start = read_cycles();
      tiled_opcode_pool_auto_multi(
          conv_1_params_alex1.batch_size,
          conv_1_params_alex1.out_channels, conv_1_params_alex1.out_dim, conv_1_params_alex1.out_dim_pooled,
          conv_1_params_alex1.out_stride,
          conv_1_params_alex1.pool_size, conv_1_params_alex1.pool_stride, conv_1_params_alex1.pool_padding,

          (elem_t*)conv_1_out_alex1, (elem_t*)conv_1_out_alex1_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          */
      // conv_2
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_2_params_alex1.batch_size, conv_2_params_alex1.in_dim, conv_2_params_alex1.in_channels,
          conv_2_params_alex1.out_channels, conv_2_params_alex1.out_dim,
          conv_2_params_alex1.stride, 1, conv_2_params_alex1.padding, conv_2_params_alex1.kernel_size,
          conv_2_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_1_out_alex1_pooled, (elem_t*)conv_2_w_alex1, (acc_t*)conv_2_b_alex1, (elem_t*)conv_2_out_alex1,

          RELU, conv_2_params_alex1.output_scale, 0,
          1, 1, 0, false,
    //conv_2_params_alex1.pool_size, conv_2_params_alex1.pool_stride, conv_2_params_alex1.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
    
//printf("before pool 2\n");       
//if(cid == 0 || cid == 1) 
/* 
      start = read_cycles();
      tiled_opcode_pool_auto_multi(
          conv_2_params_alex1.batch_size,
          conv_2_params_alex1.out_channels, conv_2_params_alex1.out_dim, conv_2_params_alex1.out_dim_pooled,
          conv_2_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          conv_2_params_alex1.pool_size, conv_2_params_alex1.pool_stride, conv_2_params_alex1.pool_padding,

          (elem_t*)conv_2_out_alex1, (elem_t*)conv_2_out_alex1_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += (end > start ? end - start : 0);
      pool_cycles[1] = (end > start ? end - start : 0);
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif          
*/
      // conv_3
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_3_params_alex1.batch_size, conv_3_params_alex1.in_dim, conv_3_params_alex1.in_channels,
          conv_3_params_alex1.out_channels, conv_3_params_alex1.out_dim,
          conv_3_params_alex1.stride, 1, conv_3_params_alex1.padding, conv_3_params_alex1.kernel_size,
          conv_3_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_2_out_alex1_pooled, (elem_t*)conv_3_w_alex1, (acc_t*)conv_3_b_alex1, (elem_t*)conv_3_out_alex1,

          RELU, conv_3_params_alex1.output_scale, 0,
          conv_3_params_alex1.pool_size, 0, conv_3_params_alex1.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_4
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_4_params_alex1.batch_size, conv_4_params_alex1.in_dim, conv_4_params_alex1.in_channels,
          conv_4_params_alex1.out_channels, conv_4_params_alex1.out_dim,
          conv_4_params_alex1.stride, 1, conv_4_params_alex1.padding, conv_4_params_alex1.kernel_size,
          conv_4_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_3_out_alex1, (elem_t*)conv_4_w_alex1, (acc_t*)conv_4_b_alex1, (elem_t*)conv_4_out_alex1,

          RELU, conv_4_params_alex1.output_scale, 0,
          conv_4_params_alex1.pool_size, 0, conv_4_params_alex1.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_5
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_5_params_alex1.batch_size, conv_5_params_alex1.in_dim, conv_5_params_alex1.in_channels,
          conv_5_params_alex1.out_channels, conv_5_params_alex1.out_dim,
          conv_5_params_alex1.stride, 1, conv_5_params_alex1.padding, conv_5_params_alex1.kernel_size,
          conv_5_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_4_out_alex1, (elem_t*)conv_5_w_alex1, (acc_t*)conv_5_b_alex1, (elem_t*)conv_5_out_alex1,

          RELU, conv_5_params_alex1.output_scale, 0,
          1, 1, 0, false,
    //conv_5_params_alex1.pool_size, conv_5_params_alex1.pool_stride, conv_5_params_alex1.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    if(block == -1 || block == 1){

//printf("before pool 3\n");       
//if(cid == 0 || cid == 1)  
/*
      start = read_cycles();
      tiled_opcode_pool_auto_multi(
          conv_5_params_alex1.batch_size,
          conv_5_params_alex1.out_channels, conv_5_params_alex1.out_dim, conv_5_params_alex1.out_dim_pooled,
          conv_5_params_alex1.out_stride,
          conv_5_params_alex1.pool_size, conv_5_params_alex1.pool_stride, conv_5_params_alex1.pool_padding,

          (elem_t*)conv_5_out_alex1, (elem_t*)conv_5_out_alex1_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif           
*/
      // Global averaging
      
      static elem_t average[1][9216] row_align(MAX_BLOCK_LEN);
/*
      start = read_cycles();
      //if(cid == 0)
          tiled_global_average_auto(conv_5_out_alex1_pooled, average, conv_5_params_alex1.batch_size,                         
              conv_5_params_alex1.out_channels, conv_5_params_alex1.out_dim, WS);
         

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif
*/
//printf("entering FC\n");
      // fc_6
      start = read_cycles();

      tiled_opcode_matmul_nn_default(fc_6_params_alex1.I, (int)(fc_6_params_alex1.J), fc_6_params_alex1.K, fc_6_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          (elem_t*)average, (elem_t*)fc_6_w_alex1, (acc_t*)fc_6_b_alex1, (elem_t*)fc_6_out_alex1,
          RELU, fc_6_params_alex1.output_scale, 0, false,
          WS, num_array, cid);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] = end - start;
      //printf("fc_6 cycles: %d\n", end - start);
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    if(block == -1 || block == 2){

//printf("after fc6\n");
      // fc_7
      start = read_cycles();

      tiled_opcode_matmul_nn_default(fc_7_params_alex1.I, (int)(fc_7_params_alex1.J), fc_7_params_alex1.K, fc_7_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          (elem_t*)fc_6_out_alex1, (elem_t*)fc_7_w_alex1, (acc_t*)fc_7_b_alex1, (elem_t*)fc_7_out_alex1,
          RELU, fc_7_params_alex1.output_scale, 0, false,
          WS, num_array, cid);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
     //printf("after fc7\n");
      // fc_8
      start = read_cycles();

      tiled_opcode_matmul_nn_default(fc_8_params_alex1.I, fc_8_params_alex1.J, fc_8_params_alex1.K, fc_8_params_alex1.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          (elem_t*)fc_7_out_alex1, (elem_t*)fc_8_w_alex1, (acc_t*)fc_8_b_alex1, (elem_t*)fc_8_out_alex1,
          NO_ACTIVATION, fc_8_params_alex1.output_scale, 0, false,
          WS, num_array, cid);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 5){
        cycles[i] = conv_cycles[i];
      }
      else if(i < 8){
        cycles[i] = matmul_cycles[i - 5];
      }
      else if (i < 11){
        cycles[i] = pool_cycles[i - 8];
      }
      else{
        if(i == 11) cycles[i] = total_conv_cycles;
        if(i == 12) cycles[i] = total_matmul_cycles;
        if(i == 13) cycles[i] = total_pool_cycles;
        if(i == 14) cycles[i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles;
      }
    }
    return cycles;
#undef num_cycle
}

#if FULL == 1
uint64_t* alexnet_function_11(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
#define num_cycle (5+3+3+4)

  static uint64_t cycles[num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[5];
    uint64_t matmul_cycles[3];
    uint64_t pool_cycles[3];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif
    if(block == -1 || block == 0){
      // conv_1
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_1_params_alex11.batch_size, conv_1_params_alex11.in_dim, conv_1_params_alex11.in_channels,
          conv_1_params_alex11.out_channels, conv_1_params_alex11.out_dim,
          conv_1_params_alex11.stride, 1, conv_1_params_alex11.padding, conv_1_params_alex11.kernel_size,
          conv_1_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          
	  (elem_t*)image44, (elem_t*)conv_1_w_alex11, (acc_t*)conv_1_b_alex11, (elem_t*)conv_1_out_alex11,

          RELU, conv_1_params_alex11.output_scale, 0,
          1, 1, 0, false,
    //conv_1_params_alex11.pool_size, conv_1_params_alex11.pool_stride, conv_1_params_alex11.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif 
//printf("before pool 1\n");       
//if(cid == 0 || cid == 1)  
/* 
      start = read_cycles();
      tiled_opcode_pool_auto_multi(
          conv_1_params_alex11.batch_size,
          conv_1_params_alex11.out_channels, conv_1_params_alex11.out_dim, conv_1_params_alex11.out_dim_pooled,
          conv_1_params_alex11.out_stride,
          conv_1_params_alex11.pool_size, conv_1_params_alex11.pool_stride, conv_1_params_alex11.pool_padding,

          (elem_t*)conv_1_out_alex11, (elem_t*)conv_1_out_alex11_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          */
      // conv_2
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_2_params_alex11.batch_size, conv_2_params_alex11.in_dim, conv_2_params_alex11.in_channels,
          conv_2_params_alex11.out_channels, conv_2_params_alex11.out_dim,
          conv_2_params_alex11.stride, 1, conv_2_params_alex11.padding, conv_2_params_alex11.kernel_size,
          conv_2_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_1_out_alex11_pooled, (elem_t*)conv_2_w_alex11, (acc_t*)conv_2_b_alex11, (elem_t*)conv_2_out_alex11,

          RELU, conv_2_params_alex11.output_scale, 0,
          1, 1, 0, false,
    //conv_2_params_alex11.pool_size, conv_2_params_alex11.pool_stride, conv_2_params_alex11.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
    
//printf("before pool 2\n");       
//if(cid == 0 || cid == 1) 
/* 
      start = read_cycles();
      tiled_opcode_pool_auto_multi(
          conv_2_params_alex11.batch_size,
          conv_2_params_alex11.out_channels, conv_2_params_alex11.out_dim, conv_2_params_alex11.out_dim_pooled,
          conv_2_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          conv_2_params_alex11.pool_size, conv_2_params_alex11.pool_stride, conv_2_params_alex11.pool_padding,

          (elem_t*)conv_2_out_alex11, (elem_t*)conv_2_out_alex11_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += (end > start ? end - start : 0);
      pool_cycles[1] = (end > start ? end - start : 0);
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif          
*/
      // conv_3
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_3_params_alex11.batch_size, conv_3_params_alex11.in_dim, conv_3_params_alex11.in_channels,
          conv_3_params_alex11.out_channels, conv_3_params_alex11.out_dim,
          conv_3_params_alex11.stride, 1, conv_3_params_alex11.padding, conv_3_params_alex11.kernel_size,
          conv_3_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_2_out_alex11_pooled, (elem_t*)conv_3_w_alex11, (acc_t*)conv_3_b_alex11, (elem_t*)conv_3_out_alex11,

          RELU, conv_3_params_alex11.output_scale, 0,
          conv_3_params_alex11.pool_size, 0, conv_3_params_alex11.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_4
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_4_params_alex11.batch_size, conv_4_params_alex11.in_dim, conv_4_params_alex11.in_channels,
          conv_4_params_alex11.out_channels, conv_4_params_alex11.out_dim,
          conv_4_params_alex11.stride, 1, conv_4_params_alex11.padding, conv_4_params_alex11.kernel_size,
          conv_4_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_3_out_alex11, (elem_t*)conv_4_w_alex11, (acc_t*)conv_4_b_alex11, (elem_t*)conv_4_out_alex11,

          RELU, conv_4_params_alex11.output_scale, 0,
          conv_4_params_alex11.pool_size, 0, conv_4_params_alex11.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_5
      start = read_cycles();
      tiled_opcode_conv_default(
          conv_5_params_alex11.batch_size, conv_5_params_alex11.in_dim, conv_5_params_alex11.in_channels,
          conv_5_params_alex11.out_channels, conv_5_params_alex11.out_dim,
          conv_5_params_alex11.stride, 1, conv_5_params_alex11.padding, conv_5_params_alex11.kernel_size,
          conv_5_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,

          (elem_t*)conv_4_out_alex11, (elem_t*)conv_5_w_alex11, (acc_t*)conv_5_b_alex11, (elem_t*)conv_5_out_alex11,

          RELU, conv_5_params_alex11.output_scale, 0,
          1, 1, 0, false,
    //conv_5_params_alex11.pool_size, conv_5_params_alex11.pool_stride, conv_5_params_alex11.pool_padding, false,

          num_array, cid);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    if(block == -1 || block == 1){

//printf("before pool 3\n");       
//if(cid == 0 || cid == 1)  
/*
      start = read_cycles();
      tiled_opcode_pool_auto_multi(
          conv_5_params_alex11.batch_size,
          conv_5_params_alex11.out_channels, conv_5_params_alex11.out_dim, conv_5_params_alex11.out_dim_pooled,
          conv_5_params_alex11.out_stride,
          conv_5_params_alex11.pool_size, conv_5_params_alex11.pool_stride, conv_5_params_alex11.pool_padding,

          (elem_t*)conv_5_out_alex11, (elem_t*)conv_5_out_alex11_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif           
*/
      // Global averaging
      
      static elem_t average[1][9216] row_align(MAX_BLOCK_LEN);
/*
      start = read_cycles();
      //if(cid == 0)
          tiled_global_average_auto(conv_5_out_alex11_pooled, average, conv_5_params_alex11.batch_size,                         
              conv_5_params_alex11.out_channels, conv_5_params_alex11.out_dim, WS);
         

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif
*/
//printf("entering FC\n");
      // fc_6
      start = read_cycles();

      tiled_opcode_matmul_nn_default(fc_6_params_alex11.I, (int)(fc_6_params_alex11.J), fc_6_params_alex11.K, fc_6_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          (elem_t*)average, (elem_t*)fc_6_w_alex11, (acc_t*)fc_6_b_alex11, (elem_t*)fc_6_out_alex11,
          RELU, fc_6_params_alex11.output_scale, 0, false,
          WS, num_array, cid);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] = end - start;
     // printf("fc_6 cycles: %d\n", end - start);
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    if(block == -1 || block == 2){

//printf("after fc6\n");
      // fc_7
      start = read_cycles();

      tiled_opcode_matmul_nn_default(fc_7_params_alex11.I, (int)(fc_7_params_alex11.J), fc_7_params_alex11.K, fc_7_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          (elem_t*)fc_6_out_alex11, (elem_t*)fc_7_w_alex11, (acc_t*)fc_7_b_alex11, (elem_t*)fc_7_out_alex11,
          RELU, fc_7_params_alex11.output_scale, 0, false,
          WS, num_array, cid);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
     //printf("after fc7\n");
      // fc_8
      start = read_cycles();

      tiled_opcode_matmul_nn_default(fc_8_params_alex11.I, fc_8_params_alex11.J, fc_8_params_alex11.K, fc_8_params_alex11.out_stride,
	  input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram,
          (elem_t*)fc_7_out_alex11, (elem_t*)fc_8_w_alex11, (acc_t*)fc_8_b_alex11, (elem_t*)fc_8_out_alex11,
          NO_ACTIVATION, fc_8_params_alex11.output_scale, 0, false,
          WS, num_array, cid);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 5){
        cycles[i] = conv_cycles[i];
      }
      else if(i < 8){
        cycles[i] = matmul_cycles[i - 5];
      }
      else if (i < 11){
        cycles[i] = pool_cycles[i - 8];
      }
      else{
        if(i == 11) cycles[i] = total_conv_cycles;
        if(i == 12) cycles[i] = total_matmul_cycles;
        if(i == 13) cycles[i] = total_pool_cycles;
        if(i == 14) cycles[i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles;
      }
    }
    return cycles;
#undef num_cycle
}
#endif

#else

uint64_t* alexnet_function_1(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){

    int round = 3;
    int total_single_time = 18000000;
    if(block >= 0){
        round = 1;
        total_single_time /= 2;
    }
    dummy_workload(cid, round, total_single_time, num_array);
}

uint64_t* alexnet_function_11(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){

    int round = 3;
    int total_single_time = 18000000;
    if(block >= 0){
        round = 1;
        total_single_time /= 2;
    }
    dummy_workload(cid, round, total_single_time, num_array);
}


#endif
