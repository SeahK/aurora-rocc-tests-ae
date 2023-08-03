#ifndef GEMMINI_NN_H
#define GEMMINI_NN_H

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/rerocc.h"
#include "include/gemmini.h"
#include "include/gemmini_testutils.h"

struct ConvParams {
    int batch_size;
    int in_dim, out_dim;
    int kernel_size;
    int in_channels;
    int out_channels;
    int stride;
    int out_stride, weight_stride, in_stride;
    int padding;
    bool bias;
    bool depthwise;
    int n_patches;
    int patch_size;
    acc_scale_t output_scale;
    scale_t res_scale;
    int pool_size, pool_stride, pool_padding, out_dim_pooled;
    
    int I, J, K;
};

struct FcParams {
    int batch_size;
    int in_features;
    int out_features;
    int out_stride;
    acc_scale_t output_scale;
    bool bias;

    int I, J, K;
};

struct BertParams {
    int seq_len;
    int hidden_dim;
    int num_head;
    int expansion_dim;
    int seq_stride;
    int expansion_stride;
    int hidden_stride;

};


#define HIST_IMAGES(IMAGES) \
    for (int num = -128; num <= 127; num++) { \
        int count = 0; \
        for (int i = 0; i < sizeof(IMAGES)/sizeof(IMAGES[0]); i++) { \
            for (int j = 0; j < sizeof(IMAGES[0])/sizeof(IMAGES[0][0]); j++) { \
                for (int k = 0; k < sizeof(IMAGES[0][0])/sizeof(IMAGES[0][0][0]); k++) { \
                    for (int l = 0; l < sizeof(IMAGES[0][0][0])/sizeof(IMAGES[0][0][0][0]); l++) { \
                        if (IMAGES[i][j][k][l] == num) { \
                            count++; \
                        } \
                    } \
                } \
            } \
        } \
        if (count > 0) \
            printf("%d: %d times\n", num, count); \
    }

#define HIST_MATRIX(MATRIX) \
    for (int num = -128; num <= 127; num++) { \
        int count = 0; \
        for (int i = 0; i < sizeof(MATRIX)/sizeof(MATRIX[0]); i++) { \
            for (int j = 0; j < sizeof(MATRIX[0])/sizeof(MATRIX[0][0]); j++) { \
                if (MATRIX[i][j] == num) { \
                    count++; \
                } \
            } \
        } \
        if (count > 0) \
            printf("%d: %d times\n", num, count); \
    }



static void tiled_matmul_nn_default(size_t dim_I, size_t dim_J, size_t dim_K,
  size_t stride_C,
  bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram,
  elem_t* A, elem_t* B,
  void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  size_t num_array, size_t cid){
  //size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

  size_t stride_A = (dim_K % 128 == 0) ? dim_K + 64 : dim_K;
  size_t stride_B = (dim_J % 128 == 0) ? dim_J + 64 : dim_J;

//  printf("matmul A: 0x%08lx, B: 0x%08lx, C: 0x08lx\n", A, B, C);
  tiled_matmul_auto_multi(
      dim_I, dim_J, dim_K,
      stride_A, stride_B, stride_C, stride_C,
      A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
      A, B, D, C,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 
      false, false,
      act, scale, relu6_shift, repeating_bias, false,
      num_array, cid, -1, -1); // start tracker: 0
      //orow_divide, batch_divide, cid, group_id);

}

/*
// This function runs a tiled matrix multiplication, with explicit tiling
// factors
static void tiled_matmul_nn(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    if (check)
        printf("%s: gemmini\n", layer_name);

    tiled_matmul(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, relu6_shift, repeating_bias,
        tile_I, tile_J, tile_K,
        false, false,
        false, false,
        0,
        tiled_matmul_type);

    if (check) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[dim_I][dim_J];
        tiled_matmul_auto(dim_I, dim_J, dim_K,
            (elem_t*)A, (elem_t*)B, D, (elem_t*)gold, 
            dim_K, dim_J, dim_J, dim_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            act, scale, relu6_shift, repeating_bias,
            false, false,
            false, false,
            0,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_nn_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    if (check)
        printf("%s: gemmini\n", layer_name);

    tiled_matmul_auto(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, relu6_shift, repeating_bias,
        false, false,
        false, false,
        0,
        tiled_matmul_type);

    if (check) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[dim_I][dim_J];
        tiled_matmul_auto(dim_I, dim_J, dim_K,
            (elem_t*)A, (elem_t*)B, D, (elem_t*)gold, 
            dim_K, dim_J, dim_J, dim_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            act, scale, relu6_shift, repeating_bias,
            false, false,
            false, false,
            0,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

static void conv_dw(size_t I, size_t J,
    const size_t batch_size, const size_t channels, const size_t in_dim, const size_t out_dim, const size_t kernel_size,
    const elem_t input[batch_size][in_dim][in_dim][channels],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_dim][out_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_dim && in_col >= 0 && in_col < params->in_dim) {
                                result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t scaled = ACC_SCALE(result, params->output_scale);

                    if (scaled > elem_t_max) {
                        scaled = elem_t_max;
                    } else if (scaled < elem_t_min) {
                        scaled = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_dim * params->out_dim + out_row * params->out_dim + out_col;
                    output[r][channel] = scaled;
                    // output[batch][out_row][out_col][channel] = scaled;
                }
            }
        }
    }
}

static void conv_dw_with_col2im(size_t prev_I, size_t prev_J, size_t I, size_t J,
    const size_t batch_size, const size_t channels, const size_t out_dim, const size_t kernel_size,
    const elem_t input[prev_I][prev_J],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_dim][out_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_dim && in_col >= 0 && in_col < params->in_dim) {
                                // result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];

                                size_t r = batch * params->in_dim * params->in_dim + in_row * params->in_dim + in_col;

                                result += input[r][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t scaled = ACC_SCALE(result, params->output_scale);

                    if (scaled > elem_t_max) {
                        scaled = elem_t_max;
                    } else if (scaled < elem_t_min) {
                        scaled = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_dim * params->out_dim + out_row * params->out_dim + out_col;
                    output[r][channel] = scaled;
                    // output[batch][out_row][out_col][channel] = scaled;
                }
            }
        }
    }
}

static void im2col(size_t batch_size, size_t channels, size_t im_dim,
    size_t I, size_t K,
    const elem_t input[batch_size][im_dim][im_dim][channels],
    elem_t output[I][K],
    const struct ConvParams * params)
{
    int patch_row = 0;

    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int patch_col = 0;

                for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                    for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                        for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;
                            
                            if (pixel_row < 0 || pixel_row >= params->in_dim
                                || pixel_col < 0 || pixel_col >= params->in_dim) {
                                // output[patch_row][patch_col] = 0;
                            } else {
                                output[patch_row][patch_col] = input[n_batch][pixel_row][pixel_col][im_channel];
                            }

                            patch_col++;
                        }
                    }
                }
                
                patch_row++;
            }
        }
    }
}

static void im2col_with_col2im(size_t prev_I, size_t prev_J,
    size_t next_I, size_t next_K,
    const elem_t input[prev_I][prev_J],
    elem_t output[next_I][next_K],
    const struct ConvParams * params)
{
    int out_row = 0;

    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int out_col = 0;

                for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                    for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                        for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;

                            if (pixel_row < 0 || pixel_row >= params->in_dim
                                || pixel_col < 0 || pixel_col >= params->in_dim) {
                                // output[out_row][out_col] = 0;
                            } else {
                                int in_row = n_batch * params->in_dim * params->in_dim + pixel_row * params->in_dim + pixel_col;
                                int in_col = im_channel;

                                output[out_row][out_col] = input[in_row][in_col];
                            }

                            out_col++;
                        }
                    }
                }

                out_row++;
            }
        }
    }
}

// Compute C = A + B with saturating add
void vecadd(size_t len, const elem_t * A, const elem_t * B, elem_t * C, scale_t A_shift) {
    for (size_t i = 0; i < len; i++) {
        acc_t result = MVIN_SCALE(A[i], A_shift) + B[i];

        if (result > elem_t_max) {
            result = elem_t_max;
        } else if (result < elem_t_min) {
            result = elem_t_min;
        }

        C[i] = result;
    }
}

void resadd1(const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[batch_size][im_dim][im_dim][channels],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    acc_t result = MVIN_SCALE(A[batch][row][col][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd2(const size_t I, const size_t J,
    const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[I][J],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = MVIN_SCALE(A[r][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd3(const size_t I, const size_t J,
    const elem_t A[I][J],
    const elem_t B[I][J],
    elem_t C[I][J],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = MVIN_SCALE(A[r][channel], params->res_scale) + B[r][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[r][channel] = result;
                }
            }
        }
    }
}

// Pooling
void pool(size_t batch_size, size_t channels, size_t in_dim, size_t out_dim,
    elem_t input[batch_size][in_dim][in_dim][channels],
    elem_t output[batch_size][out_dim][out_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    // size_t in_dim = params->out_dim;
    size_t padding = params->pool_padding;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                if (input[batch][in_row][in_col][channel] > result) {
                                    result = input[batch][in_row][in_col][channel];
                                }
                            } else if (0 > result) {
                                result = 0;
                            }

                            in_col++;
                        }

                        in_row++;
                    }
                    
                    output[batch][out_row][out_col][channel] = result;
                }
            }
        }
    }
}

void pool_with_col2im(size_t I, size_t J,
    size_t batch_size, size_t channels, size_t out_dim,
    elem_t input[I][J],
    elem_t output[batch_size][out_dim][out_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    size_t in_dim = params->out_dim;
    size_t padding = params->pool_padding;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                if (input[batch * in_dim * in_dim + in_row * in_dim + in_col][channel] > result) {
                                    result = input[batch * in_dim * in_dim + in_row * in_dim + in_col][channel];
                                }
                            } else if (0 > result) {
                                result = 0;
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    output[batch][out_row][out_col][channel] = result;
                }
            }
        }
    }
}
*/

#ifndef BAREMETAL 
void dummy_workload(int cid, int round, uint64_t total_time, int num_array){
    int single_time = (int)(total_time / round);

    // ToDo: update core_required_gemmini based on current score
    // ToDo: update requested_gemmini based on current score
    // ToDo: update mutex lock with different keys
    int queue_id = workload_running[cid]; 
    int workload_type = total_queue_type[queue_id];
    for (int i = 0; i < round; i++){
      num_array = core_num_gemmini[cid];
      if(i % 2 == 1 && (mode != 1 && mode != 2)){
        int num_array_index = 0;
        if(num_array == 2) num_array_index = 1;
        else if(num_array == 4) num_array_index = 2;
        uint64_t current_cycle = read_cycles() - global_start_time[cid];
        uint64_t old = current_cycle - total_queue_dispatch[queue_id];
        uint64_t slack = target_cycles[workload_type] - old;
        int score = current_score[cid];
        //int score = (int)((100*slack) / core_togo[num_array_index][cid]); 
      
        pthread_mutex_lock(&ex_queue_mutex);
        bool need_release = false;
        if(num_array >= 2)
          for(int c = 0; c < NUM_CORE; c++){
             if(score > current_score[c] * 2 && core_num_gemmini_share[c] < 4){ // this one is significantly faster thab other
                need_release = true;
             }
             else if(current_score[c] > score && core_num_gemmini_share[c] > 1){ // can release other one first
                need_release = false;
                break;
             }
          }
        bool need_acquire = (num_array < 4);
        for(int c = 0; c < NUM_CORE; c++){
            if(score > current_score[c] && core_num_gemmini_share[c] < 4){
                need_acquire = false;
                break;
            }
        }
        if(need_release){
            int accel = core_gemmini[cid][num_array-1];
            num_array --;
            //core_num_gemmini[cid] --;
            core_gemmini[cid][num_array] = -1;
#if PRINT == 1
            printf("cid %d release accel %d\n", cid, accel);
#endif
            rerocc_release(num_array);
            gemmini_status[accel] = -1;
            if(num_array == 3){
              accel = core_gemmini[cid][num_array-1];
#if PRINT == 1
              printf("cid %d release accel %d\n", cid, accel);
#endif
 
              num_array --;
              //core_num_gemmini[cid] --;
              core_gemmini[cid][num_array] = -1;
              rerocc_release(num_array);
              gemmini_status[accel] = -1;
            }
        }
        else if(need_acquire){
            int save_accel = -1;
            for(int accel = 0; accel < NUM_ARRAY; accel ++){
                if(gemmini_status[accel] == -1){
                    if(core_num_gemmini[cid] == 1){
                        core_gemmini[cid][num_array] = accel;
#if PRINT == 1
                        printf("cid %d (prev %d num accel) claim new accel %d\n", cid, num_array, accel);
#endif
                        gemmini_status[accel] = cid;
                        while(!rerocc_acquire(num_array, 1<<accel)){}
                        //core_num_gemmini[cid]++;
                        num_array ++;
                        break;
                    }
                    else if(core_num_gemmini[cid] == 2){
                        if (save_accel == -1)
                            save_accel = accel;
                        else {
                            core_gemmini[cid][num_array] = save_accel;
                            gemmini_status[save_accel] = cid;
                            core_gemmini[cid][num_array+1] = accel;
                            gemmini_status[accel] = cid;
#if PRINT == 1
                           printf("cid %d (prev %d num accel) claim new accel %d, %d\n", cid, num_array, save_accel, accel);
#endif 
                            while(!rerocc_acquire(num_array, 1<<save_accel)){}
                            while(!rerocc_acquire(num_array+1, 1<<accel)){}
                            num_array+=2;
                            break;
                        }
                    }
                }
            }
        }
        //acquire or release happend -> need to update score
        if(core_num_gemmini[cid] != num_array){
#if PRINT == 1
            //printf("cid %d realloc happend from num %d to num %d\n", cid, core_num_gemmini[cid], num_array);
#endif
            core_num_gemmini[cid] = num_array; // locally
            core_num_gemmini_share[cid] = num_array; 
            num_array_index = 0;
            if(num_array == 2) num_array_index = 1;
            else if(num_array == 4) num_array_index = 2;
            score = (int)((100*slack)/core_togo[num_array_index][cid]);
            current_score[cid] = score;
        }
        pthread_mutex_unlock(&ex_queue_mutex);
      }
      uint64_t time_gap = 1000;
      uint64_t current = read_cycles();
      int this_time_gap = (int)(single_time / num_array);
      while(time_gap < this_time_gap){
        time_gap = read_cycles() - current;
      }
      for(int j = 0; j < 3; j++){
          core_togo[j][cid] -= sp_layer_cycles[j][workload_type][layer_pointer[cid]];
      }
      layer_pointer[cid] ++;
    }
}
#endif
#endif // GEMMINI_NN_H

