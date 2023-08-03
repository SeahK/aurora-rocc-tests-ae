#ifndef midasnet_PARAMS_H
#define midasnet_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_0_midas1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_0_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_0_midas1_out[1][96][96][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_0_midas1_params = {.batch_size=1, .in_dim=96, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=96, .out_dim_pooled=96, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=9216, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_1_midas1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_1_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_1_midas1_out[1][96][96][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_midas1_params = {.batch_size=1, .in_dim=96, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=96, .out_dim_pooled=96, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=9216, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_2_midas1_w[64][144] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_midas1_b[144] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_2_midas1_out[1][96][96][144] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_midas1_params = {.batch_size=1, .in_dim=96, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=144, .out_stride=144, .weight_stride=144, .stride=1, .padding=0, .out_dim=96, .out_dim_pooled=96, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=9216, .K=64, .J=144, .output_scale=1, .res_scale=1};

static const elem_t conv_3_midas1_w[144][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_3_midas1_out[1][48][48][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=1, .in_channels=144, .in_stride=144, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=144, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_4_midas1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_midas1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_4_midas1_out[1][48][48][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_5_midas1_w[1728][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_midas1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_5_midas1_out[1][48][48][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=1728, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_6_midas1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_6_midas1_out[1][48][48][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_7_midas1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_midas1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_7_midas1_out[1][48][48][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_8_midas1_w[1728][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_midas1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_8_midas1_out[1][48][48][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=1728, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_9_midas1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_9_midas1_out[1][48][48][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_10_midas1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_midas1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_10_midas1_out[1][48][48][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_midas1_params = {.batch_size=1, .in_dim=48, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=48, .out_dim_pooled=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=2304, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_11_midas1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_11_midas1_out[1][24][24][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_12_midas1_w[64][288] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_midas1_b[288] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_12_midas1_out[1][24][24][288] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=288, .out_stride=288, .weight_stride=288, .stride=1, .padding=0, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=64, .J=288, .output_scale=1, .res_scale=1};

static const elem_t conv_13_midas1_w[7200][288] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_midas1_b[288] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_13_midas1_out[1][24][24][288] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=5, .in_channels=288, .in_stride=288, .out_channels=288, .out_stride=288, .weight_stride=288, .stride=1, .padding=2, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=7200, .J=288, .output_scale=1, .res_scale=1};

static const elem_t conv_14_midas1_w[288][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_14_midas1_out[1][24][24][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=1, .in_channels=288, .in_stride=288, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=288, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_15_midas1_w[64][288] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_midas1_b[288] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_15_midas1_out[1][24][24][288] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=288, .out_stride=288, .weight_stride=288, .stride=1, .padding=0, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=64, .J=288, .output_scale=1, .res_scale=1};

static const elem_t conv_16_midas1_w[7200][288] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_midas1_b[288] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_16_midas1_out[1][24][24][288] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=5, .in_channels=288, .in_stride=288, .out_channels=288, .out_stride=288, .weight_stride=288, .stride=1, .padding=2, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=7200, .J=288, .output_scale=1, .res_scale=1};

static const elem_t conv_17_midas1_w[288][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_midas1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_17_midas1_out[1][24][24][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=1, .in_channels=288, .in_stride=288, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=288, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_18_midas1_w[64][288] row_align(MAX_BLOCK_LEN);
static const acc_t conv_18_midas1_b[288] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_18_midas1_out[1][24][24][288] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_midas1_params = {.batch_size=1, .in_dim=24, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=288, .out_stride=288, .weight_stride=288, .stride=1, .padding=0, .out_dim=24, .out_dim_pooled=24, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=576, .K=64, .J=288, .output_scale=1, .res_scale=1};

static const elem_t conv_19_midas1_w[288][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_19_midas1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_19_midas1_out[1][12][12][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=288, .in_stride=288, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=288, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_20_midas1_w[96][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_20_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_20_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=96, .in_stride=96, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=96, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_21_midas1_w[5184][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_21_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_21_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=3, .in_channels=576, .in_stride=576, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=5184, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_22_midas1_w[576][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_22_midas1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_22_midas1_out[1][12][12][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=576, .in_stride=576, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=576, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_23_midas1_w[96][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_23_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_23_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=96, .in_stride=96, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=96, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_24_midas1_w[5184][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_24_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_24_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=3, .in_channels=576, .in_stride=576, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=5184, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_25_midas1_w[576][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_25_midas1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_25_midas1_out[1][12][12][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=576, .in_stride=576, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=576, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_26_midas1_w[96][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_26_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_26_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_26_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=96, .in_stride=96, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=96, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_27_midas1_w[5184][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_27_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_27_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_27_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=3, .in_channels=576, .in_stride=576, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=5184, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_28_midas1_w[576][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_28_midas1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_28_midas1_out[1][12][12][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_28_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=576, .in_stride=576, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=576, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_29_midas1_w[96][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_29_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_29_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_29_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=96, .in_stride=96, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=96, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_30_midas1_w[5184][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_30_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_30_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_30_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=3, .in_channels=576, .in_stride=576, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=5184, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_31_midas1_w[576][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_31_midas1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_31_midas1_out[1][12][12][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_31_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=576, .in_stride=576, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=576, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_32_midas1_w[96][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_32_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_32_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_32_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=96, .in_stride=96, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=96, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_33_midas1_w[14400][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_33_midas1_b[576] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_33_midas1_out[1][12][12][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_33_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=5, .in_channels=576, .in_stride=576, .out_channels=576, .out_stride=576, .weight_stride=576, .stride=1, .padding=2, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=14400, .J=576, .output_scale=1, .res_scale=1};

static const elem_t conv_34_midas1_w[576][136] row_align(MAX_BLOCK_LEN);
static const acc_t conv_34_midas1_b[136] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_34_midas1_out[1][12][12][136] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_34_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=576, .in_stride=576, .out_channels=136, .out_stride=136, .weight_stride=136, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=576, .J=136, .output_scale=1, .res_scale=1};

static const elem_t conv_35_midas1_w[136][816] row_align(MAX_BLOCK_LEN);
static const acc_t conv_35_midas1_b[816] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_35_midas1_out[1][12][12][816] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_35_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=136, .in_stride=136, .out_channels=816, .out_stride=816, .weight_stride=816, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=136, .J=816, .output_scale=1, .res_scale=1};

static const elem_t conv_36_midas1_w[20400][816] row_align(MAX_BLOCK_LEN);
static const acc_t conv_36_midas1_b[816] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_36_midas1_out[1][12][12][816] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_36_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=5, .in_channels=816, .in_stride=816, .out_channels=816, .out_stride=816, .weight_stride=816, .stride=1, .padding=2, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=20400, .J=816, .output_scale=1, .res_scale=1};

static const elem_t conv_37_midas1_w[816][136] row_align(MAX_BLOCK_LEN);
static const acc_t conv_37_midas1_b[136] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_37_midas1_out[1][12][12][136] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_37_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=816, .in_stride=816, .out_channels=136, .out_stride=136, .weight_stride=136, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=816, .J=136, .output_scale=1, .res_scale=1};

static const elem_t conv_38_midas1_w[136][816] row_align(MAX_BLOCK_LEN);
static const acc_t conv_38_midas1_b[816] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_38_midas1_out[1][12][12][816] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_38_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=136, .in_stride=136, .out_channels=816, .out_stride=816, .weight_stride=816, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=136, .J=816, .output_scale=1, .res_scale=1};

static const elem_t conv_39_midas1_w[20400][816] row_align(MAX_BLOCK_LEN);
static const acc_t conv_39_midas1_b[816] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_39_midas1_out[1][12][12][816] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_39_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=5, .in_channels=816, .in_stride=816, .out_channels=816, .out_stride=816, .weight_stride=816, .stride=1, .padding=2, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=20400, .J=816, .output_scale=1, .res_scale=1};

static const elem_t conv_40_midas1_w[816][136] row_align(MAX_BLOCK_LEN);
static const acc_t conv_40_midas1_b[136] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_40_midas1_out[1][12][12][136] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_40_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=816, .in_stride=816, .out_channels=136, .out_stride=136, .weight_stride=136, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=816, .J=136, .output_scale=1, .res_scale=1};

static const elem_t conv_41_midas1_w[136][816] row_align(MAX_BLOCK_LEN);
static const acc_t conv_41_midas1_b[816] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_41_midas1_out[1][12][12][816] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_41_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=136, .in_stride=136, .out_channels=816, .out_stride=816, .weight_stride=816, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=136, .J=816, .output_scale=1, .res_scale=1};

static const elem_t conv_42_midas1_w[20400][816] row_align(MAX_BLOCK_LEN);
static const acc_t conv_42_midas1_b[816] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_42_midas1_out[1][12][12][816] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_42_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=5, .in_channels=816, .in_stride=816, .out_channels=816, .out_stride=816, .weight_stride=816, .stride=1, .padding=2, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=20400, .J=816, .output_scale=1, .res_scale=1};

static const elem_t conv_43_midas1_w[816][136] row_align(MAX_BLOCK_LEN);
static const acc_t conv_43_midas1_b[136] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_43_midas1_out[1][12][12][136] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_43_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=816, .in_stride=816, .out_channels=136, .out_stride=136, .weight_stride=136, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=816, .J=136, .output_scale=1, .res_scale=1};

static const elem_t conv_44_midas1_w[136][816] row_align(MAX_BLOCK_LEN);
static const acc_t conv_44_midas1_b[816] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_44_midas1_out[1][12][12][816] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_44_midas1_params = {.batch_size=1, .in_dim=12, .kernel_size=1, .in_channels=136, .in_stride=136, .out_channels=816, .out_stride=816, .weight_stride=816, .stride=1, .padding=0, .out_dim=12, .out_dim_pooled=12, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=144, .K=136, .J=816, .output_scale=1, .res_scale=1};

#endif
