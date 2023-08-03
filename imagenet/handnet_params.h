#ifndef handnet_PARAMS_H
#define handnet_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_0_hand1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_0_hand1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_0_hand1_out[1][128][128][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_0_hand1_params = {.batch_size=1, .in_dim=128, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=128, .out_dim_pooled=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16384, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_1_hand1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_1_hand1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_1_hand1_out[1][64][64][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_hand1_params = {.batch_size=1, .in_dim=64, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=64, .out_dim_pooled=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4096, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_2_hand1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_hand1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_2_hand1_out[1][64][64][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_hand1_params = {.batch_size=1, .in_dim=64, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=64, .out_dim_pooled=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4096, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_3_hand1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_3_hand1_out[1][64][64][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_hand1_params = {.batch_size=1, .in_dim=64, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=64, .out_dim_pooled=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4096, .K=64, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_4_hand1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_4_hand1_out[1][64][64][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_hand1_params = {.batch_size=1, .in_dim=64, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=64, .out_dim_pooled=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4096, .K=64, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_5_hand1_w[128][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_hand1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_5_hand1_out[1][32][32][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=128, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_6_hand1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_hand1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_6_hand1_out[1][32][32][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_7_hand1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_7_hand1_out[1][32][32][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=64, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_8_hand1_w[128][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_8_hand1_out[1][32][32][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=128, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_9_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_9_hand1_out[1][32][32][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_10_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_10_hand1_out[1][32][32][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_11_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_11_hand1_out[1][32][32][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_12_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_12_hand1_out[1][32][32][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_13_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_13_hand1_out[1][32][32][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_14_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_14_hand1_out[1][32][32][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_15_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_15_hand1_out[1][32][32][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_16_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_16_hand1_out[1][32][32][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_17_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_17_hand1_out[1][32][32][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_hand1_params = {.batch_size=1, .in_dim=32, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=32, .out_dim_pooled=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1024, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_18_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_18_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_18_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_19_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_19_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_19_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_20_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_20_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_20_hand1_out[1][16][16][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_21_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_21_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_21_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_22_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_22_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_22_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_23_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_23_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_23_hand1_out[1][16][16][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_24_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_24_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_24_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_25_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_25_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_25_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_26_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_26_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_26_hand1_out[1][16][16][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_26_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_27_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_27_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_27_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_27_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_28_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_28_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_28_hand1_out[1][16][16][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_28_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_29_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_29_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_29_hand1_out[1][16][16][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_29_hand1_params = {.batch_size=1, .in_dim=16, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=16, .out_dim_pooled=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=256, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_30_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_30_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_30_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_30_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_31_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_31_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_31_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_31_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_32_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_32_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_32_hand1_out[1][8][8][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_32_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_33_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_33_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_33_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_33_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_34_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_34_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_34_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_34_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_35_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_35_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_35_hand1_out[1][8][8][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_35_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_36_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_36_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_36_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_36_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_37_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_37_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_37_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_37_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_38_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_38_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_38_hand1_out[1][8][8][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_38_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_39_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_39_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_39_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_39_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_40_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_40_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_40_hand1_out[1][8][8][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_40_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_41_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_41_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_41_hand1_out[1][8][8][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_41_hand1_params = {.batch_size=1, .in_dim=8, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=8, .out_dim_pooled=8, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=64, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_42_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_42_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_42_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_42_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_43_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_43_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_43_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_43_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_44_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_44_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_44_hand1_out[1][4][4][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_44_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_45_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_45_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_45_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_45_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_46_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_46_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_46_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_46_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_47_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_47_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_47_hand1_out[1][4][4][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_47_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_48_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_48_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_48_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_48_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_49_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_49_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_49_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_49_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_50_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_50_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_50_hand1_out[1][4][4][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_50_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_51_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_51_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_51_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_51_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_52_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_52_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_52_hand1_out[1][4][4][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_52_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_53_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_53_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_53_hand1_out[1][4][4][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_53_hand1_params = {.batch_size=1, .in_dim=4, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=4, .out_dim_pooled=4, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=16, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_54_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_54_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_54_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_54_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_55_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_55_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_55_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_55_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_56_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_56_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_56_hand1_out[1][2][2][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_56_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_57_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_57_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_57_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_57_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_58_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_58_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_58_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_58_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_59_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_59_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_59_hand1_out[1][2][2][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_59_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_60_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_60_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_60_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_60_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_61_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_61_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_61_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_61_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_62_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_62_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_62_hand1_out[1][2][2][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_62_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_63_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_63_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_63_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_63_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_64_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_64_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_64_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_64_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_65_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_65_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_65_hand1_out[1][2][2][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_65_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_66_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_66_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_66_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_66_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_67_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_67_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_67_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_67_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_68_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_68_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_68_hand1_out[1][2][2][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_68_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_69_hand1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_69_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_69_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_69_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_70_hand1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_70_hand1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_70_hand1_out[1][2][2][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_70_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_71_hand1_w[128][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_71_hand1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_71_hand1_out[1][2][2][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_71_hand1_params = {.batch_size=1, .in_dim=2, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=128, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_72_hand1_w[1024][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_72_hand1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_72_hand1_out[1][1][1][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_72_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=1024, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_73_hand1_w[512][5184] row_align(MAX_BLOCK_LEN);
static const acc_t conv_73_hand1_b[5120] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_73_hand1_out[1][1][1][5184] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_73_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=5120, .out_stride=5184, .weight_stride=5184, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=512, .J=5120, .output_scale=1, .res_scale=1};

static const elem_t conv_74_hand1_w[192][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_74_hand1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_74_hand1_out[1][1][1][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_74_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=192, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_75_hand1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_75_hand1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_75_hand1_out[1][1][1][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_75_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_76_hand1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_76_hand1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_76_hand1_out[1][1][1][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_76_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_77_hand1_w[192][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_77_hand1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_77_hand1_out[1][1][1][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_77_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=192, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_78_hand1_w[192][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_78_hand1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_78_hand1_out[1][1][1][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_78_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=192, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_79_hand1_w[512][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_79_hand1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_79_hand1_out[1][1][1][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_79_hand1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=512, .J=512, .output_scale=1, .res_scale=1};

#endif
