#ifndef ritnet_PARAMS_H
#define ritnet_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_0_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_0_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_0_rit1_out[1][160][160][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_0_rit1_params = {.batch_size=1, .in_dim=160, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=160, .out_dim_pooled=160, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_1_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_1_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_1_rit1_out[1][160][160][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_rit1_params = {.batch_size=1, .in_dim=160, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=160, .out_dim_pooled=160, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25600, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_2_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_2_rit1_out[1][160][160][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_rit1_params = {.batch_size=1, .in_dim=160, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=160, .out_dim_pooled=160, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_3_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_3_rit1_out[1][160][160][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_rit1_params = {.batch_size=1, .in_dim=160, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=160, .out_dim_pooled=160, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25600, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_4_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_4_rit1_out[1][160][160][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_rit1_params = {.batch_size=1, .in_dim=160, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=160, .out_dim_pooled=160, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_5_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_5_rit1_out[1][80][80][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_rit1_params = {.batch_size=1, .in_dim=80, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_6_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_6_rit1_out[1][80][80][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_rit1_params = {.batch_size=1, .in_dim=80, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_7_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_7_rit1_out[1][80][80][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_rit1_params = {.batch_size=1, .in_dim=80, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_8_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_8_rit1_out[1][80][80][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_rit1_params = {.batch_size=1, .in_dim=80, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_9_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_9_rit1_out[1][80][80][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_rit1_params = {.batch_size=1, .in_dim=80, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_10_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_10_rit1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_rit1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_11_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_11_rit1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_rit1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_12_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_12_rit1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_rit1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_13_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_13_rit1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_rit1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_14_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_14_rit1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_rit1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_15_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_15_rit1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_rit1_params = {.batch_size=1, .in_dim=20, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_16_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_16_rit1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_rit1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_17_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_17_rit1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_rit1_params = {.batch_size=1, .in_dim=20, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_18_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_18_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_18_rit1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_rit1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_19_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_19_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_19_rit1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_rit1_params = {.batch_size=1, .in_dim=20, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_20_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_20_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_20_rit1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_rit1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_21_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_21_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_21_rit1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_rit1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_22_rit1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_22_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_22_rit1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_rit1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_23_rit1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_23_rit1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_23_rit1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_rit1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=64, .J=64, .output_scale=1, .res_scale=1};

#endif
