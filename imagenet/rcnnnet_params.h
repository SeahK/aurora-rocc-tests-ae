#ifndef rcnnnet_PARAMS_H
#define rcnnnet_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_0_rcnn1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_0_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_0_rcnn1_out[1][160][160][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_0_rcnn1_params = {.batch_size=1, .in_dim=160, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=160, .out_dim_pooled=160, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_1_rcnn1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_1_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_1_rcnn1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_2_rcnn1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_2_rcnn1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_3_rcnn1_w[64][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_3_rcnn1_out[1][40][40][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_4_rcnn1_w[64][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_4_rcnn1_out[1][40][40][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_5_rcnn1_w[256][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_5_rcnn1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=256, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_6_rcnn1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_6_rcnn1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_7_rcnn1_w[64][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_7_rcnn1_out[1][40][40][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_8_rcnn1_w[256][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_8_rcnn1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=256, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_9_rcnn1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_9_rcnn1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_10_rcnn1_w[64][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_10_rcnn1_out[1][40][40][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_11_rcnn1_w[256][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_11_rcnn1_out[1][40][40][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=256, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_12_rcnn1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_12_rcnn1_out[1][40][40][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_13_rcnn1_w[128][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_13_rcnn1_out[1][20][20][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=128, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_14_rcnn1_w[256][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_14_rcnn1_out[1][40][40][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_rcnn1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=256, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_15_rcnn1_w[512][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_15_rcnn1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=512, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_16_rcnn1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_16_rcnn1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_17_rcnn1_w[128][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_17_rcnn1_out[1][20][20][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=128, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_18_rcnn1_w[512][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_18_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_18_rcnn1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=512, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_19_rcnn1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_19_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_19_rcnn1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_20_rcnn1_w[128][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_20_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_20_rcnn1_out[1][20][20][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=128, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_21_rcnn1_w[512][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_21_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_21_rcnn1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=512, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_22_rcnn1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_22_rcnn1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_22_rcnn1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_23_rcnn1_w[128][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_23_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_23_rcnn1_out[1][20][20][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=128, .in_stride=192, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=128, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_24_rcnn1_w[512][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_24_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_24_rcnn1_out[1][20][20][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=512, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_25_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_25_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_25_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_26_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_26_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_26_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_26_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_27_rcnn1_w[512][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_27_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_27_rcnn1_out[1][20][20][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_27_rcnn1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=512, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_28_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_28_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_28_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_28_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_29_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_29_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_29_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_29_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_30_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_30_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_30_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_30_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_31_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_31_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_31_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_31_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_32_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_32_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_32_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_32_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_33_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_33_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_33_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_33_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_34_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_34_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_34_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_34_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_35_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_35_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_35_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_35_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_36_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_36_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_36_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_36_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_37_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_37_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_37_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_37_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_38_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_38_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_38_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_38_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_39_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_39_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_39_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_39_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_40_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_40_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_40_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_40_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_41_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_41_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_41_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_41_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_42_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_42_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_42_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_42_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_43_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_43_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_43_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_43_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_44_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_44_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_44_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_44_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_45_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_45_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_45_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_45_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_46_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_46_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_46_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_46_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_47_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_47_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_47_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_47_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_48_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_48_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_48_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_48_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_49_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_49_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_49_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_49_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_50_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_50_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_50_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_50_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_51_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_51_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_51_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_51_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_52_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_52_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_52_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_52_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_53_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_53_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_53_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_53_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_54_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_54_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_54_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_54_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_55_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_55_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_55_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_55_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_56_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_56_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_56_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_56_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_57_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_57_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_57_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_57_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_58_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_58_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_58_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_58_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_59_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_59_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_59_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_59_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_60_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_60_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_60_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_60_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_61_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_61_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_61_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_61_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_62_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_62_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_62_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_62_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_63_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_63_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_63_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_63_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_64_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_64_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_64_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_64_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_65_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_65_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_65_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_65_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_66_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_66_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_66_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_66_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_67_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_67_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_67_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_67_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_68_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_68_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_68_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_68_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_69_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_69_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_69_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_69_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_70_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_70_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_70_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_70_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_71_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_71_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_71_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_71_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_72_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_72_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_72_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_72_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_73_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_73_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_73_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_73_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_74_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_74_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_74_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_74_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_75_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_75_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_75_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_75_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_76_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_76_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_76_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_76_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_77_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_77_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_77_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_77_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_78_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_78_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_78_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_78_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_79_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_79_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_79_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_79_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_80_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_80_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_80_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_80_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_81_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_81_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_81_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_81_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_82_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_82_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_82_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_82_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_83_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_83_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_83_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_83_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_84_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_84_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_84_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_84_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_85_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_85_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_85_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_85_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_86_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_86_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_86_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_86_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_87_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_87_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_87_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_87_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_88_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_88_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_88_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_88_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_89_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_89_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_89_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_89_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_90_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_90_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_90_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_90_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_91_rcnn1_w[1024][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_91_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_91_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_91_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_92_rcnn1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_92_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_92_rcnn1_out[1][10][10][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_92_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_93_rcnn1_w[256][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_93_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_93_rcnn1_out[1][10][10][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_93_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=256, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=256, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_94_rcnn1_w[1024][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_94_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_94_rcnn1_out[1][10][10][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_94_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_95_rcnn1_w[4608][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_95_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_95_rcnn1_out[1][5][5][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_95_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=3, .in_channels=512, .in_stride=576, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=4608, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_96_rcnn1_w[512][2112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_96_rcnn1_b[2048] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_96_rcnn1_out[1][5][5][2112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_96_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=2048, .out_stride=2112, .weight_stride=2112, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=512, .J=2048, .output_scale=1, .res_scale=1};

static const elem_t conv_97_rcnn1_w[1024][2112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_97_rcnn1_b[2048] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_97_rcnn1_out[1][10][10][2112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_97_rcnn1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=2048, .out_stride=2112, .weight_stride=2112, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=1024, .J=2048, .output_scale=1, .res_scale=1};

static const elem_t conv_98_rcnn1_w[2048][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_98_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_98_rcnn1_out[1][5][5][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_98_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=2048, .in_stride=2112, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=2048, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_99_rcnn1_w[4608][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_99_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_99_rcnn1_out[1][5][5][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_99_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=3, .in_channels=512, .in_stride=576, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=4608, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_100_rcnn1_w[512][2112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_100_rcnn1_b[2048] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_100_rcnn1_out[1][5][5][2112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_100_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=2048, .out_stride=2112, .weight_stride=2112, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=512, .J=2048, .output_scale=1, .res_scale=1};

static const elem_t conv_101_rcnn1_w[2048][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_101_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_101_rcnn1_out[1][5][5][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_101_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=2048, .in_stride=2112, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=2048, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_102_rcnn1_w[4608][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_102_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_102_rcnn1_out[1][5][5][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_102_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=3, .in_channels=512, .in_stride=576, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=4608, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_103_rcnn1_w[512][2112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_103_rcnn1_b[2048] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_103_rcnn1_out[1][5][5][2112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_103_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=2048, .out_stride=2112, .weight_stride=2112, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=512, .J=2048, .output_scale=1, .res_scale=1};

static const elem_t conv_104_rcnn1_w[2048][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_104_rcnn1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_104_rcnn1_out[1][5][5][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_104_rcnn1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=2048, .in_stride=2112, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=2048, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_105_rcnn1_w[2304][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_105_rcnn1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_105_rcnn1_out[1][3][3][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_105_rcnn1_params = {.batch_size=1, .in_dim=3, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=3, .out_dim_pooled=3, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=9, .K=2304, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_106_rcnn1_w[512][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_106_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_106_rcnn1_out[1][3][3][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_106_rcnn1_params = {.batch_size=1, .in_dim=3, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=3, .out_dim_pooled=3, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=9, .K=512, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_107_rcnn1_w[512][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_107_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_107_rcnn1_out[1][3][3][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_107_rcnn1_params = {.batch_size=1, .in_dim=3, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=3, .out_dim_pooled=3, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=9, .K=512, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_108_rcnn1_w[5120][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_108_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_108_rcnn1_out[1][2][2][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_108_rcnn1_params = {.batch_size=1, .in_dim=2, .kernel_size=4, .in_channels=320, .in_stride=320, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=1, .out_dim=2, .out_dim_pooled=2, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=4, .K=5120, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_109_rcnn1_w[1024][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_109_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_109_rcnn1_out[1][1][1][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_109_rcnn1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=1024, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_110_rcnn1_w[1024][1088] row_align(MAX_BLOCK_LEN);
static const acc_t conv_110_rcnn1_b[1024] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_110_rcnn1_out[1][1][1][1088] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_110_rcnn1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=1024, .out_stride=1088, .weight_stride=1088, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=1024, .J=1024, .output_scale=1, .res_scale=1};

static const elem_t conv_111_rcnn1_w[1024][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_111_rcnn1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_111_rcnn1_out[1][1][1][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_111_rcnn1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=1024, .in_stride=1088, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=1024, .J=64, .output_scale=1, .res_scale=1};

#endif
