#ifndef rgbnet_PARAMS_H
#define rgbnet_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_0_rgb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_0_rgb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_0_rgb1_out[1][456][456][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_0_rgb1_params = {.batch_size=1, .in_dim=456, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=456, .out_dim_pooled=456, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=207936, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_1_rgb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_1_rgb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_1_rgb1_out[1][228][228][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_rgb1_params = {.batch_size=1, .in_dim=228, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=228, .out_dim_pooled=228, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=51984, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_2_rgb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_rgb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_2_rgb1_out[1][228][228][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_rgb1_params = {.batch_size=1, .in_dim=228, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=228, .out_dim_pooled=228, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=51984, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_3_rgb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_rgb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_3_rgb1_out[1][228][228][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_rgb1_params = {.batch_size=1, .in_dim=228, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=228, .out_dim_pooled=228, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=51984, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_4_rgb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_rgb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_4_rgb1_out[1][228][228][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_rgb1_params = {.batch_size=1, .in_dim=228, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=228, .out_dim_pooled=228, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=51984, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_5_rgb1_w[576][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_rgb1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_5_rgb1_out[1][228][228][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_rgb1_params = {.batch_size=1, .in_dim=228, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=228, .out_dim_pooled=228, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=51984, .K=576, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_6_rgb1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_rgb1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_6_rgb1_out[1][114][114][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_rgb1_params = {.batch_size=1, .in_dim=114, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=114, .out_dim_pooled=114, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=12996, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_7_rgb1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_rgb1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_7_rgb1_out[1][114][114][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_rgb1_params = {.batch_size=1, .in_dim=114, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=114, .out_dim_pooled=114, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=12996, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_8_rgb1_w[1152][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_rgb1_b[128] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_8_rgb1_out[1][114][114][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_rgb1_params = {.batch_size=1, .in_dim=114, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=128, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=114, .out_dim_pooled=114, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=12996, .K=1152, .J=128, .output_scale=1, .res_scale=1};

static const elem_t conv_9_rgb1_w[1152][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_rgb1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_9_rgb1_out[1][114][114][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_rgb1_params = {.batch_size=1, .in_dim=114, .kernel_size=3, .in_channels=128, .in_stride=192, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=114, .out_dim_pooled=114, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=12996, .K=1152, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_10_rgb1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_rgb1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_10_rgb1_out[1][57][57][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_rgb1_params = {.batch_size=1, .in_dim=57, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=57, .out_dim_pooled=57, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=3249, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_11_rgb1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_rgb1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_11_rgb1_out[1][57][57][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_rgb1_params = {.batch_size=1, .in_dim=57, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=57, .out_dim_pooled=57, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=3249, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_12_rgb1_w[2304][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_rgb1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_12_rgb1_out[1][57][57][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_rgb1_params = {.batch_size=1, .in_dim=57, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=1, .out_dim=57, .out_dim_pooled=57, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=3249, .K=2304, .J=256, .output_scale=1, .res_scale=1};

static const elem_t conv_13_rgb1_w[2304][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_rgb1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_13_rgb1_out[1][57][57][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_rgb1_params = {.batch_size=1, .in_dim=57, .kernel_size=3, .in_channels=256, .in_stride=320, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=57, .out_dim_pooled=57, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=3249, .K=2304, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_14_rgb1_w[4608][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_rgb1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_14_rgb1_out[1][29][29][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_rgb1_params = {.batch_size=1, .in_dim=29, .kernel_size=3, .in_channels=512, .in_stride=576, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=29, .out_dim_pooled=29, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=841, .K=4608, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_15_rgb1_w[4608][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_rgb1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_15_rgb1_out[1][29][29][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_rgb1_params = {.batch_size=1, .in_dim=29, .kernel_size=3, .in_channels=512, .in_stride=576, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=29, .out_dim_pooled=29, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=841, .K=4608, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_16_rgb1_w[4608][576] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_rgb1_b[512] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_16_rgb1_out[1][29][29][576] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_rgb1_params = {.batch_size=1, .in_dim=29, .kernel_size=3, .in_channels=512, .in_stride=576, .out_channels=512, .out_stride=576, .weight_stride=576, .stride=1, .padding=1, .out_dim=29, .out_dim_pooled=29, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=841, .K=4608, .J=512, .output_scale=1, .res_scale=1};

static const elem_t conv_17_rgb1_w[512][320] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_rgb1_b[256] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_17_rgb1_out[1][29][29][320] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_rgb1_params = {.batch_size=1, .in_dim=29, .kernel_size=1, .in_channels=512, .in_stride=576, .out_channels=256, .out_stride=320, .weight_stride=320, .stride=1, .padding=0, .out_dim=29, .out_dim_pooled=29, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=841, .K=512, .J=256, .output_scale=1, .res_scale=1};

#endif
