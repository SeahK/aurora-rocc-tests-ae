#ifndef fbnet_PARAMS_H
#define fbnet_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_0_fb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_0_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_0_fb1_out[1][160][160][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_0_fb1_params = {.batch_size=1, .in_dim=160, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=160, .out_dim_pooled=160, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_1_fb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_1_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_1_fb1_out[1][80][80][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_fb1_params = {.batch_size=1, .in_dim=80, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_2_fb1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_2_fb1_out[1][80][80][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_fb1_params = {.batch_size=1, .in_dim=80, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_3_fb1_w[64][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_fb1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_3_fb1_out[1][80][80][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_fb1_params = {.batch_size=1, .in_dim=80, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=0, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=64, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_4_fb1_w[864][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_fb1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_4_fb1_out[1][80][80][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_fb1_params = {.batch_size=1, .in_dim=80, .kernel_size=3, .in_channels=96, .in_stride=96, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=1, .out_dim=80, .out_dim_pooled=80, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=6400, .K=864, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_5_fb1_w[96][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_5_fb1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_fb1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=96, .in_stride=96, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=96, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_6_fb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_6_fb1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_fb1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_7_fb1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_7_fb1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_fb1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_8_fb1_w[576][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_8_fb1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_fb1_params = {.batch_size=1, .in_dim=40, .kernel_size=3, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=1, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=576, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_9_fb1_w[64][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_9_fb1_out[1][40][40][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_fb1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_10_fb1_w[64][144] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_fb1_b[144] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_10_fb1_out[1][40][40][144] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_fb1_params = {.batch_size=1, .in_dim=40, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=144, .out_stride=144, .weight_stride=144, .stride=1, .padding=0, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=64, .J=144, .output_scale=1, .res_scale=1};

static const elem_t conv_11_fb1_w[3600][144] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_fb1_b[144] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_11_fb1_out[1][40][40][144] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_fb1_params = {.batch_size=1, .in_dim=40, .kernel_size=5, .in_channels=144, .in_stride=144, .out_channels=144, .out_stride=144, .weight_stride=144, .stride=1, .padding=2, .out_dim=40, .out_dim_pooled=40, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1600, .K=3600, .J=144, .output_scale=1, .res_scale=1};

static const elem_t conv_12_fb1_w[144][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_12_fb1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=144, .in_stride=144, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=144, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_13_fb1_w[64][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_fb1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_13_fb1_out[1][20][20][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=64, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_14_fb1_w[2400][96] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_fb1_b[96] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_14_fb1_out[1][20][20][96] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=5, .in_channels=96, .in_stride=96, .out_channels=96, .out_stride=96, .weight_stride=96, .stride=1, .padding=2, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=2400, .J=96, .output_scale=1, .res_scale=1};

static const elem_t conv_15_fb1_w[96][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_15_fb1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=96, .in_stride=96, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=96, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_16_fb1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_16_fb1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_17_fb1_w[4800][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_17_fb1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=5, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=2, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=4800, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_18_fb1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_18_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_18_fb1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_19_fb1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_19_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_19_fb1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_20_fb1_w[1728][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_20_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_20_fb1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=3, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=1, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=1728, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_21_fb1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_21_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_21_fb1_out[1][20][20][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_22_fb1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_22_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_22_fb1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_23_fb1_w[4800][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_23_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_23_fb1_out[1][20][20][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_fb1_params = {.batch_size=1, .in_dim=20, .kernel_size=5, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=2, .out_dim=20, .out_dim_pooled=20, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=400, .K=4800, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_24_fb1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_24_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_24_fb1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_25_fb1_w[64][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_25_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_25_fb1_out[1][10][10][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=64, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_26_fb1_w[4800][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_26_fb1_b[192] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_26_fb1_out[1][10][10][192] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_26_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=192, .in_stride=192, .out_channels=192, .out_stride=192, .weight_stride=192, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=4800, .J=192, .output_scale=1, .res_scale=1};

static const elem_t conv_27_fb1_w[192][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_27_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_27_fb1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_27_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=192, .in_stride=192, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=192, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_28_fb1_w[64][448] row_align(MAX_BLOCK_LEN);
static const acc_t conv_28_fb1_b[384] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_28_fb1_out[1][10][10][448] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_28_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=384, .out_stride=448, .weight_stride=448, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=64, .J=384, .output_scale=1, .res_scale=1};

static const elem_t conv_29_fb1_w[9600][448] row_align(MAX_BLOCK_LEN);
static const acc_t conv_29_fb1_b[384] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_29_fb1_out[1][10][10][448] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_29_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=384, .in_stride=448, .out_channels=384, .out_stride=448, .weight_stride=448, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=9600, .J=384, .output_scale=1, .res_scale=1};

static const elem_t conv_30_fb1_w[384][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_30_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_30_fb1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_30_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=384, .in_stride=448, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=384, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_31_fb1_w[64][448] row_align(MAX_BLOCK_LEN);
static const acc_t conv_31_fb1_b[384] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_31_fb1_out[1][10][10][448] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_31_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=384, .out_stride=448, .weight_stride=448, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=64, .J=384, .output_scale=1, .res_scale=1};

static const elem_t conv_32_fb1_w[9600][448] row_align(MAX_BLOCK_LEN);
static const acc_t conv_32_fb1_b[384] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_32_fb1_out[1][10][10][448] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_32_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=384, .in_stride=448, .out_channels=384, .out_stride=448, .weight_stride=448, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=9600, .J=384, .output_scale=1, .res_scale=1};

static const elem_t conv_33_fb1_w[384][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_33_fb1_b[64] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_33_fb1_out[1][10][10][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_33_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=384, .in_stride=448, .out_channels=64, .out_stride=64, .weight_stride=64, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=384, .J=64, .output_scale=1, .res_scale=1};

static const elem_t conv_34_fb1_w[64][448] row_align(MAX_BLOCK_LEN);
static const acc_t conv_34_fb1_b[384] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_34_fb1_out[1][10][10][448] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_34_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=64, .in_stride=64, .out_channels=384, .out_stride=448, .weight_stride=448, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=64, .J=384, .output_scale=1, .res_scale=1};

static const elem_t conv_35_fb1_w[9600][448] row_align(MAX_BLOCK_LEN);
static const acc_t conv_35_fb1_b[384] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_35_fb1_out[1][10][10][448] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_35_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=384, .in_stride=448, .out_channels=384, .out_stride=448, .weight_stride=448, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=9600, .J=384, .output_scale=1, .res_scale=1};

static const elem_t conv_36_fb1_w[384][112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_36_fb1_b[112] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_36_fb1_out[1][10][10][112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_36_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=384, .in_stride=448, .out_channels=112, .out_stride=112, .weight_stride=112, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=384, .J=112, .output_scale=1, .res_scale=1};

static const elem_t conv_37_fb1_w[112][672] row_align(MAX_BLOCK_LEN);
static const acc_t conv_37_fb1_b[672] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_37_fb1_out[1][10][10][672] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_37_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=112, .in_stride=112, .out_channels=672, .out_stride=672, .weight_stride=672, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=112, .J=672, .output_scale=1, .res_scale=1};

static const elem_t conv_38_fb1_w[16800][672] row_align(MAX_BLOCK_LEN);
static const acc_t conv_38_fb1_b[672] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_38_fb1_out[1][10][10][672] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_38_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=672, .in_stride=672, .out_channels=672, .out_stride=672, .weight_stride=672, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=16800, .J=672, .output_scale=1, .res_scale=1};

static const elem_t conv_39_fb1_w[672][112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_39_fb1_b[112] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_39_fb1_out[1][10][10][112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_39_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=672, .in_stride=672, .out_channels=112, .out_stride=112, .weight_stride=112, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=672, .J=112, .output_scale=1, .res_scale=1};

static const elem_t conv_40_fb1_w[112][672] row_align(MAX_BLOCK_LEN);
static const acc_t conv_40_fb1_b[672] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_40_fb1_out[1][10][10][672] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_40_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=112, .in_stride=112, .out_channels=672, .out_stride=672, .weight_stride=672, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=112, .J=672, .output_scale=1, .res_scale=1};

static const elem_t conv_41_fb1_w[16800][672] row_align(MAX_BLOCK_LEN);
static const acc_t conv_41_fb1_b[672] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_41_fb1_out[1][10][10][672] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_41_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=672, .in_stride=672, .out_channels=672, .out_stride=672, .weight_stride=672, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=16800, .J=672, .output_scale=1, .res_scale=1};

static const elem_t conv_42_fb1_w[672][112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_42_fb1_b[112] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_42_fb1_out[1][10][10][112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_42_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=672, .in_stride=672, .out_channels=112, .out_stride=112, .weight_stride=112, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=672, .J=112, .output_scale=1, .res_scale=1};

static const elem_t conv_43_fb1_w[112][336] row_align(MAX_BLOCK_LEN);
static const acc_t conv_43_fb1_b[336] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_43_fb1_out[1][10][10][336] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_43_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=112, .in_stride=112, .out_channels=336, .out_stride=336, .weight_stride=336, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=112, .J=336, .output_scale=1, .res_scale=1};

static const elem_t conv_44_fb1_w[8400][336] row_align(MAX_BLOCK_LEN);
static const acc_t conv_44_fb1_b[336] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_44_fb1_out[1][10][10][336] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_44_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=336, .in_stride=336, .out_channels=336, .out_stride=336, .weight_stride=336, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=8400, .J=336, .output_scale=1, .res_scale=1};

static const elem_t conv_45_fb1_w[336][112] row_align(MAX_BLOCK_LEN);
static const acc_t conv_45_fb1_b[112] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_45_fb1_out[1][10][10][112] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_45_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=336, .in_stride=336, .out_channels=112, .out_stride=112, .weight_stride=112, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=336, .J=112, .output_scale=1, .res_scale=1};

static const elem_t conv_46_fb1_w[112][672] row_align(MAX_BLOCK_LEN);
static const acc_t conv_46_fb1_b[672] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_46_fb1_out[1][10][10][672] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_46_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=112, .in_stride=112, .out_channels=672, .out_stride=672, .weight_stride=672, .stride=1, .padding=0, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=112, .J=672, .output_scale=1, .res_scale=1};

static const elem_t conv_47_fb1_w[16800][672] row_align(MAX_BLOCK_LEN);
static const acc_t conv_47_fb1_b[672] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_47_fb1_out[1][10][10][672] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_47_fb1_params = {.batch_size=1, .in_dim=10, .kernel_size=5, .in_channels=672, .in_stride=672, .out_channels=672, .out_stride=672, .weight_stride=672, .stride=1, .padding=2, .out_dim=10, .out_dim_pooled=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=100, .K=16800, .J=672, .output_scale=1, .res_scale=1};

static const elem_t conv_48_fb1_w[672][184] row_align(MAX_BLOCK_LEN);
static const acc_t conv_48_fb1_b[184] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_48_fb1_out[1][5][5][184] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_48_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=672, .in_stride=672, .out_channels=184, .out_stride=184, .weight_stride=184, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=672, .J=184, .output_scale=1, .res_scale=1};

static const elem_t conv_49_fb1_w[184][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_49_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_49_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_49_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=184, .in_stride=184, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=184, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_50_fb1_w[27600][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_50_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_50_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_50_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=5, .in_channels=1104, .in_stride=1104, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=2, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=27600, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_51_fb1_w[1104][184] row_align(MAX_BLOCK_LEN);
static const acc_t conv_51_fb1_b[184] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_51_fb1_out[1][5][5][184] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_51_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=1104, .in_stride=1104, .out_channels=184, .out_stride=184, .weight_stride=184, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=1104, .J=184, .output_scale=1, .res_scale=1};

static const elem_t conv_52_fb1_w[184][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_52_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_52_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_52_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=184, .in_stride=184, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=184, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_53_fb1_w[27600][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_53_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_53_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_53_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=5, .in_channels=1104, .in_stride=1104, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=2, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=27600, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_54_fb1_w[1104][184] row_align(MAX_BLOCK_LEN);
static const acc_t conv_54_fb1_b[184] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_54_fb1_out[1][5][5][184] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_54_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=1104, .in_stride=1104, .out_channels=184, .out_stride=184, .weight_stride=184, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=1104, .J=184, .output_scale=1, .res_scale=1};

static const elem_t conv_55_fb1_w[184][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_55_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_55_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_55_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=184, .in_stride=184, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=184, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_56_fb1_w[27600][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_56_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_56_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_56_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=5, .in_channels=1104, .in_stride=1104, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=2, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=27600, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_57_fb1_w[1104][184] row_align(MAX_BLOCK_LEN);
static const acc_t conv_57_fb1_b[184] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_57_fb1_out[1][5][5][184] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_57_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=1104, .in_stride=1104, .out_channels=184, .out_stride=184, .weight_stride=184, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=1104, .J=184, .output_scale=1, .res_scale=1};

static const elem_t conv_58_fb1_w[184][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_58_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_58_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_58_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=184, .in_stride=184, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=184, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_59_fb1_w[9936][1104] row_align(MAX_BLOCK_LEN);
static const acc_t conv_59_fb1_b[1104] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_59_fb1_out[1][5][5][1104] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_59_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=3, .in_channels=1104, .in_stride=1104, .out_channels=1104, .out_stride=1104, .weight_stride=1104, .stride=1, .padding=1, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=9936, .J=1104, .output_scale=1, .res_scale=1};

static const elem_t conv_60_fb1_w[1104][352] row_align(MAX_BLOCK_LEN);
static const acc_t conv_60_fb1_b[352] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_60_fb1_out[1][5][5][352] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_60_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=1104, .in_stride=1104, .out_channels=352, .out_stride=352, .weight_stride=352, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=1104, .J=352, .output_scale=1, .res_scale=1};

static const elem_t conv_61_fb1_w[352][1984] row_align(MAX_BLOCK_LEN);
static const acc_t conv_61_fb1_b[1984] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_61_fb1_out[1][5][5][1984] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_61_fb1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=352, .in_stride=352, .out_channels=1984, .out_stride=1984, .weight_stride=1984, .stride=1, .padding=0, .out_dim=5, .out_dim_pooled=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=25, .K=352, .J=1984, .output_scale=1, .res_scale=1};

static const elem_t conv_62_fb1_w[1984][1000] row_align(MAX_BLOCK_LEN);
static const acc_t conv_62_fb1_b[1000] row_align(MAX_BLOCK_LEN_ACC); 
static elem_t conv_62_fb1_out[1][1][1][1000] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_62_fb1_params = {.batch_size=1, .in_dim=1, .kernel_size=1, .in_channels=1984, .in_stride=1984, .out_channels=1000, .out_stride=1000, .weight_stride=1000, .stride=1, .padding=0, .out_dim=1, .out_dim_pooled=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .I=1, .K=1984, .J=1000, .output_scale=1, .res_scale=1};

#endif
