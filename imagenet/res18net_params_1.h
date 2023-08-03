#ifndef RES18NET_MT_PARAMS_H
#define RES18NET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w_r18es[147][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_r18es[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_in_r18es[12544][147] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_r18es[12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_r18es_pooled[1][56][56][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_r18es = {.batch_size=1, .in_dim=224, .kernel_size=7, .in_channels=3, .out_channels=64, .out_stride=64, .stride=2, .padding=3, .bias=1, .depthwise=0, .out_dim=112, .n_patches=12544, .patch_size=147, .pool_size=3, .pool_stride=2, .pool_padding=1, .out_dim_pooled=56, .output_scale=8, .I=12544, .J=64, .K=147, .res_scale=0};


static const elem_t conv_2_w_r18es[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_r18es[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_in_r18es[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_r18es[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_r18es = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=8, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_3_w_r18es[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_r18es[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_in_r18es[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_r18es[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_r18es = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_4_w_r18es[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_r18es[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_in_r18es[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_r18es[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_r18es = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_5_w_r18es[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_r18es[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_in_r18es[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_r18es[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_r18es = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=8, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_6_w_r18es[576][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_r18es[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_in_r18es[784][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_r18es[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_r18es = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=576, .res_scale=0};


static const elem_t conv_7_w_r18es[1152][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_r18es[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_in_r18es[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_r18es[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_r18es = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=9, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_8_w_r18es[64][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_8_b_r18es[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_8_in_r18es[784][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_r18es[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_8_params_r18es = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=128, .K=64, .res_scale=0};


static const elem_t conv_9_w_r18es[1152][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_9_b_r18es[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_9_in_r18es[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out_r18es[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_9_params_r18es = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_10_w_r18es[1152][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_10_b_r18es[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_10_in_r18es[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out_r18es[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_10_params_r18es = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_11_w_r18es[1152][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_11_b_r18es[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_11_in_r18es[196][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out_r18es[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_11_params_r18es = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=9, .I=196, .J=256, .K=1152, .res_scale=0};


static const elem_t conv_12_w_r18es[2304][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_12_b_r18es[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_12_in_r18es[196][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_r18es[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_12_params_r18es = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_13_w_r18es[128][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_13_b_r18es[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_13_in_r18es[196][128+64] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_r18es[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_13_params_r18es = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=128, .res_scale=0};


static const elem_t conv_14_w_r18es[2304][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_14_b_r18es[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_14_in_r18es[196][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out_r18es[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_14_params_r18es = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_15_w_r18es[2304][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_15_b_r18es[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_15_in_r18es[196][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out_r18es[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_15_params_r18es = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_16_w_r18es[2304][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_16_b_r18es[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_16_in_r18es[49][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out_r18es[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_16_params_r18es = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=2304, .res_scale=0};


static const elem_t conv_17_w_r18es[4608][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_17_b_r18es[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_17_in_r18es[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out_r18es[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_17_params_r18es = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_18_w_r18es[256][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_18_b_r18es[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_18_in_r18es[49][256+64] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out_r18es[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_18_params_r18es = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=512, .K=256, .res_scale=0};


static const elem_t conv_19_w_r18es[4608][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_19_b_r18es[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_19_in_r18es[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out_r18es[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_19_params_r18es = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_20_w_r18es[4608][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_20_b_r18es[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_20_in_r18es[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out_r18es[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_20_params_r18es = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t fc_21_w_r18es[512][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_21_b_r18es[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC);
static elem_t fc_21_out_r18es[1][1024] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_21_params_r18es = {.batch_size=1, .in_features=512, .out_features=1024, .out_stride=(1024+64), .bias=1, .output_scale=9, .J=1024, .I=1, .K=512};

static const elem_t conv_1_w_r18es11[147][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_r18es11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_in_r18es11[12544][147] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_r18es11[12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_r18es11_pooled[1][56][56][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_r18es11 = {.batch_size=1, .in_dim=224, .kernel_size=7, .in_channels=3, .out_channels=64, .out_stride=64, .stride=2, .padding=3, .bias=1, .depthwise=0, .out_dim=112, .n_patches=12544, .patch_size=147, .pool_size=3, .pool_stride=2, .pool_padding=1, .out_dim_pooled=56, .output_scale=8, .I=12544, .J=64, .K=147, .res_scale=0};


static const elem_t conv_2_w_r18es11[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_r18es11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_in_r18es11[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_r18es11[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_r18es11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=8, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_3_w_r18es11[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_r18es11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_in_r18es11[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_r18es11[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_r18es11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_4_w_r18es11[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_r18es11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_in_r18es11[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_r18es11[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_r18es11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_5_w_r18es11[576][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_r18es11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_in_r18es11[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_r18es11[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_r18es11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=8, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_6_w_r18es11[576][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_r18es11[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_in_r18es11[784][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_r18es11[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_r18es11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=576, .res_scale=0};


static const elem_t conv_7_w_r18es11[1152][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_r18es11[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_in_r18es11[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_r18es11[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_r18es11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=9, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_8_w_r18es11[64][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_8_b_r18es11[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_8_in_r18es11[784][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_r18es11[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_8_params_r18es11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=128, .K=64, .res_scale=0};


static const elem_t conv_9_w_r18es11[1152][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_9_b_r18es11[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_9_in_r18es11[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out_r18es11[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_9_params_r18es11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_10_w_r18es11[1152][128+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_10_b_r18es11[128+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_10_in_r18es11[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out_r18es11[784][128+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_10_params_r18es11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_11_w_r18es11[1152][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_11_b_r18es11[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_11_in_r18es11[196][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out_r18es11[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_11_params_r18es11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=9, .I=196, .J=256, .K=1152, .res_scale=0};


static const elem_t conv_12_w_r18es11[2304][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_12_b_r18es11[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_12_in_r18es11[196][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_r18es11[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_12_params_r18es11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_13_w_r18es11[128][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_13_b_r18es11[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_13_in_r18es11[196][128+64] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_r18es11[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_13_params_r18es11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=128, .res_scale=0};


static const elem_t conv_14_w_r18es11[2304][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_14_b_r18es11[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_14_in_r18es11[196][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out_r18es11[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_14_params_r18es11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_15_w_r18es11[2304][256+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_15_b_r18es11[256+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_15_in_r18es11[196][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out_r18es11[196][256+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_15_params_r18es11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_16_w_r18es11[2304][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_16_b_r18es11[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_16_in_r18es11[49][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out_r18es11[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_16_params_r18es11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=2304, .res_scale=0};


static const elem_t conv_17_w_r18es11[4608][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_17_b_r18es11[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_17_in_r18es11[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out_r18es11[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_17_params_r18es11 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_18_w_r18es11[256][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_18_b_r18es11[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_18_in_r18es11[49][256+64] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out_r18es11[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_18_params_r18es11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=512, .K=256, .res_scale=0};


static const elem_t conv_19_w_r18es11[4608][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_19_b_r18es11[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_19_in_r18es11[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out_r18es11[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_19_params_r18es11 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_20_w_r18es11[4608][512+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_20_b_r18es11[512+64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_20_in_r18es11[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out_r18es11[49][512+64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_20_params_r18es11 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t fc_21_w_r18es11[512][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_21_b_r18es11[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC);
static elem_t fc_21_out_r18es11[1][1024] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_21_params_r18es11 = {.batch_size=1, .in_features=512, .out_features=1024, .out_stride=(1024+64), .bias=1, .output_scale=9, .J=1024, .I=1, .K=512};


#endif

