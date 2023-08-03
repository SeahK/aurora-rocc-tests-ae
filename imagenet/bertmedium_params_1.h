#ifndef BERTMEDIUM_MT_PARAMS_H
#define BERTMEDIUM_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

// HIDDEN_LAYERS = 12
// SEQ_LEN = 128
// HIDDEN_DIM = 512
// NUM_HEAD = 12
// EXPANSION 2048


static elem_t input_medium_0[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_0[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_0[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_0[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_0[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_0[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_0[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_0[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_0[128][(2048+64)] row_align(MAX_BLOCK_LEN);

static const struct BertParams bert_medium_params = {.seq_len=128, .hidden_dim=512, .num_head=8, .expansion_dim=2048, .seq_stride=(128+64), .expansion_stride=(2048+64), .hidden_stride=(512+64)};

static elem_t input_medium_1[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_1[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_1[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_1[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_1[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_1[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_1[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_1[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_1[128][(2048+64)] row_align(MAX_BLOCK_LEN);

static elem_t input_medium_2[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_2[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_2[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_2[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_2[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_2[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_2[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_2[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_2[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium_3[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_3[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_3[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_3[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_3[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_3[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_3[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_3[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_3[128][(2048+64)] row_align(MAX_BLOCK_LEN);

/*
static elem_t input_medium_4[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_4[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_4[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_4[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_4[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_4[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_4[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_4[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_4[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium_5[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_5[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_5[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_5[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_5[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_5[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_5[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_5[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_5[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium_6[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_6[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_6[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_6[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_6[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_6[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_6[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_6[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_6[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium_7[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium_7[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium_7[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium_7[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium_7[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium_7[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium_7[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium_7[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium_7[128][(2048+64)] row_align(MAX_BLOCK_LEN);
*/
static elem_t input_medium1_0[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_0[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_0[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_0[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_0[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_0[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_0[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_0[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_0[128][(2048+64)] row_align(MAX_BLOCK_LEN);

static const struct BertParams bert_medium1_params = {.seq_len=128, .hidden_dim=512, .num_head=8, .expansion_dim=2048, .seq_stride=(128+64), .expansion_stride=(2048+64), .hidden_stride=(512+64)};

static elem_t input_medium1_1[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_1[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_1[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_1[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_1[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_1[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_1[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_1[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_1[128][(2048+64)] row_align(MAX_BLOCK_LEN);

static elem_t input_medium1_2[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_2[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_2[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_2[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_2[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_2[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_2[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_2[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_2[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium1_3[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_3[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_3[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_3[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_3[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_3[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_3[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_3[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_3[128][(2048+64)] row_align(MAX_BLOCK_LEN);

/*
static elem_t input_medium1_4[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_4[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_4[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_4[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_4[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_4[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_4[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_4[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_4[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium1_5[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_5[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_5[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_5[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_5[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_5[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_5[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_5[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_5[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium1_6[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_6[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_6[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_6[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_6[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_6[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_6[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_6[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_6[128][(2048+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_medium1_7[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_medium1_7[128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_medium1_7[4][512][(512+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_medium1_7[2][512*2048] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_medium1_7[2048] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_medium1_7[512] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_medium1_7[3][128][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_medium1_7[8][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_medium1_7[128][(2048+64)] row_align(MAX_BLOCK_LEN);
*/


#endif
