#ifndef BERTBASE_MT_PARAMS_H
#define BERTBASE_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

// HIDDEN_LAYERS = 12
// SEQ_LEN = 128
// HIDDEN_DIM = 768
// NUM_HEAD = 12
// EXPANSION 3072


static elem_t input_base_0[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_0[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_0[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_0[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_0[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_0[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_0[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_0[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_0[128][(3072+64)] row_align(MAX_BLOCK_LEN);

static const struct BertParams bert_base_params = {.seq_len=128, .hidden_dim=768, .num_head=12, .expansion_dim=3072, .seq_stride=(128+64), .expansion_stride=(3072+64), .hidden_stride=(768+64)};

static elem_t input_base_1[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_1[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_1[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_1[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_1[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_1[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_1[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_1[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_1[128][(3072+64)] row_align(MAX_BLOCK_LEN);

static elem_t input_base_2[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_2[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_2[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_2[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_2[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_2[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_2[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_2[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_2[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_3[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_3[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_3[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_3[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_3[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_3[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_3[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_3[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_3[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_4[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_4[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_4[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_4[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_4[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_4[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_4[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_4[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_4[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_5[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_5[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_5[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_5[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_5[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_5[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_5[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_5[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_5[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_6[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_6[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_6[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_6[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_6[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_6[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_6[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_6[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_6[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_7[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_7[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_7[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_7[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_7[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_7[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_7[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_7[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_7[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_8[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_8[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_8[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_8[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_8[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_8[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_8[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_8[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_8[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_9[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_9[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_9[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_9[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_9[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_9[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_9[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_9[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_9[128][(3072+64)] row_align(MAX_BLOCK_LEN);

/*
static elem_t input_base_10[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_10[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_10[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_10[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_10[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_10[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_10[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_10[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_10[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base_11[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base_11[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base_11[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base_11[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base_11[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base_11[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base_11[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base_11[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base_11[128][(3072+64)] row_align(MAX_BLOCK_LEN);
*/
#if FULL == 1
static elem_t input_base1_0[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_0[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_0[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_0[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_0[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_0[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_0[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_0[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_0[128][(3072+64)] row_align(MAX_BLOCK_LEN);

static const struct BertParams bert_base1_params = {.seq_len=128, .hidden_dim=768, .num_head=12, .expansion_dim=3072, .seq_stride=(128+64), .expansion_stride=(3072+64), .hidden_stride=(768+64)};

static elem_t input_base1_1[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_1[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_1[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_1[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_1[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_1[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_1[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_1[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_1[128][(3072+64)] row_align(MAX_BLOCK_LEN);

static elem_t input_base1_2[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_2[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_2[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_2[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_2[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_2[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_2[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_2[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_2[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_3[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_3[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_3[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_3[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_3[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_3[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_3[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_3[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_3[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_4[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_4[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_4[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_4[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_4[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_4[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_4[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_4[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_4[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_5[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_5[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_5[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_5[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_5[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_5[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_5[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_5[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_5[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_6[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_6[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_6[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_6[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_6[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_6[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_6[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_6[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_6[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_7[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_7[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_7[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_7[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_7[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_7[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_7[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_7[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_7[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_8[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_8[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_8[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_8[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_8[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_8[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_8[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_8[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_8[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_9[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_9[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_9[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_9[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_9[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_9[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_9[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_9[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_9[128][(3072+64)] row_align(MAX_BLOCK_LEN);

/*
static elem_t input_base1_10[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_10[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_10[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_10[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_10[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_10[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_10[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_10[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_10[128][(3072+64)] row_align(MAX_BLOCK_LEN);


static elem_t input_base1_11[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t enc_out_base1_11[128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo_base1_11[4][768][(768+64)] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w_base1_11[2][768*3072] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b_base1_11[3072] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b_base1_11[768] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf_base1_11[3][128][(768+64)] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf_base1_11[12][128][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t out_buf_base1_11[128][(3072+64)] row_align(MAX_BLOCK_LEN);
*/

#endif

#endif
