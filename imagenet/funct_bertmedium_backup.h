#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#ifndef DEBUG
#include "bertmedium_params_1.h"

#define THREAD_SYNC 0

uint64_t* bertmedium_function_1(int block, bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, int num_array, int cid){
#define num_cycle (30+2+3)
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif        

    int hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    static uint64_t cycles[num_cycle];
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_resadd_cycles = 0;
    uint64_t matmul_cycles[30];
    uint64_t resadd_cycles[2];
   
    if(block == -1 || block == 0){
    // layer 0
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_0, (elem_t*) Wqkvo_medium_0[0], NULL, (elem_t*) QKV_buf_medium_0[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] = end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_0, (elem_t*) Wqkvo_medium_0[1], NULL, (elem_t*) QKV_buf_medium_0[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] = end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_0, (elem_t*) Wqkvo_medium_0[2], NULL, (elem_t*) QKV_buf_medium_0[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_0[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_0[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_0[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] = end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_0[head];
      elem_t * B = (elem_t*) QKV_buf_medium_0[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_0 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] = end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_0, (elem_t*) Wqkvo_medium_0[3], NULL, (elem_t*) enc_out_medium_0,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_0,
        (elem_t*) enc_out_medium_0,
        (elem_t*) input_medium_0,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    }
    if(block == -1 || block == 1){
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_0, (elem_t*) ff_w_medium_0[0], (acc_t*) ff1_b_medium_0, (elem_t*) out_buf_medium_0,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_0, (elem_t*) ff_w_medium_0[1], (acc_t*) ff2_b_medium_0, (elem_t*) enc_out_medium_0,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_0,
        (elem_t*) enc_out_medium_0,
        (elem_t*) input_medium_1,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    if(block == -1 || block == 2){
    // layer 1
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_1, (elem_t*) Wqkvo_medium_1[0], NULL, (elem_t*) QKV_buf_medium_1[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_1, (elem_t*) Wqkvo_medium_1[1], NULL, (elem_t*) QKV_buf_medium_1[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_1, (elem_t*) Wqkvo_medium_1[2], NULL, (elem_t*) QKV_buf_medium_1[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_1[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_1[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_1[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_1[head];
      elem_t * B = (elem_t*) QKV_buf_medium_1[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_1 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_1, (elem_t*) Wqkvo_medium_1[3], NULL, (elem_t*) enc_out_medium_1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_1,
        (elem_t*) enc_out_medium_1,
        (elem_t*) input_medium_1,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_1, (elem_t*) ff_w_medium_1[0], (acc_t*) ff1_b_medium_1, (elem_t*) out_buf_medium_1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    }
    if(block == -1 || block == 3){
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_1, (elem_t*) ff_w_medium_1[1], (acc_t*) ff2_b_medium_1, (elem_t*) enc_out_medium_1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_1,
        (elem_t*) enc_out_medium_1,
        (elem_t*) input_medium_2,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    if(block == -1 || block == 4){
    // layer 2
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_2, (elem_t*) Wqkvo_medium_2[0], NULL, (elem_t*) QKV_buf_medium_2[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_2, (elem_t*) Wqkvo_medium_2[1], NULL, (elem_t*) QKV_buf_medium_2[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_2, (elem_t*) Wqkvo_medium_2[2], NULL, (elem_t*) QKV_buf_medium_2[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_2[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_2[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_2[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_2[head];
      elem_t * B = (elem_t*) QKV_buf_medium_2[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_2 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_2, (elem_t*) Wqkvo_medium_2[3], NULL, (elem_t*) enc_out_medium_2,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_2,
        (elem_t*) enc_out_medium_2,
        (elem_t*) input_medium_2,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    }
    if(block == -1 || block == 5){
    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_2, (elem_t*) ff_w_medium_2[0], (acc_t*) ff1_b_medium_2, (elem_t*) out_buf_medium_2,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_2, (elem_t*) ff_w_medium_2[1], (acc_t*) ff2_b_medium_2, (elem_t*) enc_out_medium_2,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_2,
        (elem_t*) enc_out_medium_2,
        (elem_t*) input_medium_3,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    if(block == -1 || block == 6){
    // layer 3
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_3, (elem_t*) Wqkvo_medium_3[0], NULL, (elem_t*) QKV_buf_medium_3[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_3, (elem_t*) Wqkvo_medium_3[1], NULL, (elem_t*) QKV_buf_medium_3[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_3, (elem_t*) Wqkvo_medium_3[2], NULL, (elem_t*) QKV_buf_medium_3[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_3[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_3[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_3[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_3[head];
      elem_t * B = (elem_t*) QKV_buf_medium_3[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_3 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_3, (elem_t*) Wqkvo_medium_3[3], NULL, (elem_t*) enc_out_medium_3,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_3,
        (elem_t*) enc_out_medium_3,
        (elem_t*) input_medium_3,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    }
    if(block == -1 || block == 7){
    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_3, (elem_t*) ff_w_medium_3[0], (acc_t*) ff1_b_medium_3, (elem_t*) out_buf_medium_3,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_3, (elem_t*) ff_w_medium_3[1], (acc_t*) ff2_b_medium_3, (elem_t*) enc_out_medium_3,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_3,
        (elem_t*) enc_out_medium_3,
        (elem_t*) input_medium_3,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    /*
    // layer 4
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_4, (elem_t*) Wqkvo_medium_4[0], NULL, (elem_t*) QKV_buf_medium_4[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_4, (elem_t*) Wqkvo_medium_4[1], NULL, (elem_t*) QKV_buf_medium_4[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_4, (elem_t*) Wqkvo_medium_4[2], NULL, (elem_t*) QKV_buf_medium_4[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_4[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_4[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_4[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_4[head];
      elem_t * B = (elem_t*) QKV_buf_medium_4[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_4 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_4, (elem_t*) Wqkvo_medium_4[3], NULL, (elem_t*) enc_out_medium_4,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_4,
        (elem_t*) enc_out_medium_4,
        (elem_t*) input_medium_4,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_4, (elem_t*) ff_w_medium_4[0], (acc_t*) ff1_b_medium_4, (elem_t*) out_buf_medium_4,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_4, (elem_t*) ff_w_medium_4[1], (acc_t*) ff2_b_medium_4, (elem_t*) enc_out_medium_4,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_4,
        (elem_t*) enc_out_medium_4,
        (elem_t*) input_medium_5,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");


    // layer 5
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_5, (elem_t*) Wqkvo_medium_5[0], NULL, (elem_t*) QKV_buf_medium_5[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_5, (elem_t*) Wqkvo_medium_5[1], NULL, (elem_t*) QKV_buf_medium_5[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_5, (elem_t*) Wqkvo_medium_5[2], NULL, (elem_t*) QKV_buf_medium_5[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_5[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_5[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_5[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_5[head];
      elem_t * B = (elem_t*) QKV_buf_medium_5[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_5 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_5, (elem_t*) Wqkvo_medium_5[3], NULL, (elem_t*) enc_out_medium_5,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_5,
        (elem_t*) enc_out_medium_5,
        (elem_t*) input_medium_5,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_5, (elem_t*) ff_w_medium_5[0], (acc_t*) ff1_b_medium_5, (elem_t*) out_buf_medium_5,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_5, (elem_t*) ff_w_medium_5[1], (acc_t*) ff2_b_medium_5, (elem_t*) enc_out_medium_5,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_5,
        (elem_t*) enc_out_medium_5,
        (elem_t*) input_medium_6,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");


    // layer 6
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_6, (elem_t*) Wqkvo_medium_6[0], NULL, (elem_t*) QKV_buf_medium_6[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_6, (elem_t*) Wqkvo_medium_6[1], NULL, (elem_t*) QKV_buf_medium_6[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_6, (elem_t*) Wqkvo_medium_6[2], NULL, (elem_t*) QKV_buf_medium_6[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_6[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_6[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_6[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_6[head];
      elem_t * B = (elem_t*) QKV_buf_medium_6[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_6 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_6, (elem_t*) Wqkvo_medium_6[3], NULL, (elem_t*) enc_out_medium_6,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_6,
        (elem_t*) enc_out_medium_6,
        (elem_t*) input_medium_6,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_6, (elem_t*) ff_w_medium_6[0], (acc_t*) ff1_b_medium_6, (elem_t*) out_buf_medium_6,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_6, (elem_t*) ff_w_medium_6[1], (acc_t*) ff2_b_medium_6, (elem_t*) enc_out_medium_6,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_6,
        (elem_t*) enc_out_medium_6,
        (elem_t*) input_medium_7,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");

    // layer 7
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_7, (elem_t*) Wqkvo_medium_7[0], NULL, (elem_t*) QKV_buf_medium_7[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_7, (elem_t*) Wqkvo_medium_7[1], NULL, (elem_t*) QKV_buf_medium_7[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_7, (elem_t*) Wqkvo_medium_7[2], NULL, (elem_t*) QKV_buf_medium_7[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium_7[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium_7[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium_7[head];
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.seq_len, bert_medium_params.hidden_dim,
          bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium_7[head];
      elem_t * B = (elem_t*) QKV_buf_medium_7[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium_7 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium_params.seq_len, hidden_dim_per_head, bert_medium_params.seq_len,
          bert_medium_params.seq_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.hidden_stride, 0, bert_medium_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium_7, (elem_t*) Wqkvo_medium_7[3], NULL, (elem_t*) enc_out_medium_7,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_7,
        (elem_t*) enc_out_medium_7,
        (elem_t*) input_medium_7,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.expansion_dim, bert_medium_params.hidden_dim,
        bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium_7, (elem_t*) ff_w_medium_7[0], (acc_t*) ff1_b_medium_7, (elem_t*) out_buf_medium_7,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim, bert_medium_params.expansion_dim,
        bert_medium_params.expansion_stride, bert_medium_params.hidden_stride, bert_medium_params.expansion_stride, bert_medium_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium_7, (elem_t*) ff_w_medium_7[1], (acc_t*) ff2_b_medium_7, (elem_t*) enc_out_medium_7,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium_params.seq_len, bert_medium_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium_7,
        (elem_t*) enc_out_medium_7,
        (elem_t*) input_medium_7,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
*/


    for(int i = 0; i < num_cycle; i++){
      if(i < 30){
        cycles[i] = matmul_cycles[i];
      }
      else if (i < 32){
        cycles[i] = resadd_cycles[i - 30];
      }
      else{
        if(i == 32) cycles[i] = total_matmul_cycles;
        if(i == 33) cycles[i] = total_resadd_cycles;
        if(i == 34) cycles[i] = total_matmul_cycles + total_resadd_cycles;
      }
    }
    return cycles;
#undef num_cycle
}

uint64_t* bertmedium_function_11(int block, bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, int num_array, int cid){
#define num_cycle (30+2+3)
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif        

    int hidden_dim_per_head = bert_medium_params.hidden_dim / bert_medium_params.num_head;
    
    static uint64_t cycles[num_cycle];
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_resadd_cycles = 0;
    uint64_t matmul_cycles[30];
    uint64_t resadd_cycles[2];
   
    if(block == -1 || block == 0){
    // layer 0
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_0, (elem_t*) Wqkvo_medium1_0[0], NULL, (elem_t*) QKV_buf_medium1_0[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] = end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_0, (elem_t*) Wqkvo_medium1_0[1], NULL, (elem_t*) QKV_buf_medium1_0[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] = end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_0, (elem_t*) Wqkvo_medium1_0[2], NULL, (elem_t*) QKV_buf_medium1_0[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_0[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_0[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_0[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] = end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_0[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_0[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_0 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] = end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_0, (elem_t*) Wqkvo_medium1_0[3], NULL, (elem_t*) enc_out_medium1_0,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_0,
        (elem_t*) enc_out_medium1_0,
        (elem_t*) input_medium1_0,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    }
    if(block == -1 || block == 1){
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_0, (elem_t*) ff_w_medium1_0[0], (acc_t*) ff1_b_medium1_0, (elem_t*) out_buf_medium1_0,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_0, (elem_t*) ff_w_medium1_0[1], (acc_t*) ff2_b_medium1_0, (elem_t*) enc_out_medium1_0,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_0,
        (elem_t*) enc_out_medium1_0,
        (elem_t*) input_medium1_1,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    if(block == -1 || block == 2){
    // layer 1
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_1, (elem_t*) Wqkvo_medium1_1[0], NULL, (elem_t*) QKV_buf_medium1_1[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_1, (elem_t*) Wqkvo_medium1_1[1], NULL, (elem_t*) QKV_buf_medium1_1[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_1, (elem_t*) Wqkvo_medium1_1[2], NULL, (elem_t*) QKV_buf_medium1_1[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium1_params.hidden_dim / bert_medium1_params.num_head;
    
    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_1[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_1[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_1[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_1[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_1[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_1 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_1, (elem_t*) Wqkvo_medium1_1[3], NULL, (elem_t*) enc_out_medium1_1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_1,
        (elem_t*) enc_out_medium1_1,
        (elem_t*) input_medium1_1,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_1, (elem_t*) ff_w_medium1_1[0], (acc_t*) ff1_b_medium1_1, (elem_t*) out_buf_medium1_1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    }
    if(block == -1 || block == 3){
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_1, (elem_t*) ff_w_medium1_1[1], (acc_t*) ff2_b_medium1_1, (elem_t*) enc_out_medium1_1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_1,
        (elem_t*) enc_out_medium1_1,
        (elem_t*) input_medium1_2,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    if(block == -1 || block == 4){
    // layer 2
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_2, (elem_t*) Wqkvo_medium1_2[0], NULL, (elem_t*) QKV_buf_medium1_2[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_2, (elem_t*) Wqkvo_medium1_2[1], NULL, (elem_t*) QKV_buf_medium1_2[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_2, (elem_t*) Wqkvo_medium1_2[2], NULL, (elem_t*) QKV_buf_medium1_2[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium1_params.hidden_dim / bert_medium1_params.num_head;
    
    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_2[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_2[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_2[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_2[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_2[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_2 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_2, (elem_t*) Wqkvo_medium1_2[3], NULL, (elem_t*) enc_out_medium1_2,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_2,
        (elem_t*) enc_out_medium1_2,
        (elem_t*) input_medium1_2,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    }
    if(block == -1 || block == 5){
    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_2, (elem_t*) ff_w_medium1_2[0], (acc_t*) ff1_b_medium1_2, (elem_t*) out_buf_medium1_2,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_2, (elem_t*) ff_w_medium1_2[1], (acc_t*) ff2_b_medium1_2, (elem_t*) enc_out_medium1_2,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_2,
        (elem_t*) enc_out_medium1_2,
        (elem_t*) input_medium1_3,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    if(block == -1 || block == 6){
    // layer 3
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_3, (elem_t*) Wqkvo_medium1_3[0], NULL, (elem_t*) QKV_buf_medium1_3[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_3, (elem_t*) Wqkvo_medium1_3[1], NULL, (elem_t*) QKV_buf_medium1_3[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_3, (elem_t*) Wqkvo_medium1_3[2], NULL, (elem_t*) QKV_buf_medium1_3[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium1_params.hidden_dim / bert_medium1_params.num_head;
    
    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_3[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_3[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_3[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_3[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_3[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_3 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_3, (elem_t*) Wqkvo_medium1_3[3], NULL, (elem_t*) enc_out_medium1_3,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_3,
        (elem_t*) enc_out_medium1_3,
        (elem_t*) input_medium1_3,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    }
    if(block == -1 || block == 7){
    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_3, (elem_t*) ff_w_medium1_3[0], (acc_t*) ff1_b_medium1_3, (elem_t*) out_buf_medium1_3,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_3, (elem_t*) ff_w_medium1_3[1], (acc_t*) ff2_b_medium1_3, (elem_t*) enc_out_medium1_3,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_3,
        (elem_t*) enc_out_medium1_3,
        (elem_t*) input_medium1_3,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
    }
    /*
    // layer 4
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_4, (elem_t*) Wqkvo_medium1_4[0], NULL, (elem_t*) QKV_buf_medium1_4[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_4, (elem_t*) Wqkvo_medium1_4[1], NULL, (elem_t*) QKV_buf_medium1_4[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_4, (elem_t*) Wqkvo_medium1_4[2], NULL, (elem_t*) QKV_buf_medium1_4[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium1_params.hidden_dim / bert_medium1_params.num_head;
    
    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_4[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_4[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_4[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_4[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_4[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_4 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_4, (elem_t*) Wqkvo_medium1_4[3], NULL, (elem_t*) enc_out_medium1_4,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_4,
        (elem_t*) enc_out_medium1_4,
        (elem_t*) input_medium1_4,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_4, (elem_t*) ff_w_medium1_4[0], (acc_t*) ff1_b_medium1_4, (elem_t*) out_buf_medium1_4,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_4, (elem_t*) ff_w_medium1_4[1], (acc_t*) ff2_b_medium1_4, (elem_t*) enc_out_medium1_4,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_4,
        (elem_t*) enc_out_medium1_4,
        (elem_t*) input_medium1_5,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");


    // layer 5
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_5, (elem_t*) Wqkvo_medium1_5[0], NULL, (elem_t*) QKV_buf_medium1_5[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_5, (elem_t*) Wqkvo_medium1_5[1], NULL, (elem_t*) QKV_buf_medium1_5[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_5, (elem_t*) Wqkvo_medium1_5[2], NULL, (elem_t*) QKV_buf_medium1_5[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium1_params.hidden_dim / bert_medium1_params.num_head;
    
    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_5[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_5[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_5[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_5[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_5[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_5 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_5, (elem_t*) Wqkvo_medium1_5[3], NULL, (elem_t*) enc_out_medium1_5,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_5,
        (elem_t*) enc_out_medium1_5,
        (elem_t*) input_medium1_5,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_5, (elem_t*) ff_w_medium1_5[0], (acc_t*) ff1_b_medium1_5, (elem_t*) out_buf_medium1_5,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_5, (elem_t*) ff_w_medium1_5[1], (acc_t*) ff2_b_medium1_5, (elem_t*) enc_out_medium1_5,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_5,
        (elem_t*) enc_out_medium1_5,
        (elem_t*) input_medium1_6,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");


    // layer 6
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_6, (elem_t*) Wqkvo_medium1_6[0], NULL, (elem_t*) QKV_buf_medium1_6[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_6, (elem_t*) Wqkvo_medium1_6[1], NULL, (elem_t*) QKV_buf_medium1_6[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_6, (elem_t*) Wqkvo_medium1_6[2], NULL, (elem_t*) QKV_buf_medium1_6[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium1_params.hidden_dim / bert_medium1_params.num_head;
    
    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_6[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_6[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_6[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_6[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_6[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_6 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_6, (elem_t*) Wqkvo_medium1_6[3], NULL, (elem_t*) enc_out_medium1_6,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_6,
        (elem_t*) enc_out_medium1_6,
        (elem_t*) input_medium1_6,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_6, (elem_t*) ff_w_medium1_6[0], (acc_t*) ff1_b_medium1_6, (elem_t*) out_buf_medium1_6,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_6, (elem_t*) ff_w_medium1_6[1], (acc_t*) ff2_b_medium1_6, (elem_t*) enc_out_medium1_6,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_6,
        (elem_t*) enc_out_medium1_6,
        (elem_t*) input_medium1_7,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");

    // layer 7
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_7, (elem_t*) Wqkvo_medium1_7[0], NULL, (elem_t*) QKV_buf_medium1_7[0],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] += end - start;
    //printf("done Q\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_7, (elem_t*) Wqkvo_medium1_7[1], NULL, (elem_t*) QKV_buf_medium1_7[1],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] += end - start;
    //printf("done K\n");
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
      
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_7, (elem_t*) Wqkvo_medium1_7[2], NULL, (elem_t*) QKV_buf_medium1_7[2],
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
     //printf("done Q K V \n");

    // attn = Q * K

    hidden_dim_per_head = bert_medium1_params.hidden_dim / bert_medium1_params.num_head;
    
    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) QKV_buf_medium1_7[0] + head * hidden_dim_per_head;
      elem_t * B = (elem_t*) QKV_buf_medium1_7[1] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) attn_buf_medium1_7[head];
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
          bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.seq_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, true,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[3+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn\n");

    for(int head = 0; head < bert_medium1_params.num_head; head++){
      start = read_cycles();
      elem_t * A = (elem_t*) attn_buf_medium1_7[head];
      elem_t * B = (elem_t*) QKV_buf_medium1_7[2] + head * hidden_dim_per_head;
      elem_t * C = (elem_t*) enc_out_medium1_7 + head * hidden_dim_per_head;
      
      tiled_opcode_matmul_default(bert_medium1_params.seq_len, hidden_dim_per_head, bert_medium1_params.seq_len,
          bert_medium1_params.seq_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
          A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
          A, B, NULL, C,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
          false, false, false,
          num_array, cid);
 

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[15+head] += end - start;
      //printf("done head %d\n", head);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif 
    }
    //printf("done attn * V\n");


    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.hidden_stride, 0, bert_medium1_params.hidden_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) enc_out_medium1_7, (elem_t*) Wqkvo_medium1_7[3], NULL, (elem_t*) enc_out_medium1_7,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done out * Wo\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_7,
        (elem_t*) enc_out_medium1_7,
        (elem_t*) input_medium1_7,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done attention resadd\n");

    // out = FF1(input)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.expansion_dim, bert_medium1_params.hidden_dim,
        bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_7, (elem_t*) ff_w_medium1_7[0], (acc_t*) ff1_b_medium1_7, (elem_t*) out_buf_medium1_7,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
   // printf("done ff1\n");


    // out = FF2(out)
    // out = GELU(out)
    start = read_cycles();
    tiled_opcode_matmul_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim, bert_medium1_params.expansion_dim,
        bert_medium1_params.expansion_stride, bert_medium1_params.hidden_stride, bert_medium1_params.expansion_stride, bert_medium1_params.expansion_stride,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        (elem_t*) out_buf_medium1_7, (elem_t*) ff_w_medium1_7[1], (acc_t*) ff2_b_medium1_7, (elem_t*) enc_out_medium1_7,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false,
        num_array, cid);
 
    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done ff2\n");

    start = read_cycles();
    // input = out + input
    tiled_opcode_resadd_default(bert_medium1_params.seq_len, bert_medium1_params.hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY, 
        A_direct_dram, B_direct_dram, C_direct_dram,
        (elem_t*) input_medium1_7,
        (elem_t*) enc_out_medium1_7,
        (elem_t*) input_medium1_7,
         false,
        num_array, cid);


    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] += end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_bert);
#endif
    //printf("done final resadd\n");
*/


    for(int i = 0; i < num_cycle; i++){
      if(i < 30){
        cycles[i] = matmul_cycles[i];
      }
      else if (i < 32){
        cycles[i] = resadd_cycles[i - 30];
      }
      else{
        if(i == 32) cycles[i] = total_matmul_cycles;
        if(i == 33) cycles[i] = total_resadd_cycles;
        if(i == 34) cycles[i] = total_matmul_cycles + total_resadd_cycles;
      }
    }
    return cycles;
#undef num_cycle
}





#else

uint64_t* bertmedium_function_1(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
    
    int round = 4;
    int total_single_time = 8000000;
    if(block >= 0){
        total_single_time /= round;
        round = 1;
    }
    dummy_workload(cid, round, total_single_time, num_array);
}

uint64_t* bertmedium_function_11(int block, bool input_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool output_direct_dram, int num_array, int cid){
    
    int round = 4;
    int total_single_time = 8000000;
    if(block >= 0){
        total_single_time /= round;
        round = 1;
    }
    dummy_workload(cid, round, total_single_time, num_array);
}

#endif
