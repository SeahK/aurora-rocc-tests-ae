// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/rerocc.h"
#include "include/gemmini_params.h"
#include "include/workload_params.h"

//static uint64_t overhead = 0;

#define GEMMINI_ASSERTIONS

// Accelerator interface
#include "rocc-software/src/xcustom.h"

// Counter Definition
#include "include/gemmini_counter.h"

#define k_CONFIG 0
#define k_MVIN2 1
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define k_LOOP_WS 8
#define k_LOOP_WS_CONFIG_BOUNDS 9
#define k_LOOP_WS_CONFIG_ADDRS_AB 10
#define k_LOOP_WS_CONFIG_ADDRS_DC 11
#define k_LOOP_WS_CONFIG_STRIDES_AB 12
#define k_LOOP_WS_CONFIG_STRIDES_DC 13

#define k_MVIN3 14

#define k_COUNTER 126

#define k_LOOP_CONV_WS 15
#define k_LOOP_CONV_WS_CONFIG_1 16
#define k_LOOP_CONV_WS_CONFIG_2 17
#define k_LOOP_CONV_WS_CONFIG_3 18
#define k_LOOP_CONV_WS_CONFIG_4 19
#define k_LOOP_CONV_WS_CONFIG_5 20
#define k_LOOP_CONV_WS_CONFIG_6 21

#define k_LOOP_ONE 23

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define RELU6 2

#define PRINT_MEM 0
#define CALC_MEM 1
#ifdef ELEM_T_IS_FLOAT
elem_t elem_t_bits_to_elem_t(elem_t_bits x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.b = x;
    return un.f;
}

elem_t_bits elem_t_to_elem_t_bits(elem_t x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.f = x;
    return un.b;
}

acc_t acc_t_bits_to_acc_t(acc_t_bits x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.b = x;
    return un.f;
}

acc_t_bits acc_t_to_acc_t_bits(acc_t x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.f = x;
    return un.b;
}

bool elem_t_isnan(elem_t x) {
    elem_t_bits bits = elem_t_to_elem_t_bits(x);
    uint64_t exp = (bits >> (ELEM_T_SIG_BITS-1)) & (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ELEM_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}

bool acc_t_isnan(acc_t x) {
    acc_t_bits bits = acc_t_to_acc_t_bits(x);
    uint64_t exp = (bits >> (ACC_T_SIG_BITS-1)) & (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ACC_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}
#endif

#ifdef HAS_MVIN_SCALE
static scale_t scale_t_bits_to_scale_t(scale_t_bits x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.f = x;
    return un.b;
}
#else
#define scale_t_to_scale_t_bits(x) 0
#endif

#ifdef HAS_MVIN_ACC_SCALE
static scale_acc_t scale_acc_t_bits_to_scale_acc_t(scale_acc_t_bits x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_acc_t_bits scale_acc_t_to_scale_acc_t_bits(scale_acc_t x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

static acc_scale_t acc_scale_t_bits_to_acc_scale_t(acc_scale_t_bits x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.f = x;
    return un.b;
}

int get_num_array_index(int num_array){
    int num_array_index = 0;
    if(num_array == 2) num_array_index = 1;
    else if(num_array == 4) num_array_index = 2;
    return num_array_index;
}
void post_layer_process(int cid, int num_array){
  if (mode == 0)
      return;
  int queue_id = workload_running[cid]; 
  int workload_type = total_queue_type[queue_id];
  if(mode != 1 && mode != 2){
    for(int j = 0; j < 3; j++){
      core_togo[j][cid] -= sp_layer_cycles[j][workload_type][layer_pointer[cid]];
    }
  }
  layer_pointer[cid] ++;
    
  for(size_t k = 0; k < num_array; k++){
    rerocc_cfg_epochrate_by_tracker(k, 0, 0, false);
  }
}

int memory_repartition(uint64_t current_cycle, int num_tile, int layer_num, int cid, int num_array){
    int num_array_index = get_num_array_index(num_array);
    int queue_id = workload_running[cid];
    int workload_type = total_queue_type[queue_id];
    int total_from_dram = sp_layer_from_dram[num_array_index][workload_type][layer_num];
    uint64_t prediction = sp_layer_cycles[num_array_index][workload_type][layer_num];
    //uint64_t ideal_prediction = sp_layer_compute_ideal[num_layer_num][workload_type][layer_num];
    //ideal_prediction += (int)((sp_layer_alpha[num_array_index][workload_type][layer_num] * sp_layer_mem_ideal[num_array_index][workload_type][layer_num])/100);
    int ideal_dram_util = (100 * total_from_dram) / prediction;
    //int ideal_dram_util = (ideal_dram_bw_exp / DRAM_BW); // in percentage

   // uint64_t current_cycle = read_cycles_re() - global_start_time[cid];
    uint64_t old = current_cycle - total_queue_dispatch[queue_id];
    int64_t slack = target_cycles[workload_type] - old;
    int this_score = (target_cycles[workload_type] <= old) ? 10 : (int)((100*core_togo[num_array_index][cid]) / slack); 
    if(this_score > 200) this_score = 200; 
    current_mem_score[cid] = this_score;

    int sum_dram_util = ideal_dram_util;
    for (int i = 0; i < NUM_CORE; i++)
      if(i != cid)
        sum_dram_util += dram_util[i];

    int total_dram_bw = (int)(100 * DRAM_BW);
    bool contention = (sum_dram_util > total_dram_bw);
    int new_prediction = -1;
    if(contention){
       int excess = sum_dram_util - total_dram_bw; 
       //int other_dram_util = 0;
       int other_score = 0;
       int other_weight_sum = 0;
       for(int i = 0; i < NUM_CORE; i++){
           if(i != cid){
               other_score += current_mem_score[i];
               other_weight_sum += current_mem_score[i] * dram_util[i];
           }
       }
       if(other_score > 0 && other_weight_sum > 0){
         int this_dram_util = ideal_dram_util - (int)((excess * other_weight_sum) / (this_score * ideal_dram_util + other_weight_sum));
         if(this_dram_util < 10 && ideal_dram_util >= 10) this_dram_util = 10;
	 dram_util[cid] = this_dram_util;
#if PRINT == 1
         printf("ideal_dram_util: %d, sum_dram_util: %d, excess: %d, this_dram_util: %d / this_score: %d, other_weight_sum: %d\n", ideal_dram_util, sum_dram_util, excess, this_dram_util, this_score, other_weight_sum);
#endif
         new_prediction = (100 * total_from_dram) / this_dram_util;
       }
       else
           dram_util[cid] = ideal_dram_util;
    }
    else
        dram_util[cid] = ideal_dram_util;

    // for debugging
    if(new_prediction > 0 && num_tile > 5 && layer_pointer[cid] > 0)
	total_queue_throttle[queue_id] ++;

    return new_prediction; 
}


#if NOC_OPTIM == 1
int compute_repartition(int layer_num, int cid, int old_num_array){
    if(mode == 0 || mode == 1 || mode == 2 || mode == 4 || mode == 5)
        return old_num_array;

#ifndef BAREMETAL
    // ToDo
    int queue_id = workload_running[cid];
    int workload_type = total_queue_type[queue_id];
    bool perform_mutex = false;

    // skip before first layer
    if(layer_num == 0)
	return core_num_gemmini[cid];

    uint64_t expected_cycle = sp_layer_cycles[2][workload_type][layer_num];
#if SET == 1 || SET == 3
    if(expected_cycle > 100000)
	perform_mutex = true;
#else
    if(expected_cycle > 200000)
	perform_mutex = true;
#endif 



    if(!perform_mutex && (layer_num % 5 == 4) && noc_optim_need_swap[cid] != -1){
#if PRINT==1
	printf("cid %d workload %d need accel sweep\n", cid, workload_type);
#endif
        pthread_mutex_lock(&ex_queue_mutex);
        for(int g_index = 0; g_index < 3; g_index++){
	   int g = gemmini_noc_optim[workload_type][g_index];
  	   if(gemmini_status[g] == -1){
	      int accel = gemmini_noc_optim[workload_type][g_index];
	      int accel2 = hashed_gemmini_pair(accel);
	      // perform swap
	      int tracker = noc_optim_need_swap[cid];
	      rerocc_release(tracker);
	      rerocc_release(tracker+1);
	      gemmini_status[accel] = cid;
              while(!rerocc_acquire(tracker, 1<<accel)){}
	      gemmini_status[accel2] = cid;
              while(!rerocc_acquire(tracker+1, 1<<accel2)){}
	      int old_accel = core_gemmini[cid][tracker];
	      int old_accel2 = core_gemmini[cid][tracker+1];
#if PRINT==1
	      printf("cid %d swap from %d %d to %d %d on tracker %d %d\n", cid, old_accel, old_accel2, accel, accel2, tracker, tracker+1);
#endif
	      gemmini_status[old_accel] = -1;
	      gemmini_status[old_accel2] = -1;
	      core_gemmini[cid][tracker] = accel;
	      core_gemmini[cid][tracker+1] = accel2;
	      total_queue_swap[queue_id] ++;
	      noc_optim_need_swap[cid] = -1;
	      break;
	   }
	}
	pthread_mutex_unlock(&ex_queue_mutex);
    }
    if(!perform_mutex)
        return core_num_gemmini[cid];
    int num_array = core_num_gemmini[cid];
#if PRINT == 1
    printf("cid %d need to perform mutex, current number of array: %d\n", cid, num_array);
#endif
    uint64_t current_cycle = read_cycles_re() - global_start_time[cid];
    uint64_t old = current_cycle - total_queue_dispatch[queue_id];
    uint64_t slack = target_cycles[workload_type] - old;
    //int score = current_score[cid];
    int num_array_index = get_num_array_index(num_array);
    int score = (int)((100*slack) / core_togo[num_array_index][cid]); 
    bool not_meet_ddl = (slack <= core_togo[2][cid]) || (target_cycles[workload_type] <= old);
    // for debugging
    uint64_t start_lock = read_cycles_re();
    pthread_mutex_lock(&ex_queue_mutex);
    current_score[cid] = (not_meet_ddl) ? 0 : score;
    bool need_release = false;
    // for NoC OPTIM, only allow 2, 4 gemminis
    if(num_array > 2){
      int num_idle_array = 0;
      for(int array = 0; array < NUM_ARRAY; array++){
          if(gemmini_status[array] == -1){
              num_idle_array ++; 
          }
      }

      // can't meet ddl no matter what
      if(not_meet_ddl && num_idle_array <= 3)
          need_release = true;
      else if(num_idle_array <= 1 && score > 400)
          need_release = true;
      else{
        for(int c = 0; c < NUM_CORE; c++){
          if(c != cid){
            if(score > current_score[c] * 2 && core_num_gemmini_share[c] < 4 && current_score[c] > 0){ // this one is significantly faster thab other
              need_release = true;
            }
            else if(current_score[c] > score && core_num_gemmini_share[c] > 1){ // can release other one first
              need_release = false;
              break;
            }
          }
        }
      }
    }
    bool need_acquire = (num_array < 4) && !not_meet_ddl;
    for(int c = 0; c < NUM_CORE; c++){
        if(c != cid && score > current_score[c] && current_score[c] > 0 && core_num_gemmini_share[c] < 4){
            need_acquire = false;
            break;
        }
    }
#if PRINT == 1
    printf("cid %d need release: %d need acquire: %d\n", cid, need_release, need_acquire);
#endif
    //ToDo: release better pair
    if(need_release){      
      int accel = core_gemmini[cid][num_array-1];
      num_array --;
      //core_num_gemmini[cid] --;
      core_gemmini[cid][num_array] = -1;
#if PRINT == 1
      //printf("cid %d score %d release accel %d\n", cid, score, accel);
#endif
      // For debugging
      total_queue_release[queue_id] ++;
      rerocc_release(num_array);
      gemmini_status[accel] = -1;
      int accel2 = core_gemmini[cid][num_array-1];
#if PRINT == 1
      printf("cid %d score %d release accel %d, %d\n", cid, score, accel, accel2);
#endif
 
      num_array --;
      //core_num_gemmini[cid] --;
      core_gemmini[cid][num_array] = -1;
      rerocc_release(num_array);
      gemmini_status[accel2] = -1;
      int remain_g = core_gemmini[cid][0];
      if(remain_g == gemmini_noc_optim[workload_type][3] || remain_g == gemmini_noc_optim[workload_type][4]){
	   noc_optim_need_swap[cid] = 0;
#if PRINT == 1
           printf("cid %d workload type %d  need swap of remaining accel %d\n", cid, workload_type, remain_g);
#endif
      }
      else
	   noc_optim_need_swap[cid] = -1;
    }
    else if(need_acquire){
      int save_accel = -1;
      for(int g_index = 0; g_index < NUM_ARRAY_GROUP; g_index++){
	 int accel = gemmini_noc_optim[workload_type][g_index];
	 if(gemmini_status[accel] == -1){
	    core_gemmini[cid][num_array] = accel;

	    total_queue_acquire[queue_id] ++;
	    gemmini_status[accel] = cid;
            while(!rerocc_acquire(num_array, 1<<accel)){}
	    int accel2 = hashed_gemmini_pair(accel);
	    gemmini_status[accel2] = cid;
	    core_gemmini[cid][num_array+1] = accel2;
            while(!rerocc_acquire(num_array+1, 1<<accel2)){}
#if PRINT == 1
            printf("cid %d score %d (prev %d num accel) claim new accel %d, %d (gindex: %d)\n", cid, score, num_array, accel, accel2, g_index);
#endif
	    if(g_index > 2 && noc_optim_need_swap[cid] == -1) {
		 noc_optim_need_swap[cid] = num_array;
#if PRINT == 1
		 printf("cid %d type %d new acquired accel need swap of accel %d gindex %d\n", cid, workload_type, accel, g_index);
#endif
	    }
	    else if(g_index > 2 && noc_optim_need_swap[cid] != -1){
		 noc_optim_need_swap[cid] = -1;
#if PRINT == 1
		 printf("cid %d type %d all 4 can't be swapped\n", cid, workload_type);
#endif
	    }
	    num_array += 2;
	    break;

	 }
      } 
    }
    //acquire or release happend -> need to update score
    if(core_num_gemmini[cid] != num_array){
        core_num_gemmini_share[cid] = num_array; 
        /*
        num_array_index = 0;
        if(num_array == 2) num_array_index = 1;
        else if(num_array == 4) num_array_index = 2;
        */
        int num_array_index = get_num_array_index(num_array);
        score = (int)((100*slack)/core_togo[num_array_index][cid]);
        //current_score[cid] = score;
        current_score[cid] = (not_meet_ddl) ? 0 : score;
#if PRINT == 1
        printf("cid %d realloc happend from num %d to num %d, new score %d\n", cid, core_num_gemmini[cid], num_array, score);
#endif
        core_num_gemmini[cid] = num_array; // locally
    }
    pthread_mutex_unlock(&ex_queue_mutex);
   
    // for debugging
    uint64_t end_lock = read_cycles_re();
    total_queue_overhead[queue_id] += end_lock - start_lock;
    return num_array;
#endif
}
#else

int compute_repartition(int layer_num, int cid, int old_num_array){
    if(mode == 0 || mode == 1 || mode == 2 || mode == 4 || mode == 5)
        return old_num_array;
#ifndef BAREMETAL
    // ToDo
    int queue_id = workload_running[cid];
    int workload_type = total_queue_type[queue_id];
    bool perform_mutex = false;
   
    // skip before first layer
    if(layer_num == 0)
	return core_num_gemmini[cid];

    uint64_t expected_cycle = sp_layer_cycles[2][workload_type][layer_num];
#if SET == 1 || SET == 3
    if(expected_cycle > 100000)
	perform_mutex = true;
#else
    if(expected_cycle > 200000)
	perform_mutex = true;
#endif 

    if(!perform_mutex)
        return core_num_gemmini[cid];
    int num_array = core_num_gemmini[cid];

    uint64_t current_cycle = read_cycles_re() - global_start_time[cid];
    uint64_t old = current_cycle - total_queue_dispatch[queue_id];
    uint64_t slack = target_cycles[workload_type] - old;
    //int score = current_score[cid];
    int num_array_index = get_num_array_index(num_array);
    int score = (int)((100*slack) / core_togo[num_array_index][cid]); 
    bool not_meet_ddl = (slack <= core_togo[2][cid]) || (target_cycles[workload_type] <= old);
    // for debugging
    uint64_t start_lock = read_cycles_re();
    pthread_mutex_lock(&ex_queue_mutex);
    current_score[cid] = (not_meet_ddl) ? 0 : score;
    bool need_release = false;
    if(num_array >= 2){
      int num_idle_array = 0;
      for(int array = 0; array < NUM_ARRAY; array++){
          if(gemmini_status[array] == -1){
              num_idle_array ++; 
          }
      }

      // can't meet ddl no matter what
      if(not_meet_ddl && num_idle_array <= 3)
          need_release = true;
      else if(num_idle_array <= 1 && score > 400)
          need_release = true;
      else{
        for(int c = 0; c < NUM_CORE; c++){
          if(c != cid){
            if(score > current_score[c] * 2 && core_num_gemmini_share[c] < 4 && current_score[c] > 0){ // this one is significantly faster thab other
              need_release = true;
            }
            else if(current_score[c] > score && core_num_gemmini_share[c] > 1){ // can release other one first
              need_release = false;
              break;
            }
          }
        }
      }
    }
    bool need_acquire = (num_array < 4) && !not_meet_ddl;
    for(int c = 0; c < NUM_CORE; c++){
        if(c != cid && score > current_score[c] && current_score[c] > 0 && core_num_gemmini_share[c] < 4){
            need_acquire = false;
            break;
        }
    }
    if(need_release){
      int accel = core_gemmini[cid][num_array-1];
      num_array --;
      //core_num_gemmini[cid] --;
      core_gemmini[cid][num_array] = -1;
#if PRINT == 1
      printf("cid %d score %d release accel %d\n", cid, score, accel);
#endif
      // For debugging
      total_queue_release[queue_id] ++;
      rerocc_release(num_array);
      gemmini_status[accel] = -1;
      if(num_array == 3){
        // for debugging
        total_queue_release[queue_id] ++;
        accel = core_gemmini[cid][num_array-1];
#if PRINT == 1
        printf("cid %d score %d release accel %d\n", cid, score, accel);
#endif
 
        num_array --;
        //core_num_gemmini[cid] --;
        core_gemmini[cid][num_array] = -1;
        rerocc_release(num_array);
        gemmini_status[accel] = -1;
      }
    }
    else if(need_acquire){
      int save_accel = -1;
      for(int accel = 0; accel < NUM_ARRAY; accel ++){
        if(gemmini_status[accel] == -1){
          if(core_num_gemmini[cid] == 1){
            core_gemmini[cid][num_array] = accel;
#if PRINT == 1
            printf("cid %d score %d (prev %d num accel) claim new accel %d\n", cid, score, num_array, accel);
#endif
            // for debugging
            total_queue_acquire[queue_id] ++;
            gemmini_status[accel] = cid;
            while(!rerocc_acquire(num_array, 1<<accel)){}
            //core_num_gemmini[cid]++;
            num_array ++;
            break;
          }
          else if(core_num_gemmini[cid] == 2){
            if (save_accel == -1)
                save_accel = accel;
            else {
                core_gemmini[cid][num_array] = save_accel;
                core_gemmini[cid][num_array+1] = accel;
                // for debugging
                total_queue_acquire[queue_id] +=2;
                gemmini_status[save_accel] = cid;
                gemmini_status[accel] = cid;
#if PRINT == 1
               printf("cid %d (prev %d num accel) claim new accel %d, %d\n", cid, num_array, save_accel, accel);
#endif
                while(!rerocc_acquire(num_array, 1<<save_accel)){}
                while(!rerocc_acquire(num_array+1, 1<<accel)){}
                num_array+=2;
                break;
            }
          }
        }
      }
    }
    //acquire or release happend -> need to update score
    if(core_num_gemmini[cid] != num_array){
        core_num_gemmini[cid] = num_array; // locally
        core_num_gemmini_share[cid] = num_array; 
        /*
        num_array_index = 0;
        if(num_array == 2) num_array_index = 1;
        else if(num_array == 4) num_array_index = 2;
        */
        int num_array_index = get_num_array_index(num_array);
        score = (int)((100*slack)/core_togo[num_array_index][cid]);
        //current_score[cid] = score;
        current_score[cid] = (not_meet_ddl) ? 0 : score;
#if PRINT == 1
        printf("cid %d realloc happend from num %d to num %d, new score %d\n", cid, core_num_gemmini[cid], num_array, score);
#endif
    }
    pthread_mutex_unlock(&ex_queue_mutex);
   
    // for debugging
    uint64_t end_lock = read_cycles_re();
    total_queue_overhead[queue_id] += end_lock - start_lock;
    return num_array;
#endif
}

#endif

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// mvin and mvout
#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

#define gemmini_mvin(dram_addr, spad_addr) \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define gemmini_mvout(dram_addr, spad_addr) \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(A, BD) \
  gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(A, BD) \
  gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define gemmini_preload(BD, C) \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

// config
#define gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG); \

#define gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, 1, A_stride, A_transpose, B_transpose, false)

#define gemmini_extended_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)

#define gemmini_config_ex(dataflow, sys_act, sys_shift) \
    gemmini_extended_config_ex(dataflow, sys_act, sys_shift, 1, 0, 0)


// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define gemmini_extended5_config_ld(direct_dram, stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((uint64_t)(direct_dram) << 5) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define gemmini_extended4_config_ld(direct_dram, stride, scale, shrunk, block_mvin_stride, id) \
  gemmini_extended5_config_ld(direct_dram, stride, scale, shrunk, block_mvin_stride, 1, id) \

#define gemmini_extended3_config_ld(direct_dram, stride, scale, shrunk, id) \
  gemmini_extended4_config_ld(direct_dram, stride, scale, shrunk, DIM, id)

#define gemmini_extended2_config_ld(stride, scale, shrunk) \
  gemmini_extended3_config_ld(false, stride, scale, shrunk, 0)

#define gemmini_extended_config_ld(stride, scale) \
  gemmini_extended2_config_ld(stride, scale, false)

#define gemmini_config_ld(stride) \
  gemmini_extended_config_ld(stride, MVIN_SCALE_IDENTITY)

#define gemmini_extended2_config_st(direct_dram, stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(direct_dram) << 16) |  ((uint64_t)(lpad) << 14) | ((uint64_t)(upad) << 12) | ((uint64_t)(pool_size) << 8) | ((uint64_t)(pool_stride) << 4) | ((acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define gemmini_extended_config_st(direct_dram, stride, acc_act, acc_scale) \
    gemmini_extended2_config_st(direct_dram, stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_config_st(stride) \
    gemmini_extended_config_st(false, stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

// flush
#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")

// Counter access
#define gemmini_counter_access(rd, config_reg) \
  { \
    uint32_t _placeholder; \
    ROCC_INSTRUCTION(XCUSTOM_ACC, rd, config_reg, _placeholder, k_COUNTER) \
  }

// Read counter
static uint32_t counter_read(size_t index) {
  uint32_t config_reg = (index & 0x7) << 4;
  uint32_t res;
  gemmini_counter_access(res, config_reg);
  return res;
}

// Configure counter to take a new signal
static void counter_configure(size_t index, size_t counter_code) {
  int non_incremental = counter_code > INCREMENTAL_COUNTERS;
  if (non_incremental) {
    counter_code -= INCREMENTAL_COUNTERS;
  }

  uint32_t config_reg = (index & 0x7) << 4 | 0x8 | (counter_code & 0x3f) << 12 | non_incremental << 31;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

// Take a snapshot
static void counter_snapshot_take() {
  uint32_t config_reg = 0x4;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

// Counter snapshot reset
static void counter_snapshot_reset() {
  uint32_t config_reg = 0x2;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

// Counter module reset
static void counter_reset() {
  uint32_t config_reg = 0x1;
  uint32_t placeholder;
  gemmini_counter_access(placeholder, config_reg);
}

int abs_diff(int a, int b){
    if (a > b) return a-b;
    else if (a < b) return b-a;
    else return 0;
}

int ceil_divide_int(int a, int b){
  int c = (a % b == 0) ? ((int)(a/b)) :(((int)(a/b)) + 1); 
  if(a < b) c = 1;
  return c;
}

int round_divide_int(int a, int b){
  int c = (a % b == 0) ? ((int)(a/b)) : ((a % b) >= 0.5*b ? (((int)(a/b)) + 1) : (int)(a/b));
  if(a < b) c = 1;
  return c;
}

int round_int(float a){
  int int_a = (int)(a);
  if(int_a - a == 0){
    return int_a;
  }
  else
    return (int)(a + 0.5);
}

// weight-stationary matmul loop
#define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, weightA, a_ex_id, b_ex_id) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_ex_id) << 18) | ((uint64_t)(b_ex_id) << 16) | ((uint64_t)(weightA) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// weight-stationary conv loop
#define gemmini_loop_conv_ws(batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, activation, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, input_direct_dram, weight_direct_dram, output_direct_dram, bias_direct_dram, a_ex_id, b_ex_id) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(out_channels) << 48) | ((uint64_t)(in_channels) << 32) | ((uint64_t)(in_dim) << 16) | (uint64_t)(batch_size), \
      ((uint64_t)(padding) << 48) | ((uint64_t)(stride) << 32) | ((uint64_t)(pool_out_dim) << 16) | (uint64_t)(out_dim), k_LOOP_CONV_WS_CONFIG_1) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(kernel_dim) << 48) | ((uint64_t)(pool_size) << 32) | ((uint64_t)(pool_stride) << 16) | (uint64_t)(pool_padding), \
      ((uint64_t)(batches) << 48) | ((uint64_t)(porows) << 32) | ((uint64_t)(pocols) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_2) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(krows) << 48) | ((uint64_t)(kcols) << 32) | ((uint64_t)(kchs) << 16) | (uint64_t)(lpad), \
      ((uint64_t)(rpad) << 48) | ((uint64_t)(upad) << 32) | ((uint64_t)(dpad) << 16) | (uint64_t)(plpad), k_LOOP_CONV_WS_CONFIG_3) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(orows) << 48) | ((uint64_t)(prpad) << 32) | ((uint64_t)(pupad) << 21) | ((uint64_t)(pdpad) << 10) | (uint64_t)(kernel_dilation), \
      ((uint64_t)(in_stride) << 48) | ((uint64_t)(weight_stride) << 32) | ((uint64_t)(out_stride) << 16) | (uint64_t)(ocols), k_LOOP_CONV_WS_CONFIG_4) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(weight_direct_dram) << 63) | (uint64_t) weights, \
      ((uint64_t)(output_direct_dram) << 63) | (uint64_t) output, k_LOOP_CONV_WS_CONFIG_5) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(bias_direct_dram) << 63) | (uint64_t) bias, \
      ((uint64_t)(input_direct_dram) << 63) | (uint64_t) input, k_LOOP_CONV_WS_CONFIG_6) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_ex_id) << 18) | ((uint64_t)(b_ex_id) << 16) | ((uint64_t)(max_pixels_per_row) << 8) | ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((activation) << 3)| ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), \
      k_LOOP_CONV_WS) \
  }

// for resadd
#define gemmini_loop_one(dram_addr, spad_choice, dram_stride, rows, cols, cols_rounded, operation) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(operation) << 60) | ((uint64_t)(cols_rounded) << 48) | ((uint64_t) dram_addr), ((uint64_t)(rows) << 49) | ((uint64_t)(cols) << 34) | ((uint64_t)(spad_choice) << 32) | dram_stride, k_LOOP_ONE)


static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}

static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}

static int tiled_conv_total_spad_rows(bool acc,
        int stride,
        int input_dilation,
        int kernel_dilation,
        bool downsample,
        bool trans_weight_0132,
        bool trans_input_3120,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int krows_dilated = krows + (kernel_dilation - 1)*(krows - 1);
    const int kcols_dilated = kcols + (kernel_dilation - 1)*(kcols - 1);

    int irows = orows * stride + krows_dilated - 1; // - 2 * padding;
    int icols = ocols * stride + kcols_dilated - 1; // - 2 * padding;
    const int ichs = kchs;

    irows = irows / input_dilation + (irows % input_dilation != 0);
    icols = icols / input_dilation + (icols % input_dilation != 0);

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int batches_per_bank = batches / DIM + (batches % DIM != 0);

    const int A_rows = trans_input_3120 ?
        (batches_per_bank * ichs * (irows >> downsample) * (icols >> downsample)) :
        (in_channels_per_bank * batches * (irows >> downsample) * (icols >> downsample));

    const int B_rows = trans_weight_0132 ?
      in_channels_per_bank * kcols * krows * ochs :
      out_channels_per_bank * kcols * krows * kchs;

    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    return acc ? C_rows : A_rows + B_rows;
}


size_t* tiling_factor_matmul_calculate_auto(size_t dim_I_in, size_t dim_J_in, size_t dim_K_in,
  size_t orow_divide, size_t och_divide, size_t num_array, bool a_transpose, bool b_transpose, bool A_from_dram, bool write_to_dram, size_t cid, size_t args[]){

  bool fc_layer = (dim_I_in <= DIM) || (dim_J_in <= DIM);
  if(fc_layer && dim_K_in >= DIM * 128) {
      A_from_dram = true;
      write_to_dram = true;
  }

  bool row_divisible = (orow_divide > 1);
  size_t orow_offset_floor = 0;
  size_t dim_I = dim_I_in;
  size_t dim_J = dim_J_in;
  size_t dim_K = dim_K_in;
  if(row_divisible){
    dim_I = dim_I_in / orow_divide;
  }
  else
    dim_J = dim_J_in / och_divide;

 
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const bool double_buffered = true;//tiled_matmul_type == WS;
  const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
  const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  size_t tile_I, tile_J, tile_K;

  if (double_buffered) {
    tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
    tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
    tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
  } else {
    tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
    tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
    tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
  }

  const size_t dim_I_in_padded = (dim_I_in / DIM + (dim_I_in % DIM != 0)) * DIM;
  const size_t dim_J_in_padded = (dim_J_in / DIM + (dim_J_in % DIM != 0)) * DIM;
  const size_t dim_K_in_padded = (dim_K_in / DIM + (dim_K_in % DIM != 0)) * DIM;

  
  // Fill scratchpad as much as possible
  while (true) {
    bool increased = false;

    if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
        (tile_J+1) * DIM <= dim_J_padded) {
      tile_J++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
        (tile_I+1) * DIM <= dim_I_padded) {
      tile_I++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
        (tile_K+1) * DIM <= dim_K_padded) {
      tile_K++;
      increased = true;
    }
  
    if (!increased)
      break;
  }
  args[0] = tile_I; args[1] = tile_J; args[2] = tile_K;
  args[3] = dim_I; args[4] = dim_J; args[5] = dim_K;
  const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
  const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);
  const int spad_util = (spad_rows * 100) / max_spad_rows;
  const int acc_util = (acc_rows * 100) / max_acc_rows;
 
#if CALC_MEM == 1
 
  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);


  size_t a_spad_id = 0;
  size_t b_spad_id = 0;

  bool a_reuse = false;
  bool b_reuse = false;
  
  if(J0 * K0 <= 2) 
    b_reuse = true;
  if(I0 * K0 <= 2)
    a_reuse = true;
 
  // for pre-compilation
  int elem_t_bits = (int)(16 / DIM);
  int A_load = 0;
  int B_load = 0;
  int D_load = 0;
  int C_store = 0;
  int D_size = ceil_divide_int(dim_J_in, DIM) * (4/elem_t_bits);
  int A_size = dim_I_in * ceil_divide_int(dim_K_in, DIM);
  int B_size = dim_K_in * ceil_divide_int(dim_J_in, DIM);
  int C_size = dim_I_in * ceil_divide_int(dim_J_in, DIM);
  int D_size_core = ceil_divide_int(dim_J, DIM) * (4/elem_t_bits);
  int A_size_core = dim_I * ceil_divide_int(dim_K, DIM);
  int B_size_core = dim_K * ceil_divide_int(dim_J, DIM);
  int C_size_core = dim_I * ceil_divide_int(dim_J, DIM);
   
  //CALM config
  const uint64_t total_macs = dim_I * dim_J * dim_K;
  uint64_t ideal_runtime = (uint64_t)(total_macs / (DIM*DIM));
  if(fc_layer){
    C_store = dim_I * ceil_divide_int(dim_J, DIM);
    A_load = dim_I * ceil_divide_int(dim_K, DIM);
    B_load = dim_K * ceil_divide_int(dim_J, DIM);
    D_load = dim_I * ceil_divide_int(dim_J, DIM) * 4;
          //printf("number of tiles: %d, target load: %d, window: %d\n", num_tiles, target_load, window);
  }
  else{
    //window = target_tile_runtime;
    for(size_t i0 = 0; i0 < dim_I; i0+=tile_I*DIM){
      int I = i0 + tile_I*DIM > dim_I ? dim_I - i0 : tile_I*DIM;
      for(size_t j0 = 0; j0 < dim_J; j0+=tile_J*DIM){
        int J = j0 + tile_J*DIM > dim_J ? dim_J - j0 : tile_J*DIM;
        int A_load_unit = I > DIM ? DIM : I;
        int B_load_unit = J > DIM ? DIM : J;
        C_store += (ceil_divide_int(J, B_load_unit) *  I);
        D_load += ceil_divide_int(J * (4/elem_t_bits), B_load_unit) * ceil_divide_int(I, A_load_unit);//ceil_divide_int(I, A_load_unit) * 4; //ceil_divide_int(I*J, DIM);
        for (size_t k0 = 0; k0 < dim_K; k0+=tile_K*DIM) {
          int K = k0 + tile_K*DIM > dim_K ? dim_K - k0 : tile_K*DIM;
          int K_load_unit = K > DIM ? DIM : K;
          if(!a_reuse || j0 == 0) {
            if(a_transpose) A_load += (ceil_divide_int(I, A_load_unit) * K);
            else A_load += (ceil_divide_int(K, K_load_unit) * I);
          }
          if(!b_reuse || i0 == 0) {
            if(b_transpose) B_load += (ceil_divide_int(K, K_load_unit) * J);
            else B_load += (ceil_divide_int(J, B_load_unit) * K);
          }
        }
      }
    }
  }
  int tile_I_in = tile_I * orow_divide;

  int inner_tile_A = tile_I_in * DIM * (ceil_divide_int)(dim_K_in, DIM);
  int inner_tile_B = ceil_divide_int(dim_J_in, DIM) * dim_K_in;
  int outer_loop_iter_A = 1;
  int outer_loop_iter_B = (ceil_divide_int)(dim_I_in, tile_I_in*DIM);

  size_t num_tiles = I0 * J0 * K0;
  int from_dram = B_size + D_size;
  int from_dram_core = B_size_core + D_size_core;
  if(A_from_dram){
      from_dram += A_size;
      from_dram_core += A_size_core;
  }
  //int dram_bw = DRAM_BW;
  int l2_bw = CACHE_BANKS;
  int total_load = A_load + B_load + D_load;
  int total_mem = total_load + C_store;
  float effective_l2_bw = (float)(l2_bw/num_array);
  if (effective_l2_bw > 1) effective_l2_bw = 1;
  int l2_pure_mem = total_mem - (int)(from_dram_core / num_array);

  if (write_to_dram) {
      from_dram += C_size;
      from_dram_core += C_size_core;
  }
  float effective_dram_bw = DRAM_BW > (NUM_DRAM_BYTE * num_array) ? (NUM_DRAM_BYTE * num_array) : DRAM_BW;
  int dram_cycle = from_dram / effective_dram_bw;
  int l2_dram_cycle = from_dram_core / effective_l2_bw;
  int l2_pure_cycle = l2_pure_mem / effective_l2_bw;
  int mem_ideal = l2_pure_cycle;
  if(l2_dram_cycle > dram_cycle) mem_ideal += l2_dram_cycle;
  else mem_ideal += dram_cycle;
  uint64_t prediction = ideal_runtime;
  //uint64_t prediction = (mem_ideal > ideal_runtime) ? mem_ideal : ideal_runtime;

  int alpha = args[9]; // given by profiled data
  int real_cycle = args[8];

  if (alpha >= 0){
      //if(mem_ideal < ideal_runtime) 
          prediction += ((alpha * mem_ideal) / 100);
      //else prediction += ((alpha * ideal_runtime) / 100);

      // predicted cycle is more than requested cycle: no need for throttling
      if(prediction > real_cycle)
          real_cycle = -1;
      if(prediction < real_cycle) prediction = real_cycle;
  }
  else if (real_cycle >= 0) {
      int save_real_cycle = real_cycle;
      real_cycle -= prediction;
      prediction = save_real_cycle;
      // ToDo: fix alpha scaling factor of mem_ideal
      //if(mem_ideal < ideal_runtime) 
          alpha = (100 * real_cycle) / mem_ideal;
      //else alpha = (100 * real_cycle) / ideal_runtime;
  }

  if(mode == 2){
    int workload_id = workload_running[cid];
    int num_array_index = 0;
    if(num_array == 2) num_array_index = 1;
    else if(num_array == 4) num_array_index = 2; 

    sp_layer_alpha[num_array_index][workload_id][layer_pointer[cid]] = alpha;
    sp_layer_from_dram[num_array_index][workload_id][layer_pointer[cid]] = from_dram;
    sp_layer_compute_ideal[num_array_index][workload_id][layer_pointer[cid]] = ideal_runtime;
    sp_layer_mem_ideal[num_array_index][workload_id][layer_pointer[cid]] = mem_ideal;
  }

  int epoch = (real_cycle >= 0) ? prediction / num_tiles : 0;
  int max_req = (real_cycle >= 0) ? total_mem / num_tiles : 0;
#if PRINT_MEM == 1
  printf("mem_ideal: %d, ideal_runtime: %d\n", mem_ideal, ideal_runtime); 
  printf("effective l2 bw: %d, effective fram_bw: %d\n", (int)(effective_l2_bw*100), (int)(effective_dram_bw*100));

  printf("tile_I: %d\n", tile_I);
  printf("tile_J: %d\n", tile_J);
  printf("tile_K: %d\n\n", tile_K);
  printf("spad_rows: %d\n", spad_rows);
  printf("acc_rows: %d\n\n", acc_rows);
  printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);
  printf("total A load: %d, total B load: %d, total D load: %d, total load: %d, total store: %d \n", A_load, B_load, D_load, total_load, C_store);
  printf("A size: %d, B size: %d, C size: %d \n", A_size, B_size, C_size);
  printf("inner tile A: %d, inner tile B: %d, outer loop iteration A: %d, outer loop iteration B: %d \n", inner_tile_A, inner_tile_B, outer_loop_iter_A, outer_loop_iter_B);
  printf("number of tile: %d, target load per tile: %d, ideal runtime: %llu\n\n", num_tiles, (A_load + B_load + D_load) / num_tiles, ideal_runtime);
  printf("epoch: %d, max_req: %d, prediction: %d, alpha: %d\n", epoch, max_req, prediction, alpha);
#endif

  args[6] = num_tiles;//epoch;
  args[7] = total_mem;//max_req;
  args[8] = prediction;
  args[9] = alpha;

#endif

  return args;
}

// Tiling functions
static void sp_tiled_matmul_os(const elem_t * A, const elem_t * B, const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        uint8_t weightA) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN-2)) | (full_C << (ADDR_LEN-3));

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    gemmini_extended_config_ld(D_stride, D_scale_factor);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        const size_t blocks = j + D_blocks <= J ? D_blocks : J-j;

        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);

        gemmini_extended_mvin(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }

  // Move-in B
  gemmini_extended_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor);
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K-1 ? pad_K : 0);
      gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
    }
  }

  // Move-in A
  gemmini_extended_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor);
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K; k += A_blocks) {
      const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
    }
  }

  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

      for (size_t k = 0; k < K; k++) {

        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
        if (no_bias_new_matrix) {
          out_sp_addr &= ~(1 << (ADDR_LEN-2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);

        if (k == 0) { // First iteration
          gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        } else { // All other iterations
          gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        }
      }
    }
  }

  // Move-out C
  if (C != NULL) {
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + j)*DIM*sizeof_C;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);
      }
    }
  }
}


static void sp_tiled_matmul_ws(const elem_t * A, const elem_t * B,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        uint8_t weightA,
        size_t a_spad_id, size_t b_spad_id) {

  /*
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));

  const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
    (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
  const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < I; i++) {
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }

  for (size_t j = 0; j < J; j++) {
    for (size_t k = 0; k < K; k++) {
      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
          (A_sp_addr_start + (i*K + k)*DIM);
        const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
          (B_sp_addr_start + (k*J + j)*DIM);
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        // Mvin A
        if (a_transpose) {
          if (j == 0 && i % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (k*A_row_stride + i)*DIM;
            const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
            const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
          }
        } else {
          if (j == 0 && k % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
            const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
          }

        }

        // Mvin B
        if (b_transpose) {
          if (i == 0 && k % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (j*B_row_stride + k)*DIM;
            const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (j == J-1 ? pad_J : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
          }
        } else {
          if (i == 0 && j % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
            const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
            const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
          }
        }

        // Compute
        {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }

          const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
          const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
          const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
          const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

          gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);

          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          }
        }

        // Move-out C
        if (C != NULL && k == K-1 && (j == J-1 || j % C_blocks == C_blocks-1)) {
          const size_t rounded_j = (j / C_blocks) * C_blocks;

          const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
          void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;

          const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
          const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
          const size_t rows = DIM - (i == I - 1 ? pad_I : 0);

          gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
        }
      }
    }
  }
  */

  // Combined loop
  gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
    A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
    a_transpose, b_transpose,
    full_C, low_D, !no_bias || D == NULL,
    weightA, a_spad_id, b_spad_id);
}


static void tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, 
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow) {

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  gemmini_extended_config_ex(dataflow, act, 0, 1, a_transpose, b_transpose);
  gemmini_extended_config_st(C_direct_dram, stride_C * sizeof_C, act, scale);
  gemmini_extended3_config_ld(A_direct_dram, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
  gemmini_extended3_config_ld(B_direct_dram, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
  gemmini_extended3_config_ld(D_direct_dram, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

  void (*inner)(const elem_t *, const elem_t *, const void *, void *,
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        uint8_t,
        size_t, size_t);
  if (dataflow == OUTPUT_STATIONARY) {
  //  inner = &sp_tiled_matmul_os;
  } else /* if (dataflow == WEIGHT_STATIONARY) */ {
    inner = &sp_tiled_matmul_ws;
  }
  size_t a_spad_id = 0;
  size_t b_spad_id = 0;

  bool a_reuse = false;
  bool b_reuse = false;
  
#ifndef BAREMETAL
  if(J0 * K0 <= 2) 
    b_reuse = true;
  if(I0 * K0 <= 2)
    a_reuse = true;
#endif
  //printf("I0: %d, J0: %d, K0: %d, a_reuse: %d, b_reuse: %d \n", I0, J0, K0, a_reuse, b_reuse);


  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        if(a_reuse)
          a_spad_id = ((i0+k0) == 0) ? 1 : 2;
        if(b_reuse)
          b_spad_id = ((j0+k0) == 0) ? 1 : 2;

        const void * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;
        }

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;
//printf("A: %llu, B: %llu\n", A, B);
        const elem_t * a = a_transpose ? (A + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);

        const elem_t * b = b_transpose ? (B + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

        if(a_reuse && j0 >= 1) a = NULL;
        if(b_reuse && i0 >= 1) b = NULL;
//printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: %llu, b: %llu \n", a_reuse, b_reuse, a_spad_id, b_spad_id, a, b);
        (*inner)(a, b, pre, out,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, J, K,
            pad_I, pad_J, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            weightA,
            a_spad_id, b_spad_id);
      }

  gemmini_fence();
}


static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, size_t relu6_shift) {
  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == RELU) {
    x = x < 0 ? 0 : x;
  }
  // TODO add another define to check if relu6_shift is actually used or not
  else if (act == RELU6) {
    int max = 6 << relu6_shift;
    x = x < 0 ? 0 : (x > max ? max : x);
  }
  return x;
}

#ifdef HAS_MVIN_SCALE
#define GEMMINI_SCALE(x, scale) MVIN_SCALE((x), (scale))
#else
#define GEMMINI_SCALE(x, scale) (x)
#endif

#ifdef HAS_MVIN_ACC_SCALE
#define GEMMINI_ACC_SCALE(x, scale) MVIN_SCALE_ACC((x), (scale))
#else
#define GEMMINI_ACC_SCALE(x, scale) (x)
#endif

static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias) {

  const int no_bias = D == NULL;
  if (!transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {

        acc_t result[4][4]; // = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        for (size_t ii = 0; ii < 4; ii++)
          for (size_t jj = 0; jj < 4; jj++) {
            const size_t bias_row = repeating_bias ? 0 : i + ii;
            result[ii][jj] = no_bias ? 0 :
              GEMMINI_ACC_SCALE(*(D + bias_row*stride_D + j + jj), D_scale_factor);
          }

        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[0][1] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[0][2] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[0][3] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[1][0] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[1][1] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[1][2] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[1][3] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[2][0] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[2][1] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[2][2] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[2][3] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[3][0] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[3][1] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[3][2] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[3][3] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
        }

        *(C + i*stride_C + j) =
             scale_and_sat(result[0][0], act, scale, relu6_shift);
        *(C + i*stride_C + j+1) =
             scale_and_sat(result[0][1], act, scale, relu6_shift);
        *(C + i*stride_C + j+2) =
             scale_and_sat(result[0][2], act, scale, relu6_shift);
        *(C + i*stride_C + j+3) =
             scale_and_sat(result[0][3], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j) =
             scale_and_sat(result[1][0], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j+1) =
             scale_and_sat(result[1][1], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j+2) =
             scale_and_sat(result[1][2], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j+3) =
             scale_and_sat(result[1][3], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j) =
             scale_and_sat(result[2][0], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j+1) =
             scale_and_sat(result[2][1], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j+2) =
             scale_and_sat(result[2][2], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j+3) =
             scale_and_sat(result[2][3], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j) =
             scale_and_sat(result[3][0], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j+1) =
             scale_and_sat(result[3][1], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j+2) =
             scale_and_sat(result[3][2], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j+3) =
             scale_and_sat(result[3][3], act, scale, relu6_shift);
      }
    }
  } else {
    size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A}; // i, j stride
    size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1}; // j, k stride
    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        elem_t* c = C + (i * stride_C) + j;

        const size_t bias_row = repeating_bias ? 0 : i;
        acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);

        for (size_t k = 0; k < DIM_K; k++) {
          const elem_t* a = A + i * A_dim_strides[0] + k * A_dim_strides[1];
          const elem_t* b = B + j * B_dim_strides[0] + k * B_dim_strides[1];
          sum += (GEMMINI_SCALE(*a, A_scale_factor) * GEMMINI_SCALE(*b, B_scale_factor));
        }
        *c = scale_and_sat(sum, act, scale, relu6_shift);
      }
    }
  }
}

#undef GEMMINI_SCALE

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU}; // TODO rename this so it's name also applies to convs

// This function runs a tiled matrix multiplication, with hardcoded tiling
// factors
static void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, 
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > total_spad_size) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > total_acc_size) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
    (tiled_matmul_type == WS && transpose_A && transpose_B)) {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D)) {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
    tiled_matmul_outer(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        A_scale_factor, B_scale_factor, D_scale_factor,
        tile_I, tile_J, tile_K,
        act, scale, relu6_shift, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        (int)tiled_matmul_type);
  } else /*if (tiled_matmul_type == CPU)*/ {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, relu6_shift, repeating_bias);
  }
}


// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, 
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

    size_t* args_out;
    size_t args[10];
    args[8] = -1; //real_cycle
    args[9] = (0.5*100);
    args_out = tiling_factor_matmul_calculate_auto(dim_I, dim_J, dim_K, 1, 1, 1, transpose_A, transpose_B, false, false, 0, args);
    dim_I = args_out[3];
    dim_J = args_out[4];
    dim_K = args_out[5];
    size_t tile_I = args_out[0];
    size_t tile_J = args_out[1];
    size_t tile_K = args_out[2];

    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, relu6_shift, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k

#undef db_partition_rows
#undef db_mats_in_partition
#undef db_mats_in_acc
#undef db_max_tile_i_j
#undef db_max_tile_k
}

static void tiled_matmul_auto_multi(size_t dim_I, size_t dim_J, size_t dim_K,
  //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
  size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
  bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram,  
  elem_t* A, elem_t* B,
  void * D, void * C, 
  scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
  bool A_transpose, bool B_transpose,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias, bool low_D,
  const size_t num_array, size_t cid,
  float alpha, int real_cycle) 
// real_cycle: -1 -> don't do throttling
{

  size_t* args_out;
  size_t args[10];
  size_t dim_J_original = dim_J;
  size_t dim_K_original = dim_K;
  size_t dim_I_original = dim_I;
  int orow_divide = 1;
  int och_divide = 1;
  if(dim_J >= num_array * DIM * MAX_BLOCK_LEN){
    och_divide = num_array;
    dim_J = ceil_divide_int(dim_J, num_array);
    if(dim_J % DIM != 0){
      dim_J = ceil_divide_int(dim_J, DIM) * DIM;
    }
  }
  else{
    orow_divide = num_array;
    dim_I = ceil_divide_int(dim_I, num_array);
    //if(dim_I % DIM != 0){
    //  dim_I = ceil_divide_int(dim_I, DIM) * DIM;
    //}
  }

  //int alpha = (0.5 * 100);
  //args[8] = real_cycle;
  args[9] = alpha >= 0 ? (int)(100 * alpha) : -1; 
  args[8] = real_cycle;
  //args[9] = (0.5*100);
  //args[9] = -1;

  // A_from_dram, write_to_dram: false for now
  args_out = tiling_factor_matmul_calculate_auto(dim_I_original, dim_J_original, dim_K_original, orow_divide, och_divide, num_array, A_transpose, B_transpose, true, false, cid, args);

  bool no_bias = (D == NULL);
  dim_I = args_out[3];
  dim_J = args_out[4];
  dim_K = args_out[5];
  size_t tile_I = args_out[0];
  size_t tile_J = args_out[1];
  size_t tile_K = args_out[2];
  int epoch = args_out[6];
  int max_req = args_out[7];
  alpha = args_out[9];
  int prediction = args_out[8];

  int out_offset = (orow_divide > 1) ? 0 : dim_J * cid; // no need to apply offset if we divided row
  int A_orow_offset = (orow_divide > 1 && cid != 0) ? stride_A * cid * dim_I : 0; 
  int C_orow_offset = (orow_divide > 1 && cid != 0) ? stride_C * cid * dim_I : 0; 
//  printf("dim_I: %d, orow_offset_floor: %d, A_row_offset: %d \n", dim_I, orow_offset_floor, A_orow_offset);


#if rerocc_debug == 1
  printf("dim_I: %d, dim_J: %d, dim_K: %d, tile_I: %d, tile_J: %d, tile_K: %d\n", dim_I, dim_J, dim_K, tile_I, tile_J, tile_K);
#endif

  tiled_matmul_outer(dim_I, dim_J, dim_K,
      //sub_num_I, sub_num_J, sub_num_K,
      A + A_orow_offset, B + out_offset, no_bias ? NULL : D + out_offset*sizeof(acc_t), C + (C_orow_offset + out_offset)*sizeof(elem_t), // for now, disable global workload division
      //A + A_orow_offset + A_batch_offset, B + out_offset, no_bias ? NULL : D + out_offset*sizeof(acc_t), C + C_orow_offset + out_offset + C_batch_offset,
      stride_A, stride_B, stride_C, stride_C,
      A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
      A_scale_factor, B_scale_factor, D_scale_factor,
      tile_I, tile_J, tile_K,
      act, scale, relu6_shift, repeating_bias,
      A_transpose, B_transpose, false, low_D, 3,
      (int) WS);
      //num_array, start_tracker,
      //epoch, max_req);
}

int* tiling_factor_calculate(int args[], int stride, int pool_size, int pool_stride, int kernel_dilation, int padding){
  int batch_size = args[0];
  int pool_out_row = args[1];
  int pool_out_dim = args[2];
  int out_channels = args[3];
  int kernel_dim = args[4];
  int in_channels = args[6];
  const int max_args[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
/*
    printf("batches = %d\n", args[0]);
    printf("orows   = %d\n", args[1]);
    printf("ocols   = %d\n", args[2]);
    printf("ochs    = %d\n", args[3]);
    printf("krows   = %d\n", args[4]);
    printf("kcols   = %d\n", args[5]);
    printf("kchs    = %d\n\n", args[6]);
*/
  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int out_channels_idx = 3;
  const int in_channels_idx = 6;
 
  const int input_dilation = 1;
  const bool downsample = false;
  // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);
  int spad_rows = tiled_conv_total_spad_rows(false,
    stride, input_dilation, kernel_dilation, downsample, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows(true,
    stride, input_dilation, kernel_dilation, downsample, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

  while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
    int max_val = -1;
    int max_idx = -1;

    for (size_t j = 0; j < 7; j++) {
      // We avoid reducing ocols when possible to keep the spatial array fully utilized
      size_t i = 0;
      if(j == 0) i = 0;
      else if (j == 4) i = orows_idx;
      else if(j == 1) i = ocols_idx;
      else if (j == 2) i = 4;
      else if(j == 3) i = 5;
      else if(j == 5) i = out_channels_idx;
      else if(j == 6) i = in_channels_idx;

      if(i == 0 && args[0] > 1){ // batch first
        max_val = args[0];
        max_idx = 0;
        break;
      } else if(((pool_stride > 1 && args[in_channels_idx] >= DIM) || args[in_channels_idx] == MAX_BLOCK_LEN * DIM) && args[out_channels_idx] <= MAX_BLOCK_LEN * DIM){
        if(i == orows_idx && args[orows_idx] > 1 && (args[ocols_idx] <= DIM || (args[in_channels_idx] <= DIM * MAX_BLOCK_LEN && args[out_channels_idx] == MAX_BLOCK_LEN*DIM))){// && (args[orows_idx] >= args[ocols_idx] || args[ocols_idx] <= DIM)){ //decrease orows as much as possible 
          max_val = args[orows_idx];
          max_idx = orows_idx;
          break;
        }else if(i == ocols_idx && (args[i]) > DIM){
          max_val = args[ocols_idx];
          max_idx = ocols_idx;
          break;
        }else if((i==4 || i == 5) && args[i] > 1){
          max_val = args[i];
          max_idx = i;
          break;
        }else if(args[i] > DIM && pool_stride > 1 && (i == in_channels_idx || i == out_channels_idx)){
          max_val = args[i];
          max_idx = i;
        }
      }else if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1) && args[i] > max_val) { // and then move on to channels
        if(!((i==out_channels_idx || i==in_channels_idx) && args[i] <= DIM)){
            max_val = args[i];
            max_idx = i;
        }
      }
    }
    if (max_idx == out_channels_idx || max_idx == in_channels_idx) {
      if(max_val > MAX_BLOCK_LEN * DIM || pool_stride > 1){
         // For input and output channels, there's no point in subtracting by just one
        if (args[max_idx] > MAX_BLOCK_LEN*DIM && args[max_idx] % (MAX_BLOCK_LEN * DIM) != 0) {
          args[max_idx] = (args[max_idx] / (MAX_BLOCK_LEN * DIM)) * (MAX_BLOCK_LEN * DIM);
        } else {
          if(args[max_idx] % (2*DIM) == 0) args[max_idx] = args[max_idx] / 2;
          else args[max_idx] = ((args[max_idx]-1) / DIM) * DIM;
        }
        args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
      }
      else if (args[4] > 1 || args[5] > 1){
        if(args[4] > 1) args[4] = 1;//args[4]--;
        else if(args[5] > 1) args[5]--;
      }
      else if(args[in_channels_idx] < DIM){//for first layer
        args[max_idx] = args[max_idx] / 2;
      }
      else if (args[orows_idx] > (DIM/4)){
        args[orows_idx] = args[orows_idx] / 2;
      }
      else if(args[ocols_idx] > DIM){
        args[ocols_idx] = DIM;
      }
    } else {
      if(max_idx == ocols_idx){
        if(args[max_idx] % DIM != 0) args[max_idx] = (args[max_idx]/DIM)*DIM;
        else args[max_idx] -= (DIM/pool_stride);
      }else{
        if(max_idx == 4 || max_idx == 5) args[max_idx] = 1;
        else args[max_idx]--;
      }
    }
    
    if(in_channels == 3 && padding == 0 && kernel_dim == 3){
      int prop = ceil_divide_int(out_channels, args[3]);
      args[3] = out_channels;
      args[2] = args[2] / prop;
    }
    //printf("max_val: %d, max_idx: %d \n", max_val, max_idx);

    spad_rows = tiled_conv_total_spad_rows(false,
      stride, input_dilation, kernel_dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
      stride, input_dilation, kernel_dilation, false, false, false,  args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  }

/*
    printf("batches = %d\n", args[0]);
    printf("orows   = %d\n", args[1]);
    printf("ocols   = %d\n", args[2]);
    printf("ochs    = %d\n", args[3]);
    printf("krows   = %d\n", args[4]);
    printf("kcols   = %d\n", args[5]);
    printf("kchs    = %d\n\n", args[6]);
*/

  // Check if we can increase ocols
  bool not_increased = false;

  // Check if there are any parameters that we can currently still increase
  bool nothing_increased = false;
  bool kdim_increase = true;
  while (!nothing_increased) {
    nothing_increased = true;
    //kdim_increase = true;

    for (size_t j = 0; j < 7; j++) {
       //size_t i =j;//  down_sample ? j : 6-j;
      size_t i = j;
      if(j == 0) i = 5;//in_channels_idx;
      else if (j == 1) i = in_channels_idx;
      else if(j == 2) i = 4;//in_channels_idx;
      else if (j == 3) i = out_channels_idx;
      else if(j == 4) i = ocols_idx;
      else if(j == 5) i = orows_idx;
      else if(j == 6) i = 0; 
      int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
      if(i == out_channels_idx || i == in_channels_idx) args_candidate[i] *= 2;//+= MAX_BLOCK_LEN * DIM;//!down_sample ? MAX_BLOCK_LEN * DIM : DIM;
      else if(i == ocols_idx && (args[i] % DIM == 0)) args_candidate[i] += DIM;
      else args_candidate[i]+= kdim_increase && (i == 4 || i == 5) ? 2 : 1;
      if (args_candidate[i] > max_args[i])
        continue;

      spad_rows = tiled_conv_total_spad_rows(false,
         stride, input_dilation, kernel_dilation, false, false, false,  args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows(true,
         stride, input_dilation, kernel_dilation, false, false, false,  args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

      if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
        args[i] = args_candidate[i];
        nothing_increased = false;
        kdim_increase = false;
      }
    }
  }
/*
    printf("batches = %d\n", args[0]);
    printf("orows   = %d\n", args[1]);
    printf("ocols   = %d\n", args[2]);
    printf("ochs    = %d\n", args[3]);
    printf("krows   = %d\n", args[4]);
    printf("kcols   = %d\n", args[5]);
    printf("kchs    = %d\n\n", args[6]);
*/

  return args;
  
}


int* tiled_conv_bubble_calculate( 
    int args[], //target_util
    int stride, int kernel_dilation, int padding, 
    int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

    size_t orow_divide, size_t och_divide, size_t num_array, size_t cid,
    bool input_from_dram, bool write_to_dram){

  int batch_size = args[0];
  int in_dim = args[1];
  int in_channels = args[2];
  int out_channels = args[3];
  int out_dim = args[4];
  int kernel_dim = args[5];
  int pool_out_dim = args[6];
  int pool_out_row = args[7];

  const bool no_pool = pool_stride == 0;
  if (no_pool) { 
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }
  /*
  int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
  if (pool_ceil_dim){
    pool_out_dim += (out_dim + 2*pool_padding - pool_size) % pool_stride != 0;
  }
  if (och_divide > 1){
      out_channels = ceil_divide_int(out_channels, num_array);
      if(out_channels % DIM != 0)
        out_channels = ceil_divide_int(out_channels, DIM)*DIM;
  }
  int pool_out_row = pool_out_dim;
  if (orow_divide > 1){
      pool_out_row = ceil_divide_int(pool_out_row, num_array);
  }
  */
  bool row_divisible = orow_divide > 1;
 
  int args_in[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
  int* tile_args;
  tile_args = tiling_factor_calculate(args_in, stride, pool_size, pool_stride, kernel_dilation, padding);

  int batches = tile_args[0];
  int porows = tile_args[1];
  int pocols = tile_args[2];
  int pochs = tile_args[3];
  int krows = tile_args[4];
  int kcols = tile_args[5];
  int kchs = tile_args[6];
 
  if(pool_out_dim < porows){
    porows = pool_out_dim;
  }
  const int input_kernel_dilation = 1;
  int spad_rows = tiled_conv_total_spad_rows(false,
       stride, input_kernel_dilation, kernel_dilation, false, false, false,  batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows(true,
       stride, input_kernel_dilation, kernel_dilation, false, false, false,  batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

   // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);
  const int spad_util = (spad_rows*100)/max_spad_rows;
  const int acc_util = (acc_rows*100)/max_acc_rows;
/*
  printf("total spad_rows reserved: %d\n", spad_rows);
  printf("total acc_rows reserved: %d\n\n", acc_rows);
  printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
  printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);
*/

#if CALC_MEM == 1
  // for layer pre-compilation
  int weight_load = 0;
  int input_load = 0;
  int bias_load = 0;
  int output_store = 0;

  int elem_t_bits = (int)(16 / DIM);
  int weight_size = (ceil_divide_int)(out_channels*och_divide, DIM) * in_channels * kernel_dim * kernel_dim;
  int input_size = (ceil_divide_int)(in_channels, DIM) * in_dim * in_dim * batch_size;//original_batch_size; 
  int output_size = (ceil_divide_int)(out_channels*och_divide, DIM) * pool_out_dim * pool_out_dim * batch_size;// original_batch_size;
  int bias_size = ceil_divide_int(out_channels*och_divide, DIM) * (4/elem_t_bits);
  size_t num_tiles = 1;

  int weight_size_core = weight_size / och_divide;
  int input_size_core = input_size / orow_divide;
  int output_size_core = output_size / (orow_divide * och_divide);
  int bias_size_core = bias_size / och_divide;


  //int window = 0;
  //int target_load = 0;
  //printf("tiling factors: %d %d %d %d %d %d %d \n", batches, porows, pocols, pochs, krows, kcols, kchs);
  size_t out_row = (row_divisible) ? (pool_out_row - 1) * pool_stride + pool_size - 2 * pool_padding : out_dim;
  const uint64_t total_macs = out_channels * batch_size * out_dim * out_row * kernel_dim * kernel_dim * in_channels;
  uint64_t ideal_runtime = ((uint64_t)(total_macs / (DIM*DIM)));
  if(in_channels < DIM) ideal_runtime = ((uint64_t)(total_macs / (in_channels * 3 * DIM)));
  int inner_tile_A = in_dim * in_dim * ceil_divide_int(in_channels, DIM) * batch_size;// * original_batch_size;
  int inner_tile_B = kernel_dim * kernel_dim * in_channels * ceil_divide_int(pochs, DIM);
  int outer_loop_iter_A = ceil_divide_int(out_channels * och_divide, pochs);
  int outer_loop_iter_B = 1;
  //if(inner_tile_A + inner_tile_B + (output_size / outer_loop_iter_A) > CACHE_SIZE || in_channels < DIM)
  //  total_from_dram += ((outer_loop_iter_A - added_image) * inner_tile_A);

  inner_tile_A = inner_tile_A / orow_divide;
  const int porow_start = 0;//pool_out_row * cid;
  const int porow_end = pool_out_row;//(cid == orow_divide - 1) ? pool_out_dim : pool_out_row * (cid + 1);

    //const size_t out_row = (pool_out_row - 1) * pool_stride + pool_size - 2 * pool_padding;
  num_tiles = round_divide_int(out_channels, pochs) * round_divide_int(batch_size, batches) * round_divide_int(porow_end - porow_start, porows) * round_divide_int(pool_out_dim, pocols) * round_divide_int(kernel_dim, krows) * round_divide_int(kernel_dim, kcols) * round_divide_int(in_channels, kchs);
 
  bool full_power = false; // when mesh utilization is too low
  if(pochs < DIM || kchs < DIM){
      int eff_poch = pochs >= DIM ? DIM : pochs;
      int eff_koch = kchs >= DIM ? DIM : kchs;
      int ideal_tuned_runtime = ((int)(total_macs / (eff_poch * eff_koch)));
      //full_power = (ideal_tuned_runtime >= target_runtime);
  }

  size_t a_spad_id = 0;
  size_t b_spad_id = 0;

  bool a_reuse = false;
  bool b_reuse = false;
  size_t num_kch = ceil_divide_int(in_channels, kchs);
  size_t num_poch = ceil_divide_int(out_channels, pochs);
  size_t num_b = ceil_divide_int(batch_size, batches);
  size_t num_porow = ceil_divide_int((porow_end - porow_start), porows);
  size_t num_pocol = ceil_divide_int(pool_out_dim, pocols);
  size_t num_krow = ceil_divide_int(kernel_dim, krows);
  size_t num_kcol = ceil_divide_int(kernel_dim, kcols);


  for (int poch = 0; poch < out_channels; poch += pochs) {
      int eff_poch = poch + pochs > out_channels ? out_channels - poch : pochs;
      int poch_unit = eff_poch < DIM ? eff_poch : DIM;
      for (int b = 0; b < batch_size; b += batches) {
        const int batches_ = batch_size - b > batches ? batches : batch_size - b;
        for (int porow = porow_start; porow < porow_end; porow += porows) {
          int eff_porow = porow + porows > porow_end ? porow_end - porow : porows;
          int orow_position = porow * pool_stride - pool_padding;
          const int pupad = orow_position < 0 ? -orow_position : 0;
          const int orow = eff_porow * pool_stride + pool_size - 1;// eff_porow * pool_stride - pool_padding;
          const int pdpad = orow_position + orow > out_dim ? orow + orow_position - out_dim : 0;
          for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
            int eff_pocol = pocol + pocols > pool_out_dim ? pool_out_dim - pocol : pocols;
            int ocol_position = pocol * pool_stride - pool_padding;
            const int plpad = ocol_position < 0 ? -ocol_position : 0;
            const int ocol = eff_pocol * pool_stride + pool_size - 1;//eff_pocol * pool_stride - pool_padding;
            const int prpad = ocol_position + ocol > out_dim ? ocol + ocol_position - out_dim : 0;
            int ocol_unit = ocol < DIM ? ocol : DIM;
            bias_load += batches_ * orow * ceil_divide_int(ocol, ocol_unit) * ceil_divide_int(eff_poch * (4/elem_t_bits), poch_unit);// (int)(orow * ocol * eff_poch / DIM);
            output_store += batches_ * orow * ceil_divide_int(ocol, ocol_unit) * ceil_divide_int(eff_poch * elem_t_bits, poch_unit);
            for (int krow = 0; krow < kernel_dim; krow += krows) {
              int eff_krow = krow + krows > kernel_dim ? kernel_dim - krow : krows;
              int dilated_krows = eff_krow + (kernel_dilation - 1) * (eff_krow - 1);
              const int irow = (orow - pupad - pdpad) * stride + dilated_krows - 1;//orow * stride + krow*kernel_dilation - padding;
              for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                int eff_kcol = kcol + kcols > kernel_dim ? kernel_dim - kcol : kcols;
                int dilated_kcols = eff_kcol + (kernel_dilation - 1) * (eff_kcol - 1);

                const int icol = (ocol - plpad - prpad) * stride + dilated_kcols - 1;//(ocol * stride + kcol*kernel_dilation - padding;

                for (int kch = 0; kch < in_channels; kch += kchs) {
                  int eff_kch = kch + kchs > in_channels ? in_channels - kch : kchs;
                  int kch_unit = eff_kch < DIM ? eff_kch : DIM;
                  if(!a_reuse || (poch == 0)) input_load += batches_ * ceil_divide_int(eff_kch, kch_unit) * irow * icol * elem_t_bits;
                  if(!b_reuse || (pocol + (porow - porow_start) + b == 0)) weight_load += ceil_divide_int(eff_kch, kch_unit) * eff_poch * eff_krow * eff_kcol * elem_t_bits;
                }
              }
            }
          }
        }
      }
  }
    //printf("weight load: %d, input load: %d, bias_load: %d \n", weight_load, input_load, bias_load);
  uint64_t total_load = input_load + weight_load + bias_load;
  uint64_t total_mem = total_load + output_store;
  int from_dram = weight_size + bias_size;
  int from_dram_core = weight_size_core + bias_size_core;
  if (input_from_dram) {
      from_dram += input_size;
      from_dram_core += input_size_core;
  }

  int l2_bw = CACHE_BANKS;
  //int dram_bw = DRAM_BW; 
  float effective_l2_bw = (float)(l2_bw / num_array);
  if (effective_l2_bw > 1) effective_l2_bw = 1;
  int l2_pure_mem = (total_mem - (int)(from_dram_core / num_array));
  
  float effective_dram_bw = DRAM_BW > (NUM_DRAM_BYTE * num_array) ? (NUM_DRAM_BYTE * num_array) : DRAM_BW;
  if (write_to_dram) from_dram += output_size;
  int dram_cycle = from_dram / effective_dram_bw;
  int l2_dram_cycle = from_dram_core / effective_l2_bw;
  int l2_pure_cycle = l2_pure_mem / effective_l2_bw;
  int mem_ideal = l2_pure_cycle;
  if(l2_dram_cycle > dram_cycle) mem_ideal += l2_dram_cycle;
  else mem_ideal += dram_cycle;
  uint64_t prediction = ideal_runtime;
  //uint64_t prediction = (mem_ideal > ideal_runtime) ? mem_ideal : ideal_runtime;
  //printf("mem_ideal: %d, ideal_runtime: %d\n", mem_ideal, ideal_runtime);
  int alpha = args[10]; // ToDo
  int real_cycle = args[9];
  if (alpha >= 0){
      //if(mem_ideal < ideal_runtime) 
          prediction += ((alpha * mem_ideal) / 100);
      //else prediction += ((alpha * ideal_runtime) / 100);
      if(prediction > real_cycle) real_cycle = -1;
      if(prediction < real_cycle) prediction = real_cycle;
  }
  else if(real_cycle >= 0) {
      int save_real_cycle = real_cycle;
      real_cycle -= prediction;
      prediction = save_real_cycle;
      //if(mem_ideal < ideal_runtime) 
          alpha = (100 * real_cycle) / mem_ideal;
      //else alpha = (100 * real_cycle) / ideal_runtime;
  }

  if(mode == 2){
    int workload_id = workload_running[cid];
    int num_array_index = 0;
    if(num_array == 2) num_array_index = 1;
    else if(num_array == 4) num_array_index = 2; 

    sp_layer_alpha[num_array_index][workload_id][layer_pointer[cid]] = alpha;
    sp_layer_from_dram[num_array_index][workload_id][layer_pointer[cid]] = from_dram;
    sp_layer_compute_ideal[num_array_index][workload_id][layer_pointer[cid]] = ideal_runtime;
    sp_layer_mem_ideal[num_array_index][workload_id][layer_pointer[cid]] = mem_ideal;
  }
  int epoch = (real_cycle >= 0) ? prediction / num_tiles : 0;
  int max_req = (real_cycle >= 0) ? total_mem / num_tiles : 0;
#if PRINT_MEM == 1
  //printf("window: %d, target load: %d, prediction cycles: %llu, num tiles: %d \n", window, target_load, prediction, num_tiles);
  // for pre-compilation
  //printf("compute_ideal: %llu, mem_ideal: %llu, ideal prediction cycles: %llu, ideal dram bw usage: %d, ideal dram bw util: %d, result dram bw util: %d\n", ideal_runtime, mem_ideal, ideal_prediction, ideal_dram_bw_exp, ideal_dram_util, dram_util);
 
  // for pre-compilation
  //printf("compute_ideal: %llu, mem_ideal: %llu, num_tiles: %d\n", ideal_runtime, mem_ideal, num_tiles);
  printf("total macs: %llu, num_tiles: %d\n", total_macs, num_tiles);
  // for pre-compilation
  printf("total A load: %d, total B load: %d, total D load: %d \n", input_load, weight_load, bias_load);
  printf("A size: %d, B size: %d, C size: %d \n", input_size, weight_size, output_size);
  printf("inner tile A: %d, inner tile B: %d, outer loop iteration A: %d, outer loop iteration B: %d \n", inner_tile_A, inner_tile_B, outer_loop_iter_A, outer_loop_iter_B);
 // printf("number of tile: %d, target load per tile: %d, ideal runtime: %llu\n\n", num_tiles_store, (input_load + weight_load + bias_load) / num_tiles_store, ideal_runtime);
  printf("epoch: %d, max_req: %d, prediction: %d, alpha: %d\n", epoch, max_req, prediction, alpha);
#endif

  //return:  gemmini_config_calm(epoch, max_req);
  args[7] = num_tiles;//epoch;
  args[8] = total_mem;//max_req;
  args[9] = prediction;
  args[10] = alpha;
#endif


  args[0] = tile_args[0];
  args[1] = tile_args[1];
  args[2] = tile_args[2];
  args[3] = tile_args[3];
  args[4] = tile_args[4];
  args[5] = tile_args[5];
  args[6] = tile_args[6];
  return args;
}


static void sp_tiled_conv(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim, int pool_out_dim,

        int stride, int padding, int kernel_dim, int kernel_dilation,

        int in_stride, int weight_stride, int out_stride,
        bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,

        int pool_size, int pool_stride, int pool_padding,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        int lpad, int rpad, int upad, int dpad,
        int plpad, int prpad, int pupad, int pdpad,

        const elem_t * input,
        const elem_t * weights,
        elem_t * output,
        const acc_t * bias,

        int act, acc_scale_t scale,

        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        bool no_bias, bool no_pool, bool downsample, bool input_dilated,
        size_t a_spad_id, size_t b_spad_id) {


  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
  const int ochs = pochs;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int dilated_krows = krows + (kernel_dilation - 1)*(krows - 1);
  const int dilated_kcols = kcols + (kernel_dilation - 1)*(kcols - 1);
  int irows = orows * stride + dilated_krows - 1;
  int icols = ocols * stride + dilated_kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;
  const int ichs = kchs;

#define UNDILATED(x) ((input_dilated) ? (((x)+1)/2) : (x))

  if (input_dilated) {
    irows_unpadded = (irows_unpadded+1)/2;
    icols_unpadded = (icols_unpadded+1)/2;

    irows = irows_unpadded + UNDILATED(upad) + UNDILATED(dpad);
    icols = icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad);
  }

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
  const bool transposed = trans_output_1203 || trans_input_3120 ||
      trans_weight_1203 || trans_weight_0132;
  int max_pixels_per_row = transposed || wrot180 || downsample ||
      input_dilated || kernel_dilation > 1 ||
      ichs > DIM ? 1 : DIM/ichs;
  if (max_pixels_per_row > kcols) max_pixels_per_row = kcols;
#else
  const int max_pixels_per_row = 1;
#endif

  gemmini_loop_conv_ws(batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, in_direct_dram, weight_direct_dram, out_direct_dram, bias_direct_dram, a_spad_id, b_spad_id);

  /*

  // Calculate spad address offsets
  const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
  const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
  const int B_rows = trans_weight_0132 ?
    in_channels_per_bank * kcols * krows * ochs :
    out_channels_per_bank * kcols * krows * kchs;

  static uint32_t D_sp_addr_row = 0;
  static uint32_t C_sp_addr_row = 0;

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
  const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;

  if (bias != 0) {
    D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  if (output != 0) {
    C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }


  // mvin bias
  if (bias != NULL) {
    // TODO we probably don't need quite this many nested loops for this part

    const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
        MAX_BLOCK_LEN_ACC * DIM;

    gemmini_extended4_config_ld(0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

    for (int b = 0; b < batches; b++)
      for (int orow = 0; orow < orows; orow++)
        for (int ocol = 0; ocol < ocols; ocol += DIM) {
          const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

          for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
            const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

            const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

            const acc_t * bias_dram_addr = no_bias ? NULL : bias + och;

            gemmini_extended_mvin3(bias_dram_addr,
                    D_sp_addr,
                    J, I);
          }
        }
  }

  // mvin input
  {
    int max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
      MAX_BLOCK_LEN * DIM;
    if (trans_input_3120) {
      max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
        MAX_BLOCK_LEN * DIM;
    }

    const int dram_stride = trans_input_3120 ?
      batch_size * sizeof(elem_t) :
      in_channels * sizeof(elem_t);

    const int spad_stride = trans_input_3120 ?
      ichs * (irows >> downsample) * (icols >> downsample) :
      batches * (irows >> downsample) * (icols >> downsample);

    gemmini_extended5_config_ld(dram_stride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, max_pixels_per_row, 0);

    const int b_it = trans_input_3120 ? max_chs_per_mvin : 1;
    const int ich_it = trans_input_3120 ? 1 : max_chs_per_mvin;

    for (int b = 0; b < batches; b += b_it)
      for (int irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += 1 + downsample) {
        const int irow_padded = irow + UNDILATED(upad);

        for (int icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
          // TODO There might be some unnecessary mvins here at the edge of the image

          int I = icols_unpadded - icol > (DIM << downsample) ?
            (DIM << downsample) : icols_unpadded - icol;

          if (icol < 0) {
            I = -icol > DIM ? DIM : -icol;
          } else if (icol >= icols_unpadded) {
            I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
          }

          const int icol_padded = icol + UNDILATED(lpad);

          for (int ich = 0; ich < ichs; ich += ich_it) {
            int K = ichs - ich > max_chs_per_mvin ?
              max_chs_per_mvin : ichs - ich;
            if (trans_input_3120) {
              K = batches - b > max_chs_per_mvin ?
                max_chs_per_mvin : batches - b;
            }

#define DS(x) ((x) >> (downsample))

            uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            if (trans_input_3120) {
              A_sp_addr = A_sp_addr_start + (b / DIM) * ichs * DS(irows) * DS(icols) + ich * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            }

            const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

            const elem_t * in = input + (b*in_dim*in_dim + irow*in_dim + icol) * in_channels + ich;
            if (is_zeros) {
              in = NULL;
            } else if (trans_input_3120) {
              in = input + (ich*in_dim*in_dim + irow*in_dim + icol) * batch_size + b;
            }

            gemmini_extended_mvin(in,
                A_sp_addr,
                K, I >> downsample);
          }

          icol += I;
        }
      }
  }

  // mvin weights
  {
    int max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
        MAX_BLOCK_LEN * DIM;
    if (trans_weight_0132) {
      max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
          MAX_BLOCK_LEN * DIM;
    }

    size_t dram_stride = out_channels * sizeof(elem_t);
    if (trans_weight_1203) {
      dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
    } else if (trans_weight_0132) {
      dram_stride = in_channels * sizeof(elem_t);
    }

    const size_t spad_block_stride = trans_weight_0132 ?
      krows * kcols * ochs : krows * kcols * kchs;

    gemmini_extended4_config_ld(dram_stride, MVIN_SCALE_IDENTITY, false, spad_block_stride, 1);

    const size_t och_it = trans_weight_0132 ? DIM : max_chs_per_mvin;
    const size_t kch_it = trans_weight_0132 ? max_chs_per_mvin : DIM;

    for (int och = 0; och < ochs; och += och_it) {
      for (int krow = 0; krow < krows; krow++)
        for (int kcol = 0; kcol < kcols; kcol++)
          for (int kch = 0; kch < kchs; kch += kch_it) {
            int K = kchs - kch > DIM ? DIM : kchs - kch;
            int J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
            if (trans_weight_0132) {
              K = ochs - och > DIM ? DIM : ochs - och;
              J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
            }

            uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
            if (trans_weight_0132) {
              B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
            }

            const elem_t * w = weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * out_channels + och;
            if (trans_weight_1203) {
              w = weights + (kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och;
            } else if (trans_weight_0132) {
              w = weights + (krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch;
            }

            gemmini_extended_mvin2(w, B_sp_addr, J, K);
          }
    }
  }

  // Compute
  {
    const int b_it = trans_input_3120 ? DIM : 1;
    const int ocol_it = trans_input_3120 ? 1 : (DIM << input_dilated);

    if (trans_input_3120) {
      gemmini_extended3_config_ex(0, 0, 0, 0, 0, 0, orows * ocols, irows * icols, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true);
    }

    for (int och = 0; och < ochs; och += DIM) {
      for (int krow = 0; krow < krows; krow++) {
        for (int kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
          for (int kch = 0; kch < kchs; kch += DIM) {
            bool new_weights = true;

            for (int b = 0; b < batches; b += b_it) {
              for (int orow = 0; orow < orows; orow++) {
                // Skip some kernel rows due to input-dilation
                if (input_dilated && ((krow * kernel_dilation + orow * stride - upad) % 2 != 0)) {
                  continue;
                }

                for (int ocol = 0; ocol < ocols;) {
                  // Skip some cols dimensions due to input-dilation
                  if (input_dilated && ((kcol + ocol * stride - lpad) % 2 != 0)) {
                    ocol++;
                    continue;
                  }

                  int irow = orow * stride + krow * kernel_dilation;
                  int icol = ocol * stride + kcol * kernel_dilation;

                  if (input_dilated) {
                    irow = (irow + 1) / 2;
                    icol = (icol + 1) / 2;
                  }

                  const int pixels = kcols - kcol > max_pixels_per_row ?
                    max_pixels_per_row : kcols - kcol;

                  const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

                  // Over here, construct a new matrix
                  //
                  // Let us assume that we only ever operate on
                  // one pixel in one row.
                  // Thus, krows == kcols == 1
                  //
                  // Then, for every set of I, J, and K values
                  //     - I = ocols
                  //     - J = ochs
                  //     - K = kchs

                  int I = UNDILATED(ocols - ocol > (DIM << input_dilated) ? (DIM << input_dilated) : ocols - ocol);
                  const int J = ochs - och > DIM ? DIM : ochs - och;
                  const int K = pixels * (kchs - kch > DIM ? DIM : kchs - kch);

                  if (trans_input_3120) {
                    I = batches - b > DIM ? DIM : batches - b;
                  }

                  uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  if (trans_input_3120) {
                    A_sp_addr = A_sp_addr_start + (b / DIM) * kchs * DS(irows) * DS(icols) + kch * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  }

                  const int krow_ = wrot180 ? krows - krow - 1 : krow;
                  const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

                  uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow_ * kcols * kchs + kcol_ * kchs + kch;
                  if (trans_weight_0132) {
                    B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow_ * kcols * ochs + kcol_ * ochs + och;
                  }

                  const uint32_t pre_sp_addr = new_weights ?
                    B_sp_addr : GARBAGE_ADDR;

                  // perform matmul
                  gemmini_extended_preload(pre_sp_addr, C_sp_addr, J, K, J, I);

                  if (new_weights) {
                    gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  } else {
                    gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  }

                  ocol += ocol_it;
                  new_weights = false;
                }
              }
            }
          }
        }
      }
    }
  }

#undef DS
#undef UNDILATED

  // mvout output
  if (output != NULL) {
    if (no_pool) {
      for (int b = 0; b < batches; b++)
        for (int orow = 0; orow < orows; orow++)
          for (int ocol = 0; ocol < ocols; ocol += DIM) {
            const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

            for (int och = 0; och < ochs; och += DIM) {
              const int J = ochs - och > DIM ? DIM : ochs - och;

              const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

              elem_t * out = output + (b*out_dim*out_dim + orow*out_dim + ocol) * out_channels + och;
              if (trans_output_1203) {
                out = output + (orow*out_dim*batch_size + ocol*batch_size + b) * out_channels + och;
              }

              gemmini_extended_mvout(out,
                  C_sp_addr,
                  J, I);
            }
          }
    } else {
      gemmini_extended2_config_st(out_channels * sizeof(elem_t), act, scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);

      for (int b = 0; b < batches; b++) {
        for (int poch = 0; poch < pochs; poch += DIM) {
          const int channels = poch + DIM >= pochs ? pochs - poch : DIM;

          elem_t * pout = output + (b * pool_out_dim * pool_out_dim)*out_channels + poch;

          const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

          gemmini_extended_mvout(pout,
              C_sp_addr,
              channels, 0);
        }
      }

      gemmini_extended_config_st(out_channels * sizeof(elem_t), act, scale);
    }
  }
  */
}


static void sp_tiled_conv_dw(
        int batch_size, int in_dim, int channels, int out_dim, int pool_out_dim,

        int stride, int padding, int kernel_dim,

        int pool_size, int pool_stride, int pool_padding,

        int batches,
        int porows, int pocols,
        int krows, int kcols,

        int lpad, int rpad, int upad, int dpad,
        int plpad, int prpad, int pupad, int pdpad,

        const elem_t * input,
        const elem_t * weights,
        elem_t * output,
        const acc_t * bias,

        int act, acc_scale_t scale,

        bool no_bias, bool no_pool) {

  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  int irows = orows * stride + krows - 1;
  int icols = ocols * stride + kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
  int max_pixels_per_row = DIM;
  if (max_pixels_per_row > kcols) max_pixels_per_row = kcols;
#else
  const int max_pixels_per_row = 1;
#endif

  // Calculate spad address offsets
  const int B_rows = kcols * krows;

  static uint32_t D_sp_addr_row = 0;
  static uint32_t C_sp_addr_row = 0;

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
  const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;

  if (bias != 0) {
    D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  if (output != 0) {
    C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  // mvin bias
  if (bias != NULL) {
    // TODO we probably don't need quite this many nested loops for this part

    gemmini_extended4_config_ld(false, 0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

    for (int b = 0; b < batches; b++)
      for (int orow = 0; orow < orows; orow++)
        for (int ocol = 0; ocol < ocols; ocol += DIM) {
          const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

          const uint32_t D_sp_addr = D_sp_addr_start + b * orows * ocols + orow * ocols + ocol;

          const acc_t * bias_dram_addr = no_bias ? NULL : bias;

          gemmini_extended_mvin3(bias_dram_addr,
                  D_sp_addr,
                  1, I);
        }
  }

  // mvin input
  {
    const int max_chs_per_mvin = 1;

    const int dram_stride = channels * sizeof(elem_t);

    gemmini_extended5_config_ld(false, dram_stride, MVIN_SCALE_IDENTITY, false, DIM, max_pixels_per_row, 0);

    for (int b = 0; b < batches; b++)
      for (int irow = -upad; irow < irows_unpadded + dpad; irow++) {
        const int irow_padded = irow + upad;

        for (int icol = -lpad; icol < icols_unpadded + rpad;) {
          // TODO There might be some unnecessary mvins here at the edge of the image

          int I = icols_unpadded - icol > DIM ? DIM : icols_unpadded - icol;

          if (icol < 0) {
            I = -icol > DIM ? DIM : -icol;
          } else if (icol >= icols_unpadded) {
            I = icols_unpadded + rpad - icol > DIM ? DIM : icols_unpadded + rpad - icol;
          }

          const int icol_padded = icol + lpad;

          uint32_t A_sp_addr = A_sp_addr_start + b * irows * icols + irow_padded * icols + icol_padded;

          const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

          const elem_t * in = input + (b*in_dim*in_dim + irow*in_dim + icol) * channels;
          if (is_zeros) {
            in = NULL;
          }

          gemmini_extended_mvin(in,
              A_sp_addr,
              1, I);

          icol += I;
        }
      }
  }

  // mvin weights
  {
    gemmini_extended4_config_ld(false, 1, MVIN_SCALE_IDENTITY, false, DIM, 1);

    for (int krow = 0; krow < krows; krow++)
      for (int kcol = 0; kcol < kcols; kcol += DIM) {
        int K = kcols - kcol > DIM ? DIM : kcols - kcol;

        const uint32_t B_sp_addr = B_sp_addr_start + krow * kcols + kcol;

        const elem_t * w = weights + krow*kernel_dim + kcol;

        gemmini_extended_mvin2(w, B_sp_addr, 1, K);
      }
  }

  // Compute
  {
    for (int krow = 0; krow < krows; krow++) {
      for (int kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
        bool new_weights = true;

        for (int b = 0; b < batches; b++) {
          for (int orow = 0; orow < orows; orow++) {
            for (int ocol = 0; ocol < ocols; ocol += DIM) {

              int irow = orow * stride + krow;
              int icol = ocol * stride + kcol;

              const int pixels = kcols - kcol > max_pixels_per_row ?
                max_pixels_per_row : kcols - kcol;

              const uint32_t C_sp_addr = C_sp_addr_start + b * orows * ocols + orow * ocols + ocol;

              // Over here, construct a new matrix
              //
              // Let us assume that we only ever operate on
              // one pixel in one row.
              // Thus, krows == kcols == 1
              //
              // Then, for every set of I, J, and K values
              //     - I = ocols
              //     - J = ochs
              //     - K = kchs

              int I = ocols - ocol > DIM ? DIM : ocols - ocol;
              const int J = 1;
              const int K = pixels;

              const uint32_t A_sp_addr = A_sp_addr_start + b * irows * icols + irow * icols + icol;

              uint32_t B_sp_addr = B_sp_addr_start + krow * kcols + kcol;

              const uint32_t pre_sp_addr = new_weights ?
                B_sp_addr : GARBAGE_ADDR;

              // perform matmul
              gemmini_extended_preload(pre_sp_addr, C_sp_addr, J, K, J, I);

              if (new_weights) {
                gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
              } else {
                gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
              }

              new_weights = false;
            }
          }
        }
      }
    }
  }

  // mvout output
  if (output != NULL) {
    if (no_pool) {
      for (int b = 0; b < batches; b++)
        for (int orow = 0; orow < orows; orow++)
          for (int ocol = 0; ocol < ocols; ocol += DIM) {
            const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

            const uint32_t C_sp_addr = C_sp_addr_start + b * orows * ocols + orow * ocols + ocol;

            elem_t * out = output + (b*out_dim*out_dim + orow*out_dim + ocol) * channels;

            gemmini_extended_mvout(out,
                C_sp_addr,
                1, I);
        }
    } else {
      gemmini_extended2_config_st(false, channels * sizeof(elem_t), act, scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);

      for (int b = 0; b < batches; b++) {
        elem_t * pout = output + (b * pool_out_dim * pool_out_dim)*channels;

        const uint32_t C_sp_addr = C_sp_addr_start + b * orows * ocols;

        gemmini_extended_mvout(pout,
            C_sp_addr,
            1, 0);
      }

      gemmini_extended_config_st(false, channels * sizeof(elem_t), act, scale);
    }
  }
}


static int tiled_conv_total_spad_rows_dw(bool acc, bool weight,
        int stride,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int irows = orows * stride + krows - 1; // - 2 * padding;
    const int icols = ocols * stride + kcols - 1; // - 2 * padding;
    const int ichs = kchs;

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);

    const int A_rows = in_channels_per_bank * batches * irows * icols;
    const int B_rows = out_channels_per_bank * kcols * krows * kchs;
    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    if (acc)
        return C_rows;
    else if(weight)
        return B_rows;
    else
        return A_rows;
}

static void conv_cpu_without_pool(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_dim; orow++) {
      for (int ocol = 0; ocol < out_dim; ocol++) {
        for (int och = 0; och < out_channels; och++) {

          acc_t opixel = no_bias ? 0 : bias[och];

          for (int krow = 0; krow < kernel_dim; krow++) {
            if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
              continue;

            const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                continue;

              const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

              for (int kch = 0; kch < in_channels; kch++) {
                const elem_t * in = input + (b * in_dim * in_dim + irow * in_dim + icol) * in_channels + kch;
                if (trans_input_3120) {
                  // NHWC to CHWN
                  in = input + (kch * in_dim * in_dim + irow * in_dim + icol) * batch_size + b;
                }

                elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ?
                    0 : *in;

                const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + och);
                if (trans_weight_1203) {
                  // HWIO to WIHO
                  weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + och);
                } else if (trans_weight_0132) {
                  // HWIO to HWOI
                  weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + och) * in_channels + kch);
                }

                opixel += weight * ipixel;
              }
            }
          }

          elem_t * out = output+(b*out_dim*out_dim+orow*out_dim+ocol)*out_channels + och;
          if (trans_output_1203) {
            // NHWC to HWNC
            out = output+(orow*out_dim*batch_size+ocol*batch_size+b)*out_channels + och;
          }

          *out = scale_and_sat(opixel, act, scale, relu6_shift);
        }
      }
    }
  }
}


static void conv_dw_cpu_without_pool(
        int batch_size, int in_dim, int channels, int out_dim,
        int stride, int padding, int kernel_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_dim; orow++) {
      for (int ocol = 0; ocol < out_dim; ocol++) {
        for (int ch = 0; ch < channels; ch++) {
          acc_t opixel = no_bias ? 0 : bias[ch];

          for (int krow = 0; krow < kernel_dim; krow++) {
            const int irow = orow * stride + krow - padding;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              const int icol = ocol * stride + kcol - padding;

              const elem_t * in = input + (b * in_dim * in_dim + irow * in_dim + icol) * channels + ch;

              const elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ?
                  0 : *in;

              const elem_t weight = *(weights + (ch * kernel_dim + krow) * kernel_dim  + kcol);

              opixel += weight * ipixel;
            }
          }

          elem_t * out = output+(b*out_dim*out_dim+orow*out_dim+ocol)*channels + ch;

          *out = scale_and_sat(opixel, act, scale, relu6_shift);
        }
      }
    }
  }
}


static void conv_cpu(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_cpu_without_pool(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale, relu6_shift);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_dim; pocol++) {
        for (int poch = 0; poch < out_channels; poch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_dim || ocol < 0 || ocol >= out_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[poch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
                    continue;

                  const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                      continue;

                    const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

                    for (int kch = 0; kch < in_channels; kch++) {
                      const elem_t * in = input + (b * in_dim * in_dim + irow * in_dim + icol) * in_channels + kch;
                      if (trans_input_3120) {
                        // NHWC to CHWN
                        in = input + (kch * in_dim * in_dim + irow * in_dim + icol) * batch_size + b;
                      }

                      elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ?
                          0 : *in;

                      const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                      const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                      elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + poch);
                      if (trans_weight_1203) {
                        // HWIO to WIHO
                        weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + poch);
                      } else if (trans_weight_0132) {
                        // HWIO to HWOI
                        weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch);
                      }

                      opixel += weight * ipixel;
                    }
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, relu6_shift);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol)*out_channels + poch;
                if (trans_output_1203) {
                  // NHWC to HWNC
                  out = output + (porow*pool_out_dim*batch_size + pocol*batch_size + b)*out_channels + poch;
                }

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}


static void conv_dw_cpu(
        int batch_size, int in_dim, int channels, int out_dim,
        int stride, int padding, int kernel_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_dw_cpu_without_pool(
        batch_size, in_dim, channels, out_dim,
        stride, padding, kernel_dim,
        input, weights, bias, output,
        act, scale, relu6_shift);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_dim; pocol++) {
        for (int ch = 0; ch < channels; ch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_dim || ocol < 0 || ocol >= out_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[ch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  const int irow = orow * stride + krow - padding;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    const int icol = ocol * stride + kcol - padding;

                    const elem_t * in = input + (b * in_dim * in_dim + irow * in_dim + icol) * channels + ch;

                    elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ?
                        0 : *in;

                    const elem_t weight = *(weights + (ch * kernel_dim + krow) * kernel_dim  + kcol);

                    opixel += weight * ipixel;
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, relu6_shift);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol)*channels + ch;

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}


static void tiled_conv(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        int in_stride, int weight_stride, int out_stride,
        bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        elem_t * input,
        elem_t * weights,
        acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

        enum tiled_matmul_type_t tiled_conv_type,
		size_t orow_divide, size_t cid){
        //size_t orow_divide, size_t cid, size_t group_id) {

#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132) {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      conv_cpu(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale, relu6_shift,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }

    // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const bool downsample = false;//stride == 2 && kernel_dim == 1 && in_dim % 2 == 0
      //&& padding == 0 && no_pool && input_dilation == 1 && !trans_input_3120;

    const int input_dilated = input_dilation == 2;

#ifdef GEMMINI_ASSERTIONS
    {
        // const int orows = porows * pool_stride + pool_size - 1;
        // const int ocols = pocols * pool_stride + pool_size - 1;

        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
        if (input_dilation > 2) {
            printf("input_dilation > 2 is only supported on CPU\n");
            exit(1);
        }
        if (input_dilation > 1 && stride > 1) {
            printf("input input_dilation is only supported when stride == 1\n");
            exit(1);
        }
        if (trans_output_1203 && !no_pool) {
            printf("Output can only be transposed when pooling is disabled\n");
            exit(1);
        }
        if (trans_input_3120 && trans_weight_0132) {
            printf("Cannot transpose innermost dimensions of both inputs and weights on WS.\n");
            exit(1);
        }
    }
#endif

    const size_t st_dram_stride = trans_output_1203 ?
        batch_size * out_channels * sizeof(elem_t) :
        out_stride * sizeof(elem_t);
    gemmini_extended_config_st(out_direct_dram, st_dram_stride, act, scale);

    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
    const int dilated_in_dim = in_dim + (input_dilation-1)*(in_dim-1);

    int pool_out_row = (pool_out_dim % orow_divide == 0) ? pool_out_dim / orow_divide : ((int)(pool_out_dim/orow_divide)) + 1;
    int porow_start = (orow_divide == 1) ? 0 : pool_out_row * cid;
    int porow_end = (orow_divide == 1) ? pool_out_dim : ((cid == orow_divide - 1) ? pool_out_dim : pool_out_row * (cid + 1));

    while(kchs < DIM && pocols >= (int)(pool_out_dim / 4) && porows < (int)(pool_out_row)){
       pocols = pocols / 2;
       porows = porows * 2;
    }
    //printf("batches: %d, porows: %d, pocols: %d, pochs: %d, krows: %d, kcols: %d, kchs: %d\n", batches, porows, pocols, pochs, krows, kcols, kchs);
    
    size_t a_spad_id = 0;
    size_t b_spad_id = 0;

    bool a_reuse = false;
    bool b_reuse = false;
    size_t num_kch = ceil_divide_int(in_channels, kchs);
    size_t num_poch = ceil_divide_int(out_channels, pochs);
    size_t num_b = ceil_divide_int(batch_size, batches);
    size_t num_porow = ceil_divide_int((porow_end - porow_start), porows);
    size_t num_pocol = ceil_divide_int(pool_out_dim, pocols);
    size_t num_krow = ceil_divide_int(kernel_dim, krows);
    size_t num_kcol = ceil_divide_int(kernel_dim, kcols);


//    printf("num_kch: %d, num_poch: %d, num_b: %d, num_porow: %d, num_pocol: %d, num_krow: %d, num_kcol: %d\n", num_kch, num_poch, num_b, num_porow, num_pocol, num_krow, num_kcol);

    if(num_kch * num_poch * num_krow * num_kcol <= 2) 
      b_reuse = true;
    if(num_kch * num_krow * num_kcol * num_b * num_porow * num_pocol <= 2)
      a_reuse = true;


    for (int poch = 0; poch < out_channels; poch += pochs) {
      for (int b = 0; b < batch_size; b += batches) {
        for (int porow = porow_start; porow < porow_end; porow += porows) {
 //         printf("porow_start: %d, porow_end: %d, porow: %d \n", porow_start, porow_end, porow);
          const int orow = porow * pool_stride - pool_padding;
          for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
            const int ocol = pocol * pool_stride - pool_padding;
            for (int krow = 0; krow < kernel_dim; krow += krows) {
              const int orow_floored = orow < 0 ? 0 : orow;
              const int irow = orow_floored * stride + krow*kernel_dilation - padding;
              for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                const int ocol_floored = ocol < 0 ? 0 : ocol;
                const int icol = ocol_floored * stride + kcol*kernel_dilation - padding;
                for (int kch = 0; kch < in_channels; kch += kchs) {
                  if(a_reuse)
                    a_spad_id = (kch + krow + kcol + b + (porow - porow_start) + pocol) == 0 ? 1 : 2;
                  if(b_reuse)
                    b_spad_id = (kch + poch + krow + kcol) == 0 ? 1 : 2;

                  elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * out_stride + poch;
                  if (trans_output_1203) {
                    out = output + (porow*pool_out_dim*batch_size + pocol*batch_size + b) * out_channels + poch;
                  }

                  if (krow + krows < kernel_dim ||
                      kcol + kcols < kernel_dim ||
                      kch + kchs < in_channels) {
                    out = NULL;
                  }

                  acc_t * bias_ = bias + poch;
                  if (krow > 0 ||
                          kcol > 0 ||
                          kch > 0) {
                      bias_ = NULL;
                  }

                  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                  const int porows_ = pool_out_dim - porow > porows ? porows : pool_out_dim - porow;
                  const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                  const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                  const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                  const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                  const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

                  const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  const int orows_ = porows_ * pool_stride + pool_size - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad = ocol + ocols_ > out_dim ? ocol + ocols_ - out_dim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad = orow + orows_ > out_dim ? orow + orows_ - out_dim : 0;

                  const int dilated_krows_ = krows_ + (kernel_dilation - 1)*(krows_ - 1);
                  const int dilated_kcols_ = kcols_ + (kernel_dilation - 1)*(kcols_ - 1);

                  const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
                  const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

                  int lpad = icol < 0 ? -icol : 0;
                  int rpad = icol + icols_ > dilated_in_dim ? icol + icols_ - dilated_in_dim : 0;
                  int upad = irow < 0 ? -irow : 0;
                  int dpad = irow + irows_ > dilated_in_dim ? irow + irows_ - dilated_in_dim : 0;

                  if (input_dilated) {
                    lpad += lpad == 0 && icol % 2 != 0;
                    rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                    upad += upad == 0 && irow % 2 != 0;
                    dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                  }

                  int krow_ = krow;
                  int kcol_ = kcol;
                  if (wrot180) {
                    krow_ = kernel_dim - krow - krows_;
                    kcol_ = kernel_dim - kcol - kcols_;
                  }

                  const elem_t * weights_slice = weights + (krow_*kernel_dim*in_channels + kcol_*in_channels + kch) * weight_stride + poch;
                  if (trans_weight_1203) {
                    weights_slice = weights + (kch*kernel_dim*kernel_dim + krow_*kernel_dim+kcol_) * out_channels + poch;
                  } else if (trans_weight_0132) {
                    weights_slice = weights + (krow_*kernel_dim*out_channels + kcol_*out_channels + poch) * in_channels + kch;
                  }

                  const elem_t * in = input + (b*in_dim*in_dim + ((irow+upad)>>input_dilated)*in_dim + ((icol+lpad)>>input_dilated)) * in_stride + kch;
                  if (trans_input_3120) {
                    in = input + (kch*in_dim*in_dim + ((irow+upad)>>input_dilated)*in_dim + ((icol+lpad)>>input_dilated)) * batch_size + b;
                  }

                  if(b_reuse && (pocol + (porow - porow_start) + b > 0)) weights_slice = NULL;
                  if(a_reuse && (poch > 0)) in = NULL;
                  
                  sp_tiled_conv(
                      batch_size, in_dim, in_channels,
                      out_channels, out_dim, pool_out_dim,

                      stride, padding, kernel_dim, kernel_dilation,

                      in_stride, weight_stride, out_stride,
                      in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

                      pool_size, pool_stride, pool_padding,

                      batches_,
                      porows_, pocols_, pochs_,
                      krows_, kcols_, kchs_,

                      lpad, rpad, upad, dpad,
                      plpad, prpad, pupad, pdpad,

                      in,
                      weights_slice,
                      out,
                      bias_,

                      act, scale,

                      wrot180, trans_output_1203, trans_input_3120,
                      trans_weight_1203, trans_weight_0132,

                      no_bias, no_pool, downsample, input_dilated,
                      a_spad_id, b_spad_id);
                    
                }
              }
            }
          }
        }
      }
    }
}


static void tiled_conv_dw(
    int batch_size, int in_dim, int channels, int out_dim,
    int stride, int padding, int kernel_dim,

    int batches,
    int porows, int pocols, int chs,
    int krows, int kcols,

    const elem_t * input,
    const elem_t * weights,
    const acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type) {

    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      conv_dw_cpu(
        batch_size, in_dim, channels, out_dim,
        stride, padding, kernel_dim,
        input, weights, bias, output,
        act, scale, relu6_shift,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }

    // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

#ifdef GEMMINI_ASSERTIONS
    {
        // const int orows = porows * pool_stride + pool_size - 1;
        // const int ocols = pocols * pool_stride + pool_size - 1;

        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            batches, porows, pocols, chs, krows, kcols, 1, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            batches, porows, pocols, chs, krows, kcols, 1, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
    }
#endif

    const size_t st_dram_stride = channels * sizeof(elem_t);
    gemmini_extended_config_st(false, st_dram_stride, act, scale);

    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, stride, false, false, false);

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = 0; porow < pool_out_dim; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int ch = 0; ch < channels; ch += chs) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        int irow = orow_floored * stride + krow - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            int icol = ocol_floored * stride + kcol - padding;

                            elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * channels + ch;

                            if (krow + krows < kernel_dim ||
                                    kcol + kcols < kernel_dim) {
                                out = NULL;
                            }

                            const acc_t * bias_ = bias + ch;
                            if (krow > 0 ||
                                    kcol > 0) {
                                bias_ = NULL;
                            }

                            const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                            const int porows_ = pool_out_dim - porow > porows ? porows : pool_out_dim - porow;
                            const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                            const int chs_ = channels - ch > chs ? chs : channels - ch;
                            const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                            const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;

                            const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                            const int orows_ = porows_ * pool_stride + pool_size - 1;

                            const int plpad = ocol < 0 ? -ocol : 0;
                            const int prpad = ocol + ocols_ > out_dim ? ocol + ocols_ - out_dim : 0;
                            const int pupad = orow < 0 ? -orow : 0;
                            const int pdpad = orow + orows_ > out_dim ? orow + orows_ - out_dim : 0;

                            const int icols_ = (ocols_ - plpad - prpad) * stride + kcols_ - 1;
                            const int irows_ = (orows_ - pupad - pdpad) * stride + krows_ - 1;

                            int lpad = icol < 0 ? -icol : 0;
                            int rpad = icol + icols_ > in_dim ? icol + icols_ - in_dim : 0;
                            int upad = irow < 0 ? -irow : 0;
                            int dpad = irow + irows_ > in_dim ? irow + irows_ - in_dim : 0;

                            const elem_t * weights_slice = weights + (ch*kernel_dim + krow) * kernel_dim + kcol;

                            const elem_t * in = input + (b*in_dim*in_dim + (irow+upad)*in_dim + (icol+lpad)) * channels + ch;

                            sp_tiled_conv_dw(
                                batch_size, in_dim, channels,
                                out_dim, pool_out_dim,

                                stride, padding, kernel_dim,

                                pool_size, pool_stride, pool_padding,

                                batches_,
                                porows_, pocols_,
                                krows_, kcols_,

                                lpad, rpad, upad, dpad,
                                plpad, prpad, pupad, pdpad,

                                in,
                                weights_slice,
                                out,
                                bias_,

                                act, scale,

                                no_bias, no_pool);

                        }
                    }
                }
            }
        }
    }
}

static void tiled_conv_auto(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,

        int in_stride, int weight_stride, int out_stride,
        bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,

        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        elem_t * input,
        elem_t * weights,
        acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,
        int num_array, int cid, 

        float alpha, int real_cycle) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

    bool div_orow = false;
    bool div_och = false;
    int orow_divide = 1;
    int och_divide = 1;
    int out_channels_save = out_channels;
    int pool_out_row = pool_out_dim;
    //if(out_channels >= num_array * DIM * 2){
    if(out_channels >= num_array * DIM * MAX_BLOCK_LEN || !no_pool){
      out_channels = ceil_divide_int(out_channels, num_array);
      div_och = true;
      och_divide = num_array;
      if(out_channels % DIM != 0){
        out_channels = ceil_divide_int(out_channels, DIM) * DIM;
      }
    }
    else if(out_channels >= num_array * DIM && in_dim < num_array * ((DIM*3)/4)){
      out_channels = ceil_divide_int(out_channels, num_array);
      div_och = true;
      och_divide = num_array;
      if(out_channels % DIM != 0){
        out_channels = ceil_divide_int(out_channels, DIM) * DIM;
      }
    }
    else{
      div_orow = true;
      orow_divide = num_array;
      pool_out_row = ceil_divide_int(pool_out_row, num_array);
      
    }


    const bool downsample = false;//stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_dim % 2 == 0;
    //int args[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
 //   int args_in[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
    //int real_cycle = 0;
    int alpha_scale = alpha >= 0 ? (int)(alpha * 100) : -1;
    int args[11] = {batch_size, in_dim, in_channels, out_channels, out_dim, kernel_dim, pool_out_dim, pool_out_row, 0, real_cycle, alpha_scale};
    int* tile_args;
    tile_args = tiled_conv_bubble_calculate(args, stride, kernel_dilation, padding, pool_size, pool_stride, pool_padding, pool_ceil_dim, orow_divide, och_divide, num_array, cid, true, false); 

    //tile_args = tiling_factor_calculate(args, stride, pool_size, pool_stride, kernel_dilation, padding);

    int batches = tile_args[0];
    int porows = tile_args[1];
    int pocols = tile_args[2];
    int pochs = tile_args[3];
    int krows = tile_args[4];
    int kcols = tile_args[5];
    int kchs = tile_args[6];

    int epoch = tile_args[7];
    int max_req = tile_args[8];
    int prediction = tile_args[9];
    alpha = tile_args[10];

/* 
    printf("batches = %d\n", batches);
    printf("orows   = %d\n", porows);
    printf("ocols   = %d\n", pocols);
    printf("ochs    = %d\n", pochs);
    printf("krows   = %d\n", krows);
    printf("kcols   = %d\n", kcols);
    printf("kchs    = %d\n\n", kchs);
*/
/*  
    printf("total spad_rows reserved: %d\n", spad_rows);
    printf("total acc_rows reserved: %d\n\n", acc_rows);

    printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
    printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);

    printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
  */ 
    const int out_offset = (och_divide > 1) ? out_channels * cid : 0;
    bool no_bias = (bias == NULL);
    tiled_conv(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,

        in_stride, weight_stride, out_stride,
        in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        porows, pocols, pochs,
        krows, kcols, kchs,

        input,
        weights + out_offset,
		no_bias ? NULL : (acc_t*) bias + out_offset,
        output + out_offset,

        act, scale, relu6_shift,
        pool_size, no_pool ? 0 : pool_stride, pool_padding, pool_ceil_dim,

        WS,
        //div_orow, div_och,
        orow_divide, cid);
}

static void tiled_conv_default(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int kernel_dilation, int padding, int kernel_dim,
        int out_stride,
        //int in_stride, int weight_stride, int out_stride,
        bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,

        //bool wrot180, bool trans_output_1203, bool trans_input_3120,
        //bool trans_weight_1203, bool trans_weight_0132,

        elem_t * input,
        elem_t * weights,
        acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,
        int num_array, int cid) {

    int bank_len = DIM * MAX_BLOCK_LEN;
    int in_stride = (in_channels % (2*bank_len) == 0) ? in_channels + bank_len : in_channels;
    int weight_stride = (out_channels % (2*bank_len) == 0) ? out_channels + bank_len : out_channels;

    if (in_channels == 3) in_stride = 64; 

    //printf("conv\n");
    tiled_conv_auto(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, 1, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

        false, false, false,
        false, false,
        
        input, weights, bias, output,

        act, scale, relu6_shift,
        pool_size, pool_stride, pool_padding, pool_ceil_dim,
        
        num_array, cid,
        -1, -1);

}
// This function is for a convolution with kernel_dim=1, stride==2, padding=0, and no pooling
static void tiled_conv_downsample(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,

        int in_stride, int weight_stride, int out_stride,
        bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram, 

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,

        enum tiled_matmul_type_t tiled_conv_type) {

    const int stride = 2;

    for (int b = 0; b < batch_size; b++) {
        for (int irow = 0; irow < in_dim; irow += stride) {
            const int orow = irow / stride;

            const int I = in_dim / stride; // number of columns in row
            const int J = out_channels;
            const int K = in_channels;

            const elem_t * A = input + (b*in_dim + irow)*in_dim*in_channels;
            const elem_t * B = weights;
            const acc_t * D = bias;
            elem_t * C = output + (b*out_dim + orow)*out_dim*out_channels;

            const int A_stride = in_stride * 2;
            const int B_stride = weight_stride;
            const int D_stride = out_stride;
            const int C_stride = out_stride;

            tiled_matmul_auto(I, J, K, A, B, (void*)D, (void*)C,
                    A_stride, B_stride, D_stride, C_stride,
                    in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, act, scale, relu6_shift,
                    true, false, false, false, false, 0, tiled_conv_type);
        }
    }
}

//for mobilenet's depthwise convs
static void tiled_conv_dw_auto(
    int batch_size, int in_dim, int channels, int out_dim,
    int stride, int padding, int kernel_dim,

    elem_t * input,
    elem_t * weights,
    acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_dim, pool_out_dim, 1, kernel_dim, kernel_dim, 1};
    const int max_args[] = {batch_size, pool_out_dim, pool_out_dim, 1, kernel_dim, kernel_dim, 1};

    const int orows_idx = 1;
    const int ocols_idx = 2;
    const int out_channels_idx = 3;

    // We divide by 2 for the sake of double-buffering
    const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
    const int max_acc_rows = (ACC_ROWS / 2);

    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
        int max_val = -1;
        int max_idx = -1;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            // We avoid reducing ocols when possible to keep the spatial array fully utilized
            if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1)
                    && args[i] > max_val) {
                max_val = args[i];
                max_idx = i;
            }
        }

        if (max_idx == out_channels_idx) {
            // For input and output channels, there's no point in subtracting by just one
            if (args[max_idx] % DIM != 0) {
                args[max_idx] = (args[max_idx] / DIM) * DIM;
            } else {
                args[max_idx] -= DIM;
            }
            args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
        } else {
            args[max_idx]--;
        }

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    // Check if we can increase ocols
    bool not_increased = false;
    while (!not_increased) {
        not_increased = true;

        int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        args_candidate[ocols_idx]++;

        if (args_candidate[ocols_idx] > max_args[ocols_idx])
            continue;

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

        if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
            args[ocols_idx] = args_candidate[ocols_idx];
            not_increased = false;
        }
    }

    // Check if there are any parameters that we can currently still increase
    bool nothing_increased = false;
    while (!nothing_increased) {
        nothing_increased = true;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            args_candidate[i]++;

            if (args_candidate[i] > max_args[i])
                continue;

            spad_rows = tiled_conv_total_spad_rows(false,
                stride, 1, 1, false, false, false,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
            acc_rows = tiled_conv_total_spad_rows(true,
                stride, 1, 1, false, false, false,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

            if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
                args[i] = args_candidate[i];
                nothing_increased = false;
            }
        }
    }

    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];

    /*
    spad_rows = tiled_conv_total_spad_rows(false,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    printf("batches = %d\n", batches);
    printf("orows   = %d\n", orows);
    printf("ocols   = %d\n", ocols);
    printf("ochs    = %d\n", ochs);
    printf("krows   = %d\n", krows);
    printf("kcols   = %d\n", kcols);
    printf("kchs    = %d\n\n", kchs);

    printf("total spad_rows reserved: %d\n", spad_rows);
    printf("total acc_rows reserved: %d\n\n", acc_rows);

    printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
    printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);

    printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
    */

    tiled_conv_dw(
        batch_size, in_dim, channels, out_dim,
        stride, padding, kernel_dim,

        batches,
        orows, ocols, ochs,
        krows, kcols,

        input,
        weights,
        bias,
        output,

        act, scale, relu6_shift,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        tiled_conv_type);
}

int* tiled_resadd_bubble_calculate(
    int out_args[], // window, bubble, ideal cycles, tiling factors
    size_t I, size_t J,
    size_t num_array, size_t cid, bool A_cached, 
    int real_cycle, float alpha){

  int orow_divide = num_array;
  //bool row_divisible = orow_divide > 1 && (I % orow_divide == 0);
  //I = (row_divisible) ? I / orow_divide : I;
  //size_t och_divide = (row_divisible) ? 1 : orow_divide; // if row is divisible, no need to divide channel
  I = ceil_divide_int(I, orow_divide);
  //if (!row_divisible) orow_divide = 1;

  size_t tile_I = I;
  //J = J / och_divide;
  size_t tile_J = J;

  if(J < MAX_BLOCK_LEN * DIM){
    tile_J = J;
  }
  else{
    tile_J = MAX_BLOCK_LEN * DIM;
  }

	 
	size_t total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));

  // TODO this is a very inefficient way of doing this...
  while (total_acc_rows > ACC_ROWS) {
       if (tile_I >= tile_J)
           tile_I--;
       else
           tile_J -= DIM;

    total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
  }
  int epoch = 0;
  int max_req = 0;
  int prediction = 0;

#if CALC_MEM == 1
  uint64_t total_from_dram = I * (ceil_divide_int(J, DIM)) * 2;
  if (total_from_dram > CACHE_SIZE) total_from_dram += I * (ceil_divide_int(J, DIM));
  // computing window, target load
  //int window = 0;
 // int target_load = 0;
  int total_mem = (I * J * 3)/DIM;
  int total_load = (int)((I*J*2) / DIM);
  int total_store = (int)((I*J)/DIM);
  int num_tile = ceil_divide_int(I, tile_I) * ceil_divide_int(J, tile_J);
	//printf("total macs: %d, number of tile: %d, tile_I: %d, tile_J: %d \n", total_mem, num_tile, tile_I, tile_J);

  int effective_l2_bw = CACHE_BANKS > num_array ? num_array : CACHE_BANKS;
  int l2_mem = total_mem / effective_l2_bw;
  int l2_mem_load = total_load / effective_l2_bw;
  int l2_mem_store = total_store / effective_l2_bw;
  int from_dram = (A_cached) ? total_load / 2 : total_load;
  float effective_dram_bw = DRAM_BW > (NUM_DRAM_BYTE * num_array) ? (NUM_DRAM_BYTE * num_array) : DRAM_BW;
  int dram_cycle = from_dram / effective_dram_bw;

  prediction = l2_mem_store;
  if (A_cached){
    prediction += (l2_mem_load / 2);
    if(dram_cycle > (l2_mem_load/2)) prediction += dram_cycle;
    else prediction += (l2_mem_load/2);
  }
  else{
    if(dram_cycle > l2_mem_load) prediction += dram_cycle;
    else prediction += l2_mem_load;
  }

  if(mode == 2){
    int alpha_scale = (int)((100*real_cycle)/prediction);
    out_args[5] = alpha_scale;
    alpha = alpha_scale / 100;
    int workload_id = workload_running[cid];
    int num_array_index = 0;
    if(num_array == 2) num_array_index = 1;
    else if(num_array == 4) num_array_index = 2; 
    sp_layer_alpha[num_array_index][workload_id][layer_pointer[cid]] = alpha_scale;
    sp_layer_from_dram[num_array_index][workload_id][layer_pointer[cid]] = from_dram;
    sp_layer_compute_ideal[num_array_index][workload_id][layer_pointer[cid]] = 0;
    sp_layer_mem_ideal[num_array_index][workload_id][layer_pointer[cid]] = prediction;
  }
  prediction = prediction * alpha;
  //num_tile /= 2;
  epoch = prediction / num_tile;
  int elem_t_bits = (int)(16 / DIM);
  max_req = (int)(total_mem /num_tile);
  //max_req = elem_t_bits * max_req; 

  // real cycle not achievable
  if(real_cycle < prediction && real_cycle > 0){
      epoch = 0;
      max_req = 0;
  }
  else if(real_cycle > prediction && real_cycle > 0){
      epoch = real_cycle / num_tile;
  }
  else if(real_cycle < 0){
      epoch = 0;
      max_req = 0;
  }
#endif

#if PRINT_MEM == 1
  printf("epoch: %d, max_req: %d, prediction cycles: %llu, real_cycle: %llu \n", epoch, max_req, prediction, real_cycle);
  // for pre-compilation 
  //printf("ideal prediction cycles: %llu, expected dram bw x 100: %d, ideal dram bw util: %d, real dram util: %d \n", ideal_prediction, ideal_dram_bw_exp, ideal_dram_util, dram_util);
  printf("total from dram resadd: %d, number of tile: %d, total mems: %d\n", total_from_dram, num_tile, total_mem);
//  printf("resadd A size: %d, B size: %d, C size: %d, number of tile: %d, target load per tile: %d\n\n", A_size, A_size, A_size, num_tile, target_load);
#endif

  out_args[0] = num_tile;//epoch;
  out_args[1] = total_mem;//max_req;
  out_args[2] = prediction;
  out_args[3] = tile_I;
  out_args[4] = tile_J;

  return out_args;
}

static void resadd_cpu(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            acc_t result = MVIN_SCALE(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}


static void sp_tiled_resadd(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const elem_t * A, const elem_t * B, elem_t * C,
        size_t A_row_stride, size_t B_row_stride, size_t C_row_stride,
        bool relu) {

    // Use the new mvin2 command to overlap mvin A, mvin B, and mvout C

    size_t blocks = (J/DIM + (J % DIM != 0));
    if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

    const size_t rounded_up_J = (J / DIM + (J % DIM != 0)) * DIM;

    // LD/ST
    // dram_addr, sp_addr, loop bounds, stride

    // Mvin A
    // printf("Mving A\n");
/*
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const A_dram_addr = A + i * A_row_stride + j;
            const uint32_t A_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;

            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }
    }
*/

    gemmini_loop_one(A, 3, A_row_stride, I, J, rounded_up_J/DIM, 1); 
    // Mvin B
    // printf("Mving B\n");
/*
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const B_dram_addr = B + i * B_row_stride + j;
            const uint32_t B_sp_addr = C_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
        }
    }
*/
    gemmini_loop_one(B, 2, B_row_stride, I, J, rounded_up_J/DIM, 2);

// Mvout C from accumulator
    // printf("Mvout C from accumulator\n");
/*
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            elem_t * const C_dram_addr = C + i * C_row_stride + j;
            const uint32_t C_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvout(C_dram_addr, C_sp_addr, cols, rows);
        }
    }
*/
    gemmini_loop_one(C, 2, C_row_stride, I, J, rounded_up_J/DIM, 0);
}

// Compute MVIN_SCALE(A, A_scale) + MVIN_SCALE(B, B_scale) = C
static void tiled_resadd(const size_t I, const size_t J, const size_t stride,
        bool A_direct_dram, bool B_direct_dram, bool C_direct_dram,
        const size_t tile_I, const size_t tile_J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

    gemmini_extended_config_st(C_direct_dram, stride * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
    gemmini_config_ex(WS, 0, 0);

    gemmini_extended4_config_ld(A_direct_dram, stride * sizeof(elem_t), A_scale, true, DIM, 0);
    gemmini_extended4_config_ld(B_direct_dram, stride * sizeof(elem_t), B_scale, true, DIM, 1);

    for (size_t i = 0; i < I; i += tile_I) {
        for (size_t j = 0; j < J; j += tile_J) {
            const size_t I_tile = i + tile_I <= I ? tile_I : I - i;
            const size_t J_tile = j + tile_J <= J ? tile_J : J - j;

            const elem_t * a = A + i * stride + j;
            const elem_t * b = B + i * stride + j;
            elem_t * c = C + i * stride + j;

            sp_tiled_resadd(I_tile, J_tile,
                    A_scale, B_scale, a, b, c,
                    stride, stride, stride,
                    relu);
        }
    }

    gemmini_fence();
}

// Compute (A >> A_shift) + B = C
static void tiled_resadd_auto(size_t I, size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale, 
        const size_t stride,
        bool A_direct_dram, bool B_direct_dram, bool C_direct_dram,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type,
        int num_array, size_t cid) {

    if (matadd_type == CPU) {
        resadd_cpu(I, J,
            A_scale, B_scale, C_scale, A, B, C,
            relu);
        return;
    }

    int args_in[] = {0, 0, 0, 0, 0};
    int real_cycle = -1;
    float alpha = 1;
    int * args = tiled_resadd_bubble_calculate(args_in, I, J, num_array, cid, false, real_cycle, alpha);
    size_t tile_I = args[3];
    size_t tile_J = args[4];
    I = ceil_divide_int(I, num_array);
    int orow_offset = stride * cid * I;
/*
    printf("num array: %d, cid: %d\n", num_array, cid);
     printf("tile_I: %llu\n", tile_I);
     printf("tile_J: %llu\n", tile_J);
*/
    if (matadd_type == WS) {
      tiled_resadd(I, J, stride, 
            A_direct_dram, B_direct_dram, C_direct_dram, 
            tile_I, tile_J, 
            A_scale, B_scale, C_scale, A+orow_offset, B+orow_offset, C+orow_offset,
            relu, matadd_type);
    } else if(matadd_type == CPU){
	  resadd_cpu(I, J, A_scale, B_scale, C_scale,
		A, B, C, relu);
    }
    else {
      printf("Unsupported type\n");
      exit(1);
    }
}

static void tiled_resadd_default(size_t I, size_t J,
    const scale_t A_scale,
    const scale_t B_scale,
    const acc_scale_t C_scale,
    bool A_direct_dram, bool B_direct_dram, bool C_direct_dram,
    const elem_t * A,
    const elem_t * B,
    elem_t * C,
    bool relu,
    size_t num_array, size_t cid){
    //size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id) {
  //printf("resadd A: 0x%08lx, B: 0x%08lx, C: 0x08lx\n", A, B, C);
  size_t J_stride = (J % 128 == 0) ? J + 64 : J;
  
  tiled_resadd_auto(I, J, A_scale, B_scale, C_scale,
      J_stride,
      A_direct_dram, B_direct_dram, C_direct_dram,
      A, B, C,
      relu, WS, 
      num_array, cid);
      //orow_divide, batch_divide, cid, group_id);

}
static void global_average_cpu(const elem_t * input, elem_t * output,
    int batches, int channels, int dim) {
  const int count = dim * dim;

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      acc_t sum = 0;
      for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
          size_t pixel = batch * dim * dim + row * dim + col;

          sum += input[pixel * channels + channel];
        }
      }

#ifdef ELEM_T_IS_FLOAT
      output[batch * channels + channel] = sum / count;
#else
      output[batch * channels + channel] = (sum + count/2) / count;
#endif
    }
  }
}


static void sp_tiled_global_average(const elem_t * input, elem_t * output,
    int batches, int channels, int dim, int channel_tile_size) {
  const uint32_t C_acc_addr_start = ((uint32_t)1 << 31);

  size_t blocks = channel_tile_size/DIM + (channel_tile_size % DIM != 0);
  if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

  for (int channel = 0; channel < channel_tile_size; channel += blocks*DIM) {
    for (int row = 0; row < dim; row++) {
      for (int col = 0; col < dim; col++) {
        const elem_t * in = input +
          (row * dim + col) * channels +
          channel;

        const uint32_t acc_addr_start = C_acc_addr_start |
          ((row != 0 || col != 0) << 30);

        const uint32_t acc_addr = acc_addr_start + channel / DIM;

        const size_t cols = channel + blocks*DIM <= channel_tile_size ?
          blocks*DIM : channel_tile_size - channel;

        const size_t rows = 1;

        gemmini_extended_mvin(in, acc_addr, cols, rows);
      }
    }
  }

  for (int channel = 0; channel < channel_tile_size; channel += DIM) {
    elem_t * out = output + channel;

    const uint32_t acc_addr = C_acc_addr_start + channel / DIM;

    const size_t cols = channel + DIM <= channel_tile_size ?
      DIM : channel_tile_size - channel;

    const size_t rows = 1; // TODO we should move out more than just one row here

    gemmini_extended_mvout(out, acc_addr, cols, rows);
  }
}


static void tiled_global_average(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    int channel_tile_size) {

  gemmini_extended4_config_ld(false, DIM*sizeof(elem_t), MVIN_SCALE_IDENTITY, true, 1, 0);
  gemmini_config_ex(0, NO_ACTIVATION, 0);
  gemmini_extended_config_st(false, 0, NO_ACTIVATION, 1.0 / (dim*dim));

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel += channel_tile_size) {
      const int tile_size = channel + channel_tile_size <= channels ?
        channel_tile_size : channels - channel;

      sp_tiled_global_average(input + batch * dim * dim * channels + channel,
          output + batch * channels + channel,
          batches, channels, dim, tile_size);
    }
  }
}


static void tiled_global_average_auto(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    enum tiled_matmul_type_t type) {
  if (type == CPU) {
    return global_average_cpu(input, output, batches, channels, dim);
  }

  int channel_tile_size = channels;

  int acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  while (acc_rows > ACC_ROWS) {
    channel_tile_size--;
    acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  }

  tiled_global_average(input, output, batches, channels, dim,
      channel_tile_size);
}

static void sp_tiled_pool(
    int batch_size, int in_dim, int channels,
		int pool_out_dim, 
    int pool_size, int pool_stride, int pool_padding,
		int stride,
    bool input_direct_dram, bool output_direct_dram,

    int batches,
    int porows, int pocols, int pochs,
    int plpad, int prpad, int pupad, int pdpad,

    const elem_t * input,
    elem_t * output)
{
    const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
    const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
    const int ochs = pochs;

    int D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
    int C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;

    const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
    const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;
    gemmini_extended2_config_st(input_direct_dram, stride * sizeof(elem_t), 0, 1, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
    gemmini_extended4_config_ld(output_direct_dram, stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, batches * orows * ocols, 2);

  //  gemmini_extended4_config_ld(J_stride * sizeof(elem_t), B_scale, true, DIM, 1);


    const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs : MAX_BLOCK_LEN_ACC * DIM;

	  for (int b = 0; b < batches; b++)
			for (int orow = 0; orow < orows; orow++)
				 for (int ocol = 0; ocol < ocols; ocol += DIM) {
					  const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

					  for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
							const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

							const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

							gemmini_extended_mvin3(input + (b*in_dim*in_dim + orow*in_dim + ocol) * stride + och,
									  D_sp_addr,
									  J, I);
					  }
				 }

		for (int b = 0; b < batches; b++) {
			 for (int poch = 0; poch < pochs; poch += DIM) {
				  const int out_channels = poch + DIM >= pochs ? pochs - poch : DIM;

				  elem_t * const pout = output + (b * pool_out_dim * pool_out_dim)*stride + poch;

				  const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

				  gemmini_extended_mvout(pout,
							 C_sp_addr,
							 out_channels, 0);
			 }
		}

}

static void tiled_pool(
    int batch_size, int in_dim, int channels,
		int pool_out_dim,
		int batches,
    int porows, int pocols, int pochs,
    int out_stride,

    bool input_direct_dram, bool output_direct_dram,

		const elem_t * input,
    elem_t * pool_output,
		  
    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding,

		size_t orow_divide, size_t cid, size_t group_id, int window, int target_load) {

	 //int out_stride = channels * och_divide;

    //gemmini_extended_config_st(out_stride * sizeof(elem_t), RELU, MVIN_SCALE_IDENTITY);
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
//	 int stride = channels*och_divide;
//    gemmini_extended4_config_ld(stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, DIM, 0);

    bool row_divide = (orow_divide > 1);
    int out_row = (row_divide) ? pool_out_dim / orow_divide : pool_out_dim;
    size_t och_cid = (size_t)(cid % orow_divide);
    int porow_start = row_divide ? out_row * och_cid : 0;
    int porow_end = row_divide ? out_row * (och_cid + 1) : pool_out_dim;
 
    for (int poch = 0; poch < channels; poch += pochs) {
       for (int b = 0; b < batch_size; b += batches) {
           for (int porow = porow_start; porow < porow_end; porow += porows) {
               const int orow = porow * pool_stride - pool_padding;
               const int orow_floored = orow < 0 ? 0 : orow;        
               for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
                  const int ocol = pocol * pool_stride - pool_padding;
                  const int ocol_floored = ocol < 0 ? 0 : ocol;
             
                  elem_t * out = pool_output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * out_stride + poch;
                  const elem_t * in = input + (b*in_dim*in_dim + orow_floored*in_dim + ocol_floored) * out_stride + poch;

                  // printf("batch: %d, poch: %d, porow: %d, pocol: %d\n", b, poch, porow, pocol);
                  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                  const int porows_ = porow_end - porow > porows ? porows : porow_end - porow;
                  const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                  const int pochs_ = channels - poch > pochs ? pochs : channels - poch;
                  const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  const int orows_ = porows_ * pool_stride + pool_size - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad = ocol + ocols_ > in_dim ? ocol + ocols_ - in_dim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad = orow + orows_ > in_dim ? orow + orows_ - in_dim : 0;

                 sp_tiled_pool(
                  batch_size, in_dim, channels,
                  pool_out_dim,
                  pool_size, pool_stride, pool_padding,
                  out_stride,
                  input_direct_dram, output_direct_dram,

                  batches_,
                  porows_, pocols_, pochs_,
                  plpad, prpad, pupad, pdpad,

                  in,
                  out);
               }
            }
        }
    }
    gemmini_fence();
}

int* tiled_pool_bubble_calculate(
    int out_args[], // window, bubble, ideal cycles, tiling factors
    int batch_size, int in_dim, int channels,
    int out_dim,
    int pool_size, int pool_stride, int pool_padding,
    bool row_divide, size_t och_divide, size_t batch_divide, size_t cid, size_t group_id){
  
  batch_size = batch_size/batch_divide;
  channels = (row_divide) ? channels : channels / och_divide;

  int out_row = (row_divide) ? out_dim / och_divide : out_dim;
  int args[] = {batch_size, out_row, out_dim, channels, 1, 1, DIM};
  const int max_args[] = {batch_size, out_row, out_dim, channels, 1, 1, DIM};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int channels_idx = 3;
  const int max_spad_rows = (BANK_NUM*BANK_ROWS);
  const int max_acc_rows = (ACC_ROWS);
	const int dilation = 1;
  int acc_rows = tiled_conv_total_spad_rows(true,
        1, dilation, dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (acc_rows > max_acc_rows) {
      int max_val = -1;
      int max_idx = -1;

      for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
          // We avoid reducing ocols when possible to keep the spatial array fully utilized
      if(i == channels_idx && args[i] > MAX_BLOCK_LEN * DIM){
        args[i] = (args[i] - 1) / (MAX_BLOCK_LEN*DIM) * DIM;
        break;
      }
      else if(i == 0 && args[i] > 1){
        args[i] = 1;
        break;
      } // for batch
         else if ((i!=channels_idx) &&  args[i] > max_val) {
              max_val = args[i];
              max_idx = i;
         }
      }
		  args[max_idx]--;
      acc_rows = tiled_conv_total_spad_rows(true,
          1, dilation, dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    const int batches = args[0];
    const int porows = args[1];
    const int pocols = args[2];
    const int pochs = args[3];
  
    int window = 0;
    int target_load = 0;
    int num_tiles = 0;
    int total_load = 0;
    int ideal_cycle = 0;

#if MOCA_en == 1
    size_t och_cid = (size_t)(cid % och_divide);
    int porow_start = row_divide ? out_row * och_cid : 0;
    int porow_end = row_divide ? out_row * (och_cid + 1) : out_dim;
    for (int poch = 0; poch < channels; poch += pochs) {
       for (int b = 0; b < batch_size; b += batches) {
           for (int porow = porow_start; porow < porow_end; porow += porows) {
               const int orow = porow * pool_stride - pool_padding;
               const int orow_floored = orow < 0 ? 0 : orow;        
               for (int pocol = 0; pocol < out_dim; pocol += pocols) {
                  num_tiles += 1;
                  const int ocol = pocol * pool_stride - pool_padding;
                  const int ocol_floored = ocol < 0 ? 0 : ocol;
                  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                  const int porows_ = porow_end - porow > porows ? porows : porow_end - porow;
                  const int pocols_ = out_dim - pocol > pocols ? pocols : out_dim - pocol;
                  const int pochs_ = channels - poch > pochs ? pochs : channels - poch;
                  
                  int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  int orows_ = porows_ * pool_stride + pool_size - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad = ocol + ocols_ > in_dim ? ocol + ocols_ - in_dim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad = orow + orows_ > in_dim ? orow + orows_ - in_dim : 0;

                  ocols_ -= (pupad + pdpad);
                  orows_ -= (plpad + prpad);

                  total_load += (int)(batches_ * ocols_ * orows_ * pochs_ / DIM);
                  ideal_cycle += (int)(batches_ * ocols_ * orows_ * pochs_ / DIM) + (int)(batches_ * porows_ * pocols_ * pochs_ / DIM) * (pool_size * pool_size + 1);
                  //printf("total macs: %d, number of tile: %d, tile_I: %d, tile_J: %d \n", total_mem, num_tile, tile_I, tile_J);
 
               }
           }
       }
    }
  if (target_util != 0){
    int target_cycles = ideal_cycle * 100 / target_util;
    target_load = (int)(total_load / num_tiles) ;
    window = (int)(target_cycles / num_tiles) ;
  }
/*
  // for pre-compilation
  int C_size = batch_size * out_dim * out_dim * (int)(channels / DIM);
  printf("pool total load: %d, C size: %d, number of tile: %d, target load per tile: %d\n\n", total_load, C_size, num_tiles, target_load);
*/
#endif
  out_args[0] = window;
  out_args[1] = target_load;
  out_args[2] = ideal_cycle;
  out_args[3] = args[0];
  out_args[4] = args[1];
  out_args[5] = args[2];
  out_args[6] = args[3];

  return out_args;
}

// pooling using Gemmini DMA
static void tiled_pool_auto(int batch_size, int channels, int in_dim,
    int pool_out_dim, int stride,
    int pool_size, int pool_stride, int pool_padding,
    bool input_direct_dram, bool output_direct_dram,
    const elem_t * A,
    elem_t * C,
    size_t och_divide, size_t batch_divide, size_t cid, size_t group_id) {
  
  bool relu = true;
	//int stride = channels;

  int * args;
  int args_in[] = {0, 0, 0, 0};
  args = tiled_pool_bubble_calculate(args_in, batch_size, in_dim, channels, pool_out_dim, pool_size, pool_stride, pool_padding, 1, 1, 1, 0, 0);
 
  int window = args[0];
  int target_load = args[1]; 
  const int batches = args[3];
  const int porows = args[4];
  const int pocols = args[5];
  const int pochs = args[6];
  //printf("window: %d, target_load: %d \n", window, target_load);

  window = 0;
  target_load = 0; // for now, disable CALM on pooling
  //printf("C dram addr before pool: 0x%08lx\n", C);
  tiled_pool(batch_size, in_dim, channels, pool_out_dim,
				batches, porows, pocols, pochs,
        stride,
        input_direct_dram, output_direct_dram, 
        A, C, 
				RELU, MVIN_SCALE_IDENTITY, 0,
				pool_size, pool_stride, pool_padding,
				1, 0, 0, window, target_load);
  
  //printf("C dram addr after pool: 0x%08lx\n", C);
}
#undef abs

#endif // SRC_MAIN_C_GEMMINI_H

