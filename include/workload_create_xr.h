
// SET 0: mixed, SET 1: light, SET 2: heavy
// SET 0: mixed, SET 1: light, SET 2: heavy, 3: xr
#if SET == 3
#include "imagenet/funct_ritnet.h"
#include "imagenet/funct_midasnet.h"
#include "imagenet/funct_rcnnnet.h"
#include "imagenet/funct_handnet.h"
#include "imagenet/funct_fbnet.h"
#endif
#if SET == 2
#include "imagenet/funct_resnet_1.h"
#include "imagenet/funct_alexnet_1.h"
#include "imagenet/funct_yolonet_1.h"
#include "imagenet/funct_googlenet_1.h"
#include "imagenet/funct_bertbase_1.h"
#endif
#if SET == 1
#include "imagenet/funct_res18net_1.h"
#include "imagenet/funct_squeezenet_1.h"
#include "imagenet/funct_kwsnet_1.h"
#include "imagenet/funct_yololitenet_1.h"
// actually bert small
#include "imagenet/funct_bertmedium_1.h"
#endif
#if SET == 0
#include "imagenet/funct_resnet_1.h"
#include "imagenet/funct_alexnet_1.h"
#include "imagenet/funct_bertbase_1.h"
#include "imagenet/funct_res18net_1.h"
#include "imagenet/funct_squeezenet_1.h"
//#include "imagenet/funct_kwsnet_1.h"
#include "imagenet/funct_yololitenet_1.h"
//#include "imagenet/funct_bertmedium_1.h"
#endif



int rand_create(bool init) {
  static uint32_t x = 777;
  if(init) x = 777;
  x = x * (1664525) + 1013904223;
  return x >> 24;
}

void workload_create(int num_workload, float target_scale){ 
  // qos < 0 -> mixed
  // qos >= 0 -> workload dispatch qos apart, qos ways at once
  for(int i = 0; i < MAX_WORKLOAD; i++)
    total_queue_status[i]= -1;

  rand_create(true); // initialize random function

  int num_type = 5;

  int first_dispatch_interval = (int)((1e9/(60*num_type)));
  printf("first dispatch interval: %d\n", first_dispatch_interval);
  int num_workload_group = ceil_divide_int(num_workload*2, num_type); // assign 2x
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  //printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale))) ;
  
  for(int i = 0; i < num_workload_group; i++){
    for(int j = 0; j < num_type; j++){
      int index = num_type * i + j;
      //int workload_type = rand_base + rand_seed(seed) % rand_mod;
      int workload_type = 10 + j;
      total_queue_type[index] = workload_type; 
      total_queue_target[index] = target_scale * target_cycles[workload_type];
      for (int j = 0; j < NUM_CORE; j++){
         total_queue_finish[index] = 0;
         total_queue_runtime[index] = 0;
      }
      if(i == 0){
        total_queue_dispatch[index] = first_dispatch_interval*j;
      }
      else{
	uint64_t jitter = rand_create(false) % 200000;
	uint64_t this_interval = (uint64_t)(((target_cycles[workload_type] - 100000) + jitter) * target_scale);
	//printf("index %d interval: %llu\n", index, this_interval);
        total_queue_dispatch[index] = total_queue_dispatch[index - num_type] + this_interval;
	if(i % 20 == 13) total_queue_dispatch[index] += (target_cycles[workload_type] * target_scale); // to prevent overloading
      }
    }
  }
  
  for(int i = 0; i < num_workload; i++){
    for(int j = i+1; j < num_workload*2; j++){
      if(total_queue_dispatch[i] > total_queue_dispatch[j]){
        uint64_t a = total_queue_dispatch[i];
        total_queue_dispatch[i] = total_queue_dispatch[j];
        total_queue_dispatch[j] = a;
 
        a = total_queue_target[i];
        total_queue_target[i] = total_queue_target[j];
        total_queue_target[j] = a;
 
        int b = total_queue_type[i];
        total_queue_type[i] = total_queue_type[j];
        total_queue_type[j] = b;
                     
      }
    }
  }
  for(int i = num_workload; i < num_workload*2; i++){
    total_queue_dispatch[i] = 0;
    total_queue_type[i] = -1;
    total_queue_status[i] = -1;
  }
  bool workload_class[NUM_TYPE_WORKLOAD] = {0};
  /*
  for(int i = 0; i < num_workload; i++){
      int type = total_queue_type[i];
      if (i > 0 && i < num_workload -1){
#if FULL == 1
          if(type == total_queue_type[i-1] && type == total_queue_type[i+1]){
#else
          if(type == total_queue_type[i-1]) {
#endif
              while(type == total_queue_type[i]){
                type = workload_type_assign(seed);
              }
              printf("new type: %d, old type: %d\n", type, total_queue_type[i]);
              total_queue_type[i] = type;
              total_queue_target[i] = target_scale * target_cycles[type];
          }
      }
      if(workload_class[type]){
          workload_class[type] = false;
          total_queue_class[i] = false;
      }
      else{
          workload_class[type] = true;
          total_queue_class[i] = true;
      }
  }
   */ 
  //for(int i = 0; i < workload; i++)
    //printf("after mixing entry %d, workload id %d\n", i, total_queue_type[i]);

  for(int i = 0; i < NUM_CORE; i++){
    workload_running[i] = -1; // initialize state 
    current_score[i] = -1;
    for(int j = 0; j < NUM_ARRAY; j++)
      core_gemmini[i][j] = -1;
  }
  for(int i = 0; i < NUM_ARRAY; i++){
    gemmini_status[i] = -1;
  }
  for(int i = 0; i < QUEUE_DEPTH; i++){
    ex_queue[i] = -1;
  }
  mode = 3; // for rerocc
}

void workload_init(float target_scale, int num_workload){
  for(int i = 0; i < NUM_ARRAY; i++){
    gemmini_status[i] = -1;
  }
  for(int i = 0; i < QUEUE_DEPTH; i++){
    ex_queue[i] = -1;
  }

  global_end = false;
  last_queue_id = 0;

  for(int i = 0; i < NUM_CORE; i++){
    workload_running[i] = -1; // initialize state 
    current_score[i] = -1;
    layer_pointer[i] = 0;
    core_num_gemmini[i] = 0;
    core_num_gemmini_share[i] = 0;
    global_start_time[i] = 0;
    dram_util[i] = 0;
    current_mem_score[i] = 0;
    noc_optim_need_swap[i] = -1;
    for(int j = 0; j < NUM_ARRAY; j++)
      core_gemmini[i][j] = -1;
  }

  for(int i = 0; i < num_workload; i++){
    total_queue_status[i] = -1;
    total_queue_runtime[i] = 0;
    total_queue_finish[i] = 0;
    total_queue_core[i] = -1;
    total_queue_throttle[i] = 0;
    total_queue_release[i] = 0;
    total_queue_acquire[i] = 0;
    total_queue_overhead[i] = 0;
    total_queue_swap[i] = 0;
    total_queue_target[i] = 0;
  }

  for(int i = 0; i < num_workload; i++){
      int workload_type = total_queue_type[i];
      total_queue_target[i] = target_scale * target_cycles[workload_type];
  }
}
