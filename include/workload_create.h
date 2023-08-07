
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

int workload_type_assign(){
  // currently only batch1
// heavy
#if SET == 2
  int rand_mod = RESNUM + ALEXNUM + YOLONUM + GOOGLENUM + BERTBASENUM;
  int rand_base = 0;

  static int id = 1;
  uint32_t rand_out = rand_create(false);
  int r = rand_out % rand_mod + rand_base;

  if (r < ALEXNUM)
      id = ALEXNET;
  else if (r < ALEXNUM + RESNUM)
      id = RESNET;
  else if (r < ALEXNUM + RESNUM + YOLONUM)
      id = YOLONET;
  else if (r < ALEXNUM + RESNUM + YOLONUM + GOOGLENUM)
      id = GOOGLENET;
  else
      id = BERTBASE;
#elif SET == 1
  // light
  int rand_mod = RES18NUM + KWSNUM + SQUEEZENUM + YOLOLITENUM + BERTSMALLNUM;
  int rand_base = 0;

  static int id = 1;
  uint32_t rand_out = rand_create(false);
  int r = rand_out % rand_mod + rand_base;

  if (r < KWSNUM)
      id = KWSNET;
  else if (r < KWSNUM + RES18NUM)
      id = RES18NET;
  else if (r < KWSNUM + RES18NUM + SQUEEZENUM)
      id = SQUEEZENET;
  else if (r < KWSNUM + RES18NUM + SQUEEZENUM + YOLOLITENUM)
      id = YOLOLITENET;
  else
      id = BERTSMALL;
#else 
  // mixed
  int rand_mod = RESNUM + ALEXNUM + SQUEEZENUM + YOLOLITENUM + RES18NUM + BERTBASENUM;
  int rand_base = 0;

  static int id = 1;
  uint32_t rand_out = rand_create(false);
  int r = rand_out % rand_mod + rand_base;

  if (r < ALEXNUM)
      id = ALEXNET;
  else if (r < ALEXNUM + RESNUM)
      id = RESNET;
  else if (r < ALEXNUM + RESNUM + SQUEEZENUM)
      id = SQUEEZENET;
  else if (r < ALEXNUM + RESNUM + SQUEEZENUM + YOLOLITENUM)
      id = YOLOLITENET;
  else if (r < ALEXNUM + RESNUM + SQUEEZENUM + YOLOLITENUM + RES18NUM)
      id = RES18NET;
  else
      id = BERTBASE;
  
#endif
  //printf("rand output: %zu, rand output value: %d, workload id: %d \n", rand_out, r, id);
  return id;
}

void workload_create(int num_workload, float target_scale, float cap_scale){ 
  // qos < 0 -> mixed
  // qos >= 0 -> workload dispatch qos apart, qos ways at once
  for(int i = 0; i < MAX_WORKLOAD; i++)
    total_queue_status[i]= -1;

  rand_create(true); // initialize random function

#if SET == 1
  uint64_t interval = (rand_cycles[4]*KWSNUM + rand_cycles[5]*RES18NUM + rand_cycles[6]*SQUEEZENUM + rand_cycles[7]*YOLOLITENUM + rand_cycles[9]*BERTSMALLNUM) / (KWSNUM+RES18NUM+SQUEEZENUM+YOLOLITENUM+BERTSMALLNUM);
#elif SET == 2
  uint64_t interval = (rand_cycles[0]*ALEXNUM + rand_cycles[1]*RESNUM + rand_cycles[2]*YOLONUM + rand_cycles[3]*GOOGLENUM + rand_cycles[8]*BERTBASENUM) / (ALEXNUM+RESNUM+YOLONUM+GOOGLENUM+BERTBASENUM);
#else 
  uint64_t interval = (rand_cycles[0]*ALEXNUM + rand_cycles[1]*RESNUM + rand_cycles[5]*RES18NUM + rand_cycles[6]*SQUEEZENUM + rand_cycles[7]*YOLOLITENUM  + rand_cycles[8]*BERTBASENUM) / (ALEXNUM+RESNUM+RES18NUM+SQUEEZENUM+YOLOLITENUM+BERTBASENUM);
#endif


//#if NOC == 1
  int group = 4;//(int)(NUM_ARRAY / 2);
//#else
//  int group = (int)(NUM_ARRAY / 2);
//#endif
  int first_dispatch_interval = (int)(interval / group);
  printf("interval for set %d: %llu, first dispatch interval: %d\n", SET, interval, first_dispatch_interval);
  int num_workload_group = ceil_divide_int(num_workload+2*group, group);
  for(int s = 0; s < SEED; s++){
    printf("interval test: %d\n", (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale)));
  }
  for(int i = 0; i < num_workload_group; i++){
    for(int j = 0; j < group; j++){
      int index = group * i + j;
      int workload_type = workload_type_assign();
      //int workload_type = rand_base + rand_seed(seed) % rand_mod;
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
	uint64_t this_interval = (int)(interval * (0.01*(rand_create(false)%INTERVAL) + cap_scale));
	//printf("index %d interval: %llu\n", index, this_interval);
        total_queue_dispatch[index] = total_queue_dispatch[index - group] + this_interval;
	if(i % 15 == 12) total_queue_dispatch[index] += (interval*2.1); // to prevent overloading
        //total_queue_dispatch[index] = total_queue_dispatch[index - group] + rand_cycles[total_queue_type[index - group]] * (0.1*(rand_create(false)%5) + cap_scale); // is it enough?
      }
    }
  }
  
  for(int i = 0; i < num_workload; i++){
    for(int j = i+1; j < num_workload+2*group; j++){
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
  for(int i = num_workload; i < num_workload+2*group; i++){
    total_queue_dispatch[i] = 0;
    total_queue_type[i] = -1;
    total_queue_status[i] = -1;
  }
  bool workload_class[NUM_TYPE_WORKLOAD] = {0};
  for(int i = 0; i < num_workload; i++){
      int type = total_queue_type[i];
      if (i > 0 && i < num_workload -1){
//#if FULL == 1
//          if(type == total_queue_type[i-1] && type == total_queue_type[i+1]){
//#else
          if(type == total_queue_type[i-1]) {
//#endif
              while(type == total_queue_type[i]){
                type = workload_type_assign();
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
    total_queue_dropped[i] = false;
  }

  for(int i = 0; i < num_workload; i++){
      int workload_type = total_queue_type[i];
      total_queue_target[i] = target_scale * target_cycles[workload_type];
  }
}
