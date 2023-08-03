#if SET == 3
#include "include/workload_create_xr.h"
#else
#include "include/workload_create.h"
#endif

/*
int rand_seed(uint32_t seed, bool init) {
  static uint32_t x = 777;
  if(init) x = 777;
  x = x * (1664525 + seed) + 1013904223;
  return x >> 24;
}

int workload_type_assign(uint32_t seed){
  // currently only batch1
// heavy
#if SET == 2
  int rand_mod = RESNUM + ALEXNUM + YOLONUM + GOOGLENUM + BERTBASENUM;
  int rand_base = 0;

  static int id = 1;
  uint32_t rand_out = rand_seed(seed, false);
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
  uint32_t rand_out = rand_seed(seed, false);
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
  uint32_t rand_out = rand_seed(seed, false);
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

void workload_create(int num_workload, uint32_t seed, float target_scale, float cap_scale){ 
  // qos < 0 -> mixed
  // qos >= 0 -> workload dispatch qos apart, qos ways at once
  for(int i = 0; i < MAX_WORKLOAD; i++)
    total_queue_status[i]= -1;

  rand_seed(seed, true); // initialize random function
#if SET == 1
  uint64_t interval = (rand_cycles[4]*KWSNUM + rand_cycles[5]*RES18NUM + rand_cycles[6]*SQUEEZENUM + rand_cycles[7]*YOLOLITENUM + rand_cycles[9]*BERTSMALLNUM) / (KWSNUM+RES18NUM+SQUEEZENUM+YOLOLITENUM+BERTSMALLNUM);
#elif SET == 2
  uint64_t interval = (rand_cycles[0]*ALEXNUM + rand_cycles[1]*RESNUM + rand_cycles[2]*YOLONUM + rand_cycles[3]*GOOGLENUM + rand_cycles[8]*BERTBASENUM) / (ALEXNUM+RESNUM+YOLONUM+GOOGLENUM+BERTBASENUM);
#else 
  uint64_t interval = (rand_cycles[0]*ALEXNUM + rand_cycles[1]*RESNUM + rand_cycles[5]*RES18NUM + rand_cycles[6]*SQUEEZENUM + rand_cycles[7]*YOLOLITENUM  + rand_cycles[8]*BERTBASENUM) / (ALEXNUM+RESNUM+RES18NUM+SQUEEZENUM+YOLOLITENUM+BERTBASENUM);
#endif

  
  int group = 4; //(int)(NUM_ARRAY / 2);
  int first_dispatch_interval = (int)(interval / group);
  printf("interval for set %d: %llu, first dispatch interval: %d\n", SET, interval, first_dispatch_interval);
  int num_workload_group = ceil_divide_int(num_workload+2*group, group);
  printf("interval test: %d\n", (int)(interval * (0.01*(rand_seed(seed, false)%INTERVAL) + cap_scale))) ;
  printf("interval test: %d\n", (int)(interval * (0.01*(rand_seed(seed, false)%INTERVAL) + cap_scale))) ;
  printf("interval test: %d\n", (int)(interval * (0.01*(rand_seed(seed, false)%INTERVAL) + cap_scale))) ;
  printf("interval test: %d\n", (int)(interval * (0.01*(rand_seed(seed, false)%INTERVAL) + cap_scale))) ;
  printf("interval test: %d\n", (int)(interval * (0.01*(rand_seed(seed, false)%INTERVAL) + cap_scale))) ;
  for(int i = 0; i < num_workload_group; i++){
    for(int j = 0; j < group; j++){
      int index = group * i + j;
      int workload_type = workload_type_assign(seed);
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
	uint64_t this_interval = (int)(interval * (0.01*(rand_seed(seed, false)%INTERVAL) + cap_scale));
	printf("index %d interval: %llu\n", index, this_interval);
        total_queue_dispatch[index] = total_queue_dispatch[index - group] + this_interval;
       // total_queue_dispatch[index] = total_queue_dispatch[index - group] + rand_cycles[total_queue_type[index - group]] * (0.1*(rand_seed(seed, false)%5) + cap_scale); // is it enough?
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
    
  //for(int i = 0; i < workload; i++)
    //printf("after mixing entry %d, workload id %d\n", i, total_queue_type[i]);

  for(int i = 0; i < NUM_CORE; i++){
    gemmini_reserved_vel[i] = 0;
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
}
*/
// get max dynamic score index from ex_queue
#if SET == 3
int get_max_index(int cid, uint64_t current_cycle, bool drop_enable){
#else
int get_max_index(int cid, uint64_t current_cycle){
#endif
    int max_score = -1;
    int max_index = -1;
    for(int i = 0; i < QUEUE_DEPTH; i++){
      if(ex_queue[i] >= 0){
        int queue_id = ex_queue[i];
        int workload_type = total_queue_type[queue_id];
        int64_t old = current_cycle - total_queue_dispatch[queue_id];
        if(old < 0) continue; // for syscall printf
        int slack = target_cycles[workload_type] - old;
#if SET == 3
	if(slack <= 0 && drop_enable){
	//if(drop_enable && (slack <= 0 || slack < (sp_cycles[2][workload_type]/4))) {
        //if(slack < 0){
            total_queue_status[queue_id] = MAX_LAYERS;
            total_queue_finish[queue_id] = total_queue_dispatch[queue_id] + 1;
	    total_queue_dropped[queue_id] = true;
            ex_queue[i] = -1;
#if PRINT == 1
            printf("task %d gets aborted by cid %d with slack %d \n", queue_id, cid, slack);
#endif
            continue;
        } 
#endif
        int num_array_index = 2;
        uint64_t expected_cycle = sp_cycles[num_array_index][workload_type];
        int64_t this_score = (1000000 * old) / expected_cycle;
	    if(slack < expected_cycle || slack <= 0)
	      this_score = 0;
        //int64_t this_score = (10000 * expected_cycle) / slack;
#if PRINT == 1
        //printf("this score: %d, old: %d, expected cycle: %d \n", this_score, old, expected_cycle);
#endif
        if (max_score < this_score) {
            max_score = this_score;
            max_index = i;
        }
      }
    }
    if(max_index != -1){
      int num_array = 1;
      int queue_id = ex_queue[max_index];
      int workload_type = total_queue_type[queue_id];
      uint64_t old = current_cycle - total_queue_dispatch[queue_id];
      for(int i = 2; i >= 0; i --){
        int slack = target_cycles[workload_type] - old;
#if NOC == 1
        if (slack < (2*sp_cycles[i][workload_type])){
#else
        if (slack < (1.5*sp_cycles[i][workload_type])){
#endif
          num_array = (1 << i);
          break;
        }
      }
      core_num_gemmini[cid] = num_array;
      if(mode == 5){
        if (workload_type == YOLOLITENET) {
          //if (num_array > 2)
              core_num_gemmini[cid] = 2;
	}
      }
      else
	if(num_array < 2)
	  core_num_gemmini[cid] = 2;
    }
    else
      core_num_gemmini[cid] = 0;
    return max_index;
} 

// 3 workloads: 4+4+2
// 4 workloads: 4+2+2+2
// 5 workloads: 2+2+2+2+2
#if SET == 3
int get_block_workload(int cid, uint64_t current_cycle, bool drop_enable){
#else
int get_block_workload(int cid, uint64_t current_cycle){
#endif
    int empty_queue_num = 0;
    for(int i = 0; i < QUEUE_DEPTH; i++)
        if(ex_queue[i] == -1)
            empty_queue_num ++;

    if(empty_queue_num == QUEUE_DEPTH && last_queue_id == total_workloads){
       global_end = true;
#if PRINT == 1
       printf("global end by cid %d\n", cid);
#endif
       return -1;
    }

    // use this to track whether there is waiting workload (give up array)
    //requested_gemmini_vel = QUEUE_DEPTH - empty_queue_num;
    // need to finish current ones first
    int current_active_core = 0;
    for(int i = 0; i < NUM_CORE; i++)
        if(workload_running[i] != -1)
            current_active_core ++;
   
    int max_new_queue = NUM_CORE - current_active_core;

    if(current_active_core < 3){
        if(empty_queue_num < max_new_queue) max_new_queue = empty_queue_num;

        int end = (last_queue_id + max_new_queue) > total_workloads ? total_workloads : last_queue_id + max_new_queue;
        int index = last_queue_id;
        int ex_queue_index = 0;
        while(index < end){
          if(ex_queue[ex_queue_index] >= 0){
            ex_queue_index ++;
          }
          else if(total_queue_dispatch[index] <= current_cycle){
            ex_queue[ex_queue_index] = index;
#if PRINT == 1
            printf("new queue %d at queue by cid %d, num active core: %d\n", index, cid, current_active_core);
#endif
            //if(index == total_workloads - 1) last_queue_id = total_workloads;
            ex_queue_index ++;
            index ++;
            last_queue_id = index;
          }
          else{
            last_queue_id = index;
            break;
          }
        }
    }

#if SET == 3
    int max_index = get_max_index(cid, current_cycle, drop_enable);
#else
    int max_index = get_max_index(cid, current_cycle);
#endif


    int num_array = core_num_gemmini[cid];

    int num_idle_array = 0;
    for(int g = 0; g < NUM_ARRAY; g++){
        if(gemmini_status[g] == -1){
            num_idle_array ++;
        }
    }
    // wait until sufficient number of arrays are ready
    if(num_idle_array < num_array)
        return -1;


    //num_array = 4;
    // if ex_queue is empty
    if(max_index == -1){ 
      return -1; // schedule next turn
    }
    else{
      int queue_id = ex_queue[max_index];
#if PRINT == 1
     // printf("queue %d cid %d requires %d num gemmini\n", queue_id, cid, num_array);
#endif
      int idle_array = 0;
      // count and assign idle accel

#if NOC == 1
      for (int gg = 0; gg < NUM_ARRAY; gg++){
	  int g = (gg+3)%NUM_ARRAY;
#else
      for (int g = 0; g < NUM_ARRAY; g++){
#endif
          if (gemmini_status[g] == -1){
              core_gemmini[cid][idle_array] = g;
              idle_array ++;
#if PRINT == 1
              printf("queue %d cid %d got gemmini %d \n", queue_id, cid, g);
#endif
              gemmini_status[g] = cid;
          }
          
          if(idle_array == BASE_NUM_ARRAY && num_array <= BASE_NUM_ARRAY){
              //requested_gemmini[cid] = 0;
              break;
          }
          else if(idle_array == num_array && num_array > BASE_NUM_ARRAY){
              //requested_gemmini[cid] = 0;
              break;
          } 
      }

      /*
      //if less than required accel, request more accel
      if(idle_array < num_array && idle_array > 0){
          requested_gemmini[cid] = num_array - idle_array;
         //num_array = idle_array;
      }
      */
      /*
      else if(idle_array >= num_array){
         requested_gemmini[cid] = 0;
         if (num_array < BASE_NUM_ARRAY_VEL){
             num_array = (idle_array > BASE_NUM_ARRAY_VEL) ? BASE_NUM_ARRAY_VEL : idle_array;
         }
      }
      */

      core_num_gemmini[cid] = idle_array;
      // only empty queue when there is available array
      if(idle_array > 0){
          ex_queue[max_index] = -1;
          //requested_gemmini_vel--; 
          int num_array_index = 0;
          if(idle_array == 2) num_array_index = 1;
          else if(idle_array == 4) num_array_index = 2;
          int workload_type = total_queue_type[queue_id];
          for(int i = 0; i < 3; i ++){
             core_togo[i][cid] = sp_cycles[i][workload_type];
          }
          
          uint64_t old = current_cycle - total_queue_dispatch[queue_id];
          uint64_t slack = target_cycles[workload_type] - old;
          /*
	  current_score[cid] = (int)((100*slack) / core_togo[num_array_index][cid]);
          */
          core_num_gemmini_share[cid] = idle_array;
#if PRINT == 1
          //printf("queue %d cid %d needed %d/got %d num array before start (current time: %d, slack: %d)\n", queue_id, cid, num_array, idle_array, current_cycle, slack);
#endif 
      }
      else {
          /*
          int workload_type = total_queue_type[queue_id];
          uint64_t old = current_cycle - total_queue_dispatch[queue_id];
          uint64_t slack = target_cycles[workload_type] - old;
          current_score[cid] = (int)((1.1*100*slack) / sp_cycles[0][workload_type]);
          //requested_gemmini[NUM_CORE] = num_array;
          */
          // maybe not needed for not running workload
          return -1;
      }
      //ex_queue[max_index] = -1; // ToDo: if there's no array left     

#if PRINT == 1
      //printf("queue %d by cid %d got %d and requested %d num array \n", queue_id, cid, idle_array, requested_gemmini[cid]);
#endif
      workload_running[cid] = queue_id;
      return queue_id;
    }

}

// return new number of array when 
// current_mem_score[cid] = 0;done
// -1 if need to wait for new allocation
int block_repartition(int queue_id, size_t cid, int turn){
  int num_array = core_num_gemmini[cid];
  int workload_type = total_queue_type[queue_id];
  int num_idle_array = 0;
  for(int array = 0; array < NUM_ARRAY; array++){
    if(gemmini_status[array] == -1){
      num_idle_array ++; 
    }
  }

  uint64_t current_cycle = read_cycles_re() - global_start_time[cid];
  uint64_t old = current_cycle - total_queue_dispatch[queue_id];
  uint64_t slack = target_cycles[workload_type] - old;
  if(num_array == 2){
    // first turn: request new arrays if necessary, and wait until it gets it
    if(turn == 0){
       int num_array_index = get_num_array_index(num_array);
       if (slack < core_togo[num_array_index][cid]*1.1 || target_cycles[workload_type] < old){
         int need_array = 2;
         if(num_idle_array >= 2){
           for(int a = 0; a < NUM_ARRAY; a++){
              if(gemmini_status[a] == -1){
#if PRINT == 1
                 printf("cid %d type %d need acquire more, found idle array %d\n", cid, workload_type, a);
#endif
                 while(!rerocc_acquire(num_array, 1 << a)){}
                 gemmini_status[a] = cid;
                 core_gemmini[cid][num_array] = a;
                 num_array++;
                 core_num_gemmini[cid] ++;
                 core_num_gemmini_share[cid] ++;
                 need_array --;
                 total_queue_acquire[queue_id]++;
                 if(num_array == 4) 
                     break;
              } 
           }
         }
         
         if(need_array != 0){
             gemmini_reserved_vel[cid] = 2;
#if PRINT == 1
             printf("cid %d type %d need to acquire more\n", cid, workload_type);
#endif
             return -1;
         }
         else 
             return num_array;
       }
       else
           return num_array;
    }
    else{
      // next get newly assigned array, if there is
      for(int g = 0; g < NUM_ARRAY; g++)
        if(gemmini_status[g] == 100+cid){
           core_gemmini[cid][num_array] = g;
           total_queue_acquire[queue_id]++;
           while(!rerocc_acquire(num_array, 1 << g)){}
           gemmini_status[g] = cid;
           num_array++; 
           core_num_gemmini[cid] ++;
           core_num_gemmini_share[cid] ++;
#if PRINT == 1
           printf("cid %d acquired accel %d at turn %d\n", cid, g, turn);
#endif
           //gemmini_reserved_vel[cid]--;
        }
      if(gemmini_reserved_vel[cid] == 0)
          return num_array;
      else
          return -1;
    }
  }
  else if(num_array == 4){
    for(int core = 0; core < NUM_CORE; core++){
        if(gemmini_reserved_vel[core] > 0 && core != cid){
            num_array --;
            int array = core_gemmini[cid][num_array];
            core_gemmini[cid][num_array] = -1;
            rerocc_release(num_array);
            gemmini_status[array] = 100 + core;
            num_array --;
            array = core_gemmini[cid][num_array];
            core_gemmini[cid][num_array] = -1;
            rerocc_release(num_array);
            gemmini_status[array] = 100 + core;
            gemmini_reserved_vel[core] = 0;
            core_num_gemmini_share[cid] -= 2;
            core_num_gemmini[cid] = num_array;
            total_queue_release[cid] += 2;
#if PRINT == 1
            printf("cid %d released accel to other cid %d\n", cid, core);
#endif
            break;
        }
    }

    return num_array;
  }   
}

// with blocks
uint64_t workload_block_function(int queue_id, size_t cid){  
  int num_array = core_num_gemmini[cid];//0;
  /*
  for (int i = 0; i < NUM_ARRAY; i ++){
      if(core_gemmini[cid][i] >= 0) num_array ++;
  }
  */
  // ToDo: what if num_arry = 0?
  for(int i = 0; i < num_array; i++)
     while(!rerocc_acquire(i, 1 << (core_gemmini[cid][i]))){}

#if PRINT == 1
  printf("cid %d queue %d acquired rerocc %d \n", cid, queue_id, num_array);
#endif

  int workload_type = total_queue_type[queue_id];
  int workload_class = total_queue_class[queue_id];
  int num_block = workload_blocks[workload_type];

  for (int i = 0; i < num_array; i++) {
    rerocc_assign(OP3, i);
    gemmini_flush(0);
  }
  uint64_t* cycles;
  uint64_t total_runtime;

  /*
  int status = total_queue_status[queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;
  */
  uint64_t start = read_cycles();
  for(int block = 0; block < num_block; block++){
#if SET != 3
#if SET != 1 
      if(workload_type == ALEXNET){
#if FULL == 1
        if(workload_class) cycles = alexnet_function_1(block, false, false, false, false, num_array, cid);
        else cycles = alexnet_function_11(block, false, false, false, false, num_array, cid);
#else 
        cycles = alexnet_function_1(block, false, false, false, false, num_array, cid);
#endif
        //total_runtime = *(cycles+14);
      }
      if(workload_type == RESNET){
#if FULL == 1
        if(workload_class) cycles = resnet_function_1(block, false, false, false, false, num_array, cid);
        else cycles = resnet_function_11(block, false, false, false, false, num_array, cid);
        //total_runtime = *(cycles+72);
#else
        cycles = resnet_function_1(block, false, false, false, false, num_array, cid);
#endif
      }
#endif
#if SET == 2
      if(workload_type == YOLONET){
        if(workload_class) cycles = yolonet_function_1(block, false, false, false, false, num_array, cid);
        else cycles = yolonet_function_11(block, false, false, false, false, num_array, cid);
        //total_runtime = *(cycles+26);
      }
      if(workload_type == GOOGLENET){
        if(workload_class) cycles = googlenet_function_1(block, false, false, false, false, num_array, cid);
        else cycles = googlenet_function_11(block, false, false, false, false, num_array, cid); 
        //total_runtime = *(cycles+71);
      }
#endif
#if SET != 1
      if(workload_type == BERTBASE){
#if FULL == 1
        if(workload_class)  cycles = bertbase_function_1(block, false, false, false, false, num_array, cid);
        else cycles = bertbase_function_11(block, false, false, false, false, num_array, cid); 
        //total_runtime = *(cycles+71);
#else
        cycles = bertbase_function_1(block, false, false, false, false, num_array, cid);
#endif
      }
#endif
#if SET == 1
      if(workload_type == KWSNET){
        if(workload_class) cycles = kwsnet_function_1(block, false, false, false, false, num_array, cid);
        else cycles = kwsnet_function_11(block, false, false, false, false, num_array, cid);
        //total_runtime = *(cycles+40);
      }
#endif
#if SET != 2
      if(workload_type == RES18NET){
        if(workload_class) cycles = res18net_function_1(block, false, false, false, false, num_array, cid);
        else cycles = res18net_function_11(block, false, false, false, false, num_array, cid);
        //total_runtime = *(cycles+31);
      }
      if(workload_type == SQUEEZENET){
        if(workload_class) cycles = squeezenet_function_1(false, false, false, false, num_array, cid);
        else cycles = squeezenet_function_11(false, false, false, false, num_array, cid);
        //total_runtime = *(cycles+29);
      }
      if(workload_type == YOLOLITENET){
        if(workload_class) cycles = yololitenet_function_1(false, false, false, false, num_array, cid);
        else cycles = yololitenet_function_11(false, false, false, false, num_array, cid);
        //total_runtime = *(cycles+14);
      }
#endif
#if SET == 1
      if(workload_type == BERTSMALL){
        if(workload_class) cycles = bertmedium_function_1(block, false, false, false, false, num_array, cid);
        else cycles = bertmedium_function_11(block, false, false, false, false, num_array, cid); 
        //total_runtime = *(cycles+71);
      }
#endif

#else
     if(workload_type == RCNNET){
       cycles = rcnnnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == HANDNET){
       cycles = handnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == RITNET){
       cycles = ritnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == MIDASNET){
       cycles = midasnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == FBNET){
       cycles = fbnet_function_1(block, false, num_array, cid);
     }
#endif
      dram_util[cid] = 0;
      current_mem_score[cid] = 0;
      if(block < num_block - 1){
          uint64_t start_lock = read_cycles();
          int new_num_array = -1;
          int turn = 0;
          while(new_num_array == -1){
            pthread_mutex_lock(&ex_queue_mutex);
            new_num_array = block_repartition(queue_id, cid, turn);
            pthread_mutex_unlock(&ex_queue_mutex);
            if(new_num_array == -1){
              int i = 0;
	      int wait_cycle = 100000;//(mode == 4) ? 100000 : 20000;
              while(i < wait_cycle){
                i ++;
              }
              turn ++;
#if PRINT == 1
              if(turn % 100 == 1) 
		      printf("cid %d while loop turn %d\n", cid, turn);
#endif
            }
            else {
              num_array = new_num_array;
            }
          }
          uint64_t end_lock = read_cycles();
          total_queue_overhead[queue_id] += end_lock - start_lock;
      }
  }
  current_mem_score[cid] = 0;
 
  //core_num_gemmini[cid] = 0;
  uint64_t start_lock = read_cycles();
  pthread_mutex_lock(&ex_queue_mutex);
  //requested_gemmini[cid] = 0;
  current_score[cid] = -1;
  core_num_gemmini_share[cid] = 0;
  workload_running[cid] = -1;

  num_array = core_num_gemmini[cid];
  for(int core = 0; core < NUM_CORE; core++){
      if(num_array > 0 && gemmini_reserved_vel[core] > 0 && core != cid){ 
         num_array --;
         int array = core_gemmini[cid][num_array];
         core_gemmini[cid][num_array] = -1;
         rerocc_release(num_array);
         gemmini_status[array] = 100 + core;
         num_array --;
         array = core_gemmini[cid][num_array];
         core_gemmini[cid][num_array] = -1;
         rerocc_release(num_array);
         gemmini_status[array] = 100 + core;
         gemmini_reserved_vel[core] = 0;
         core_num_gemmini[cid] = num_array;
#if PRINT == 1
            printf("cid %d released accel to other cid %d at the end\n", cid, core);
#endif
      }
  }
  if(num_array > 0){
#if PRINT ==1
    printf("cid %d still need to release num %d accels\n", cid, num_array);
#endif
    core_num_gemmini[cid] = 0;
    num_array = 0;
    for(int i = 0; i < NUM_ARRAY; i++){
      if(core_gemmini[cid][i] >= 0){
        int accel = core_gemmini[cid][i];
        gemmini_status[accel] = -1;
        rerocc_release(i);
        core_gemmini[cid][i] = -1;
      }
      else
          break;
    }
  }
  pthread_mutex_unlock(&ex_queue_mutex); 
  uint64_t end = read_cycles();
  total_queue_overhead[queue_id] += end - start_lock;
  total_queue_finish[queue_id] = end - global_start_time[cid];
  total_queue_status[queue_id] = layer_pointer[cid];
  layer_pointer[cid] = 0;
  //if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  total_runtime = end - start;
#if PRINT == 1
  printf("cid %d queue %d released rerocc \n", cid, queue_id);
#endif
  return total_runtime;
}

void prerun_block_profile(int cid, int workload_type, int num_array){
/* 
  for (int i = 0; i < num_array; i++) {
    rerocc_assign(OP3, i);
    gemmini_flush(0);
  }
*/
  core_num_gemmini[cid] = num_array;
  uint64_t* cycles;
  uint64_t total_runtime;
  layer_pointer[cid] = 0;

  if(SET == 1){
      if(!(workload_type == 4 || workload_type == 5 || workload_type == 6 || workload_type == 7 || workload_type == 9))
          return;
  }

  if(SET == 2){
      if(!(workload_type == 0 || workload_type == 1 || workload_type == 2 || workload_type == 3 || workload_type == 8))
          return;
  }

  if(SET == 0){
      if(!(workload_type == RESNET || workload_type == ALEXNET || workload_type == BERTBASE || workload_type == RES18NET || workload_type == SQUEEZENET || workload_type == YOLOLITENET))
          return;
  }

  mode = 1;
  int num_array_index = 0;
  if(num_array == 2) num_array_index = 1;
  else if(num_array == 4) num_array_index = 2;

  workload_running[cid] = workload_type;
  //bool weight_direct_dram = false;
  int num_block = workload_blocks[workload_type];

  for (int i = 0; i < 2; i++){ 
     uint64_t start = read_cycles();
     for(int block = 0; block < num_block; block++){
#if SET != 3
#if SET != 1
         if(workload_type == ALEXNET){
           cycles = alexnet_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+14);
         }
         if(workload_type == RESNET){
           cycles = resnet_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+72);
         }
#endif
#if SET == 2
         if(workload_type == YOLONET){
           cycles = yolonet_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+26);
         }
         if(workload_type == GOOGLENET){
           cycles = googlenet_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+71);
         }
#endif
#if SET != 1
         if(workload_type == BERTBASE){
           cycles = bertbase_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+14);
         }
#endif
#if SET == 1 
         if(workload_type == KWSNET){
           cycles = kwsnet_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+40);
         }
#endif
#if SET != 2
         if(workload_type == RES18NET){
           cycles = res18net_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+31);
         }
         if(workload_type == SQUEEZENET){
           cycles = squeezenet_function_1(false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+29);
         }
         if(workload_type == YOLOLITENET){
           cycles = yololitenet_function_1(false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+14);
         }
#endif
#if SET == 1
         if(workload_type == BERTSMALL){
           cycles = bertmedium_function_1(block, false, false, false, false, num_array, cid);
           //total_runtime = *(cycles+14);
         }
#endif

#else
     if(workload_type == RCNNET){
       cycles = rcnnnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == HANDNET){
       cycles = handnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == RITNET){
       cycles = ritnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == MIDASNET){
       cycles = midasnet_function_1(block, false, num_array, cid);
     }
     if(workload_type == FBNET){
       cycles = fbnet_function_1(block, false, num_array, cid);
     }
#endif
     }
     uint64_t end = read_cycles();
     //if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
     //uint64_t runtime = read_cycles() - start;
     total_runtime = end - start;
     if (mode == 1) sp_cycles[num_array_index][workload_type] = total_runtime;
     
     if (mode == 1){
        printf("mode 1 sp runtime profile for workload %d number of array %d \n", workload_type, num_array);
        printf("number of layers: %d\n", layer_pointer[cid]);
        if (num_array == 4) printf("workload %d total runtime: %d\n", workload_type, total_runtime);
	else printf("total runtime: %d\n", total_runtime);
        workload_num_layers[workload_type] = layer_pointer[cid];
        printf("cycles: ");
        int acc_cycle = 0;
        for (int i = 0; i < layer_pointer[cid]; i++){
            acc_cycle += sp_layer_cycles[num_array_index][workload_type][i];
            printf("%d (%d, %d), ", sp_layer_cycles[num_array_index][workload_type][i], i, acc_cycle);
        }
        printf("\n");
     }
     else if (mode == 2){
        printf("mode 2 sp alpha factor profile for workload %d number of array\n", workload_type, num_array);
        printf("number of layers: %d\n", layer_pointer[cid]);
        printf("total runtime: %d\n", total_runtime);
        printf("alpha(\%): ");
        for (int i = 0; i < layer_pointer[cid]; i++){
            printf("%d, ", sp_layer_alpha[num_array_index][workload_type][i]);
        }
        printf("\n");
	/*
        printf("from_dram: ");
        for (int i = 0; i < layer_pointer[cid]; i++){
            printf("%d, ", sp_layer_from_dram[num_array_index][workload_type][i]);
        }
        printf("\n");
        printf("compute_ideal: ");
        for (int i = 0; i < layer_pointer[cid]; i++){
            printf("%d, ", sp_layer_compute_ideal[num_array_index][workload_type][i]);
        }
        printf("\n");
        printf("mem_ideal: ");
        for (int i = 0; i < layer_pointer[cid]; i++){
            printf("%d, ", sp_layer_mem_ideal[num_array_index][workload_type][i]);
        }
        printf("\n");
	*/
     }
     mode ++;
     layer_pointer[cid] = 0;
  }
  // ToDo: define mode
  mode = 3; // for dynamic compute reconfig
}


