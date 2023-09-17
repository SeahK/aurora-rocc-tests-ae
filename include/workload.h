#if SET == 3
#include "include/workload_create_xr.h"
#else
#include "include/workload_create.h"
#endif

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
      int slack = target_cycles[workload_type] - old;
      bool not_meet_ddl = ((slack <= sp_cycles[2][workload_type]) || (slack <= 0));
      if (not_meet_ddl) 
	  num_array = 1;
      else{
        for(int i = 2; i >= 0; i --){
#if NOC == 1
          if (slack < (2*sp_cycles[i][workload_type])){
#else
          if (slack < (1.5*sp_cycles[i][workload_type])){
#endif
            num_array = (1 << i);
            break;
          }
        }
      }
      core_num_gemmini[cid] = num_array;
      if(NOC_OPTIM){
	    if(num_array < 2)
	       core_num_gemmini[cid] = 2;
      }
      else{
        if (workload_type == YOLOLITENET || workload_type == SQUEEZENET) {
          if (num_array > 2)
              core_num_gemmini[cid] = 2;
        }
      }
    }
    else
      core_num_gemmini[cid] = 0;
    return max_index;
} 

#if SET == 3
int get_workload(int cid, uint64_t current_cycle, bool drop_enable){
#else
int get_workload(int cid, uint64_t current_cycle){
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

    int end = (last_queue_id + empty_queue_num) > total_workloads ? total_workloads : last_queue_id + empty_queue_num;
    int index = last_queue_id;
    int ex_queue_index = 0;
    while(index < end){
      if(ex_queue[ex_queue_index] >= 0){
        ex_queue_index ++;
      }
      else if(total_queue_dispatch[index] <= current_cycle){
        ex_queue[ex_queue_index] = index;
#if PRINT == 1
        printf("new queue %d at queue by cid %d\n", index, cid);
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

#if SET == 3
    int max_index = get_max_index(cid, current_cycle, drop_enable);
#else
    int max_index = get_max_index(cid, current_cycle);
#endif

    int num_array = core_num_gemmini[cid];
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
      int workload_type = total_queue_type[queue_id];
      // count and assign idle accel
      if(NOC_OPTIM){
         // search for the fav ones first
         for(int g_index = 0; g_index < NUM_ARRAY_GROUP; g_index ++){
            int g = gemmini_noc_optim[workload_type][g_index];
            if(gemmini_status[g] == -1){
               core_gemmini[cid][idle_array] = g;
               int g2 = hashed_gemmini_pair(g);
               core_gemmini[cid][idle_array+1] = g2;
               gemmini_status[g] = cid;
               gemmini_status[g2] = cid;
#if PRINT == 1
               printf("queue %d cid %d got gemmini %d, %d \n", queue_id, cid, g, g2);
#endif
               if(g_index > 2 && noc_optim_need_swap[cid] == -1) 
                  noc_optim_need_swap[cid] = idle_array;
               else if(g_index > 2 && noc_optim_need_swap[cid] == 0 && idle_array == 2){
                  noc_optim_need_swap[cid] = -1;
#if PRINT == 1
               printf("cid %d type %d all 4 can't be swapped\n", cid, workload_type);
#endif
               }
               else noc_optim_need_swap[cid] = -1;
               idle_array += 2;
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
      }
      else{
#if NOC == 1
          for (int gg = 0; gg < NUM_ARRAY; gg++){
          int g = (gg+6)%NUM_ARRAY;
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
         if (num_array < BASE_NUM_ARRAY){
             num_array = (idle_array > BASE_NUM_ARRAY) ? BASE_NUM_ARRAY : idle_array;
         }
      }
      */

      core_num_gemmini[cid] = idle_array;
      // clear out hanging score
      for(int c = 0; c < NUM_CORE; c++){
          if(workload_running[c] < 0 && c != cid){
              current_score[c] = -1;
          }
      }
      // only empty queue when there is available array
      if(idle_array > 0){
          ex_queue[max_index] = -1;
          int num_array_index = 0;
          if(idle_array == 2) num_array_index = 1;
          else if(idle_array == 4) num_array_index = 2;
          for(int i = 0; i < 3; i ++){
             core_togo[i][cid] = sp_cycles[i][workload_type];
          }
          uint64_t old = current_cycle - total_queue_dispatch[queue_id];
          uint64_t slack = target_cycles[workload_type] - old;
          if(slack <= 0 || slack <= core_togo[2][workload_type])
	      current_score[cid] = 0;
	  else 
	      current_score[cid] = (int)((100*slack) / core_togo[num_array_index][cid]);
          core_num_gemmini_share[cid] = idle_array;
#if PRINT == 1
          printf("queue %d cid %d needed %d/got %d num array before start (score: %d, current time: %d, slack: %d)\n", queue_id, cid, num_array, idle_array,  current_score[cid], current_cycle, slack);
#endif 
      }
      else {
          //int workload_type = total_queue_type[queue_id];
          uint64_t old = current_cycle - total_queue_dispatch[queue_id];
          uint64_t slack = target_cycles[workload_type] - old;
          if(slack <= 0 || slack < sp_cycles[2][workload_type])
	      current_score[cid] = 0;
	  else 
	      current_score[cid] = (int)((1.1*100*slack) / sp_cycles[0][workload_type]);
          //current_score[cid] = (int)((1.1*100*slack) / sp_cycles[0][workload_type]);
          //requested_gemmini[NUM_CORE] = num_array;
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

/*
uint64_t workload_function(int queue_id, int workload_id, size_t cid, int group_id, int num_array, pthread_barrier_t *barrier_funct){
 
  gemmini_flush(0);
*/
uint64_t workload_function(int queue_id, size_t cid){  
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
  for (int i = 0; i < num_array; i++) {
    rerocc_assign(OP3, i);
    gemmini_flush(0);
  }
  uint64_t* cycles;
  uint64_t total_runtime;
  curr_block[cid] = -1;

  /*
  int status = total_queue_status[queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;
  */
  uint64_t start = read_cycles();
#if SET != 3
#if SET != 1 
  if(workload_type == ALEXNET){
#if FULL == 1
    if(workload_class) cycles = alexnet_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = alexnet_function_11(-1, false, false, false, false, num_array, cid);
#else 
    cycles = alexnet_function_1(-1, false, false, false, false, num_array, cid);
#endif
    //total_runtime = *(cycles+14);
  }
  if(workload_type == RESNET){
#if FULL == 1
    if(workload_class) cycles = resnet_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = resnet_function_11(-1, false, false, false, false, num_array, cid);
    //total_runtime = *(cycles+72);
#else
    cycles = resnet_function_1(-1, false, false, false, false, num_array, cid);
#endif
  }
#endif
#if SET == 2
  if(workload_type == YOLONET){
    if(workload_class) cycles = yolonet_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = yolonet_function_11(-1, false, false, false, false, num_array, cid);
    //total_runtime = *(cycles+26);
  }
  if(workload_type == GOOGLENET){
    if(workload_class) cycles = googlenet_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = googlenet_function_11(-1, false, false, false, false, num_array, cid); 
    //total_runtime = *(cycles+71);
  }
#endif
#if SET != 1
  if(workload_type == BERTBASE){
#if FULL == 1
    if(workload_class)  cycles = bertbase_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = bertbase_function_11(-1, false, false, false, false, num_array, cid); 
    //total_runtime = *(cycles+71);
#else
    cycles = bertbase_function_1(-1, false, false, false, false, num_array, cid);
#endif
  }
#endif
#if SET == 1
  if(workload_type == KWSNET){
    if(workload_class) cycles = kwsnet_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = kwsnet_function_11(-1, false, false, false, false, num_array, cid);
    //total_runtime = *(cycles+40);
  }
#endif
#if SET != 2
  if(workload_type == RES18NET){
    if(workload_class) cycles = res18net_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = res18net_function_11(-1, false, false, false, false, num_array, cid);
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
    if(workload_class) cycles = bertmedium_function_1(-1, false, false, false, false, num_array, cid);
    else cycles = bertmedium_function_11(-1, false, false, false, false, num_array, cid); 
    //total_runtime = *(cycles+71);
  }
#endif
#else

     if(workload_type == RCNNET){
       cycles = rcnnnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == HANDNET){
       cycles = handnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == RITNET){
       cycles = ritnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == MIDASNET){
       cycles = midasnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == FBNET){
       cycles = fbnet_function_1(-1, false, num_array, cid);
     }

#endif
  uint64_t end = read_cycles();
  total_queue_finish[queue_id] = end - global_start_time[cid];
  total_queue_status[queue_id] = layer_pointer[cid];
  //if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  total_runtime = end - start;
  layer_pointer[cid] = 0;
  current_mem_score[cid] = 0;
 
  for(int i = 0; i < NUM_ARRAY; i++){
    if(core_gemmini[cid][i] >= 0){
      rerocc_release(i);
      core_gemmini[cid][i] = -1;
    }
    else 
      break;
  }
  core_num_gemmini[cid] = 0;
  layer_pointer[cid] = 0;
  pthread_mutex_lock(&ex_queue_mutex);
  //requested_gemmini[cid] = 0;
  current_score[cid] = -1;
//  likely_miss_ddl_vel[cid] = false;
  core_num_gemmini_share[cid] = 0;
  workload_running[cid] = -1;
  for(int i = 0; i < NUM_ARRAY; i++)
      if(gemmini_status[i] == cid) {
          gemmini_status[i] = -1;
          /*
          for(int j = 0; j < NUM_CORE; j++){
              if(requested_gemmini[j] > 0){
                  gemmini_status[i] = j;
                  requested_gemmini[j] --;
#if PRINT == 1
                  printf("cid %d found other cid %d request accel %d when finished\n", cid, j, i);
#endif
                  break;
              }
          }
          */
      }
  pthread_mutex_unlock(&ex_queue_mutex); 
#if PRINT == 1
  printf("cid %d queue %d released rerocc \n", cid, queue_id);
#endif
  return total_runtime;
}

void prerun_profile(int cid, int workload_type, int num_array){
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
  curr_block[cid] = -1;

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

  if(SET == 3){
      if(!(workload_type >= 10 && workload_type <= 14))
          return;
  }
  mode = 1;
  int num_array_index = 0;
  if(num_array == 2) num_array_index = 1;
  else if(num_array == 4) num_array_index = 2;

  workload_running[cid] = workload_type;

  for (int i = 0; i < 2; i++){ 
     uint64_t start = read_cycles();
#if SET != 3
#if SET != 1
     if(workload_type == ALEXNET){
       cycles = alexnet_function_1(-1, false, false, false, false, num_array, cid);
       //total_runtime = *(cycles+14);
     }
     if(workload_type == RESNET){
       cycles = resnet_function_1(-1, false, false, false, false, num_array, cid);
       //total_runtime = *(cycles+72);
     }
#endif
#if SET == 2
     if(workload_type == YOLONET){
       cycles = yolonet_function_1(-1, false, false, false, false, num_array, cid);
       //total_runtime = *(cycles+26);
     }
     if(workload_type == GOOGLENET){
       cycles = googlenet_function_1(-1, false, false, false, false, num_array, cid);
       //total_runtime = *(cycles+71);
     }
#endif
#if SET != 1
     if(workload_type == BERTBASE){
       cycles = bertbase_function_1(-1, false, false, false, false, num_array, cid);
       //total_runtime = *(cycles+14);
     }
#endif
#if SET == 1 
     if(workload_type == KWSNET){
       cycles = kwsnet_function_1(-1, false, false, false, false, num_array, cid);
       //total_runtime = *(cycles+40);
     }
#endif
#if SET != 2
     if(workload_type == RES18NET){
       cycles = res18net_function_1(-1, false, false, false, false, num_array, cid);
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
       cycles = bertmedium_function_1(-1, false, false, false, false, num_array, cid);
       //total_runtime = *(cycles+14);
     }
#endif

#else 
     if(workload_type == RCNNET){
       cycles = rcnnnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == HANDNET){
       cycles = handnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == RITNET){
       cycles = ritnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == MIDASNET){
       cycles = midasnet_function_1(-1, false, num_array, cid);
     }
     if(workload_type == FBNET){
       cycles = fbnet_function_1(-1, false, num_array, cid);
     }
#endif
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


