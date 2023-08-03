// See LICENSE for license details.
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define NUM_CORE 4
// in common
#define OP 3
#define total_workloads 200
#define SEED 0
#define TARGET_SCALE 1
#define CAP_SCALE 0.83
#define TARGET_SCALE1 0.83
#define TARGET_SCALE2 1.2

//#define DEBUG 1
#define PRINT 0
#define SET 1
#define FULL 0
#define NOC 1

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "include/workload_base.h"

struct thread_args{
    int cid, num_array, start_tracker;
};

void *thread_test(void *arg){
    //pthread_barrier_wait(&barrier_global);
    struct thread_args * nn_args = (struct thread_args *) arg;
    int cid = nn_args->cid;
    //int num_array = 2; // ToDo: change this to be flexible (1, 2, 4)
    //if(cid == 0) global_start_time = read_cycles();//+10000;
    //printf("global start time set by cid %d: %llu\n", cid, global_start_time);
    pthread_barrier_wait(&barrier_global);
    global_start_time[cid] = read_cycles();

    while(!global_end){ 
      pthread_mutex_lock(&ex_queue_mutex);
      uint64_t current_time = read_cycles() - global_start_time[cid];
#if PRINT == 1
      printf("cid %d got lock - current time: %llu\n", cid, current_time);
#endif
      int queue_id = get_block_workload(cid, current_time);
#if PRINT == 1
      printf("cid %d queue id: %d, \n", cid, queue_id);
#endif
      // when no workload is currently available
      // ToDo: when no array is currently available
      int turn = 0;
      while(queue_id == -1){
        pthread_mutex_unlock(&ex_queue_mutex);
        if(global_end) break;
        int i = 0;
        while(i < 100000){
          i ++;
        }
        turn ++;
#if PRINT == 1
        if(turn % 10000 == 0) printf("cid %d while loop turn %d\n", cid, turn);
#endif
        pthread_mutex_lock(&ex_queue_mutex);
        //printf("cid %d got lock\n", cid);
        current_time = read_cycles() - global_start_time[cid];
        queue_id = get_block_workload(cid, current_time);
      }
      if (global_end) continue;

      pthread_mutex_unlock(&ex_queue_mutex);
#if PRINT == 1
      //printf("cid %d release lock\n", cid);
#endif
      /*
      for (int i = 0; i < num_array; i++) {
         rerocc_assign(OP, i);
         gemmini_flush(0);
      }
      */

      uint64_t this_runtime = workload_block_function(queue_id, cid); // ToDo: weight direct dram

      //for debugging
      total_queue_runtime[queue_id] = this_runtime;
      total_queue_core[queue_id] = cid;

#if PRINT == 1
      //printf("cid %d finish operation for queue id %d\n", cid, queue_id);
#endif
    }
#if PRINT == 1
    printf("cid %d finished\n", cid);
#endif 
}

void *thread_profile(void *arg){
	int cid = sched_getcpu() % NUM_CORE;
    printf("entered thread for cid %d\n", cid);
	struct thread_args * nn_args = (struct thread_args *) arg;
    int num_array = nn_args->num_array;
    int start_tracker = nn_args->start_tracker;	
    uint64_t* cycles;

    uint64_t thread_start = read_cycles();

    for (int w = 0; w < NUM_TYPE_WORKLOAD; w++){
      printf("=============== start new workload ================\n");
      for(int i = 0; i < num_array; i++)
        while(!rerocc_acquire(i, 1<<(i+start_tracker))){} // 0xff->i+start_tracker
      for (int i = 0; i < num_array; i++) {
        rerocc_assign(OP3, i);
        gemmini_flush(0);
      }
      prerun_block_profile(cid, w, num_array);  
      for(int i = 0; i < num_array; i++)
        rerocc_release(i); 
    }
}
void *print_message(void *ptr){
    int cpu_id = sched_getcpu();
   // char *msg;
   // msg = (char *) ptr;
    printf("print msg - cpu_id: %d \n", cpu_id);
   // printf("%s \n", msg);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    int cpu_id;
    cpu_id = sched_getcpu();
    printf("main thread cpuid: %d \n", cpu_id);

    cpu_set_t cpuset[NUM_CORE];
    pthread_t thread[NUM_CORE];
    pthread_attr_t attr[NUM_CORE];
    for(int i = 0; i < NUM_CORE; i++)
	pthread_attr_init(&attr[i]);
    struct thread_args nn_args[NUM_CORE];

    printf("create threading \n");
    for(int i = 0; i < NUM_CORE; i++){
	  CPU_ZERO(&cpuset[i]);
	  CPU_SET(i, &cpuset[i]);
	  pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
	  pthread_create(&thread[i], &attr[i], print_message, NULL);
    }

    for(int i = 0; i < NUM_CORE; i++){
      pthread_join(thread[i], NULL);
    }
    printf("thread joined after message printing\n");
    printf("profiling start\n");
    for(int r = 0; r < 3; r+=1){
	  int num_array = (1 << r); // 1, 2, 4
	  printf("number of array: %d\n", num_array); 
	  nn_args[0].num_array =  num_array;
      nn_args[0].start_tracker = 0;
	  pthread_create(&thread[0], &attr[0], thread_profile, &nn_args[0]);
	  pthread_join(thread[0], NULL);

    }
    printf("done profiling\n\n");
    printf("start workload creation\n");
    workload_create(total_workloads, SEED, TARGET_SCALE, CAP_SCALE);
    printf("workload type: ");
    for(int j = 0; j < total_workloads; j++){
      printf(" %d,", total_queue_type[j]);
    }
    printf("\n");
    
    printf("workload class: ");
    for(int j = 0; j < total_workloads; j++){
      printf(" %d,", total_queue_class[j]);
    }
    printf("\n");
    
    printf("workload dispatch time: ");
    for(int j = 0; j < total_workloads; j++){
      printf(" %llu,", total_queue_dispatch[j]);
    }
    printf("\n");

    printf("workload target time: ");
    for(int j = 0; j < total_workloads; j++){
      printf(" %llu,", total_queue_target[j]);
    }
    printf("\n");
    printf("done creation\n\n");

    mode = 5; // for moca

    last_queue_id = 0;
    if (pthread_mutex_init(&ex_queue_mutex, NULL) != 0){
      printf("\n mutex init failed\n");
      return 1;
    }  
    pthread_barrier_init(&barrier_global, NULL, NUM_CORE);

    for (int t = 0; t < 3; t++){
       if(t != 0){
	   float target_scale = (t == 1) ? TARGET_SCALE1 : TARGET_SCALE2;
	   workload_init(target_scale, total_workloads);
       }

       printf("start target scale round %d\n", t);
       printf("mode: %d\n", mode);
	for(int i = 0; i < NUM_CORE; i++){
	  nn_args[i].cid = i;
	  nn_args[i].num_array = 2;
	  pthread_create(&thread[i], &attr[i], thread_test, &nn_args[i]);
	}

	for(int i = 0; i < NUM_CORE; i++)
	  pthread_join(thread[i], NULL);

	for(int i = 0; i < NUM_ARRAY; i++){
	    printf("final gemmini %d status %d\n", i, gemmini_status[i]);
	}


	for(int i = 0; i < total_workloads; i++){
	  printf("queue id %d workload type: %d\n", i, total_queue_type[i]);
	  printf("queue id %d dispatch to finish time: %llu\n", i, total_queue_finish[i] - total_queue_dispatch[i]);
	  printf("queue id %d dispatched time: %llu\n", i, total_queue_dispatch[i]);
	  printf("queue id %d target: %llu\n", i, total_queue_target[i]);
	  printf("queue id %d runtime: %llu\n", i, total_queue_runtime[i]);
	  printf("queue id %d core: %llu\n", i, total_queue_core[i]);
	  printf("queue id %d release: %d\n", i, total_queue_release[i]);
	  printf("queue id %d acquire: %d\n", i, total_queue_acquire[i]);
	  printf("queue id %d overhead: %d\n", i, total_queue_overhead[i]);
	}
	printf("end of target %d test\n", t);
    }
    pthread_barrier_destroy(&barrier_global);
  printf("end of test\n");
  printf("==================================\n");
}
