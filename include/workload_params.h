#include <pthread.h>
#include <sched.h>

#define total_workloads 200
#define FULL 0
#define SEED 2
// QoS H, L
#define TARGET_SCALE1 0.83
#define TARGET_SCALE2 1.2

#if SET == 3
#if NOC == 0
#define TARGET_SCALE 1.0
#elif NOC == 1
#define TARGET_SCALE 1.2
#endif
#else
#define TARGET_SCALE 1
#endif

#if NOC == 0
#if SET == 0
#define CAP_SCALE 0.85
#elif SET == 1
#define CAP_SCALE 0.72//0.70
#elif SET == 2
#define CAP_SCALE 0.89//0.88
#endif
#else
#if SET == 0
#define CAP_SCALE 0.98//0.95 
#elif SET == 1
#define CAP_SCALE 0.75
#elif SET == 2
#define CAP_SCALE 1.00//0.95
#endif
#endif

// set 0: mixed
// set 1: light
// set 2: heavy
#define ALEXNET 0
#define RESNET 1
#define YOLONET 2
#define GOOGLENET 3

#define KWSNET 4
#define RES18NET 5
#define SQUEEZENET 6
#define YOLOLITENET 7

#define BERTBASE 8
#define BERTSMALL 9

// for XR
#define RCNNET 10
#define HANDNET 11
#define RITNET 12
#define MIDASNET 13
#define FBNET 14

#define MAX_WORKLOAD 400
#define NUM_TYPE_WORKLOAD (10+5)
#define MAX_LAYERS 400

#ifndef NUM_CORE
#define NUM_CORE 8
#endif
#define NUM_ARRAY 10

#ifndef QUEUE_DEPTH
#define QUEUE_DEPTH (NUM_CORE)
#endif

#ifndef total_workloads
#define total_workloads 200
#endif

#ifndef BASE_NUM_ARRAY
#define BASE_NUM_ARRAY 2
#endif

#ifndef SET
#define SET 0
#endif

#ifndef FULL
#define FULL 0
#endif

#ifndef NOC_OPTIM 
#define NOC_OPTIM 0
#endif

#ifndef NOC
#define NOC 0
#endif
//#ifndef QOS
//#define QOS 0
//#endif

#define CACHE_SIZE 2e6
#define CACHE_BANKS 8
//#if NOC == 1
//#define NUM_DRAM_CHANNEL 3 // to model lower bandwidth
//#else
#define NUM_DRAM_CHANNEL 4
//#endif
#define NUM_DRAM_BYTE 0.5
#ifndef DRAM_BW_SCALE
#define DRAM_BW_SCALE 1.2
#endif

#define DRAM_BW (NUM_DRAM_CHANNEL * NUM_DRAM_BYTE * DRAM_BW_SCALE) // 2 channels (16 bytes / cycle)

// modes
#define MOCA 5
#define REROCC 3
#define VELTAIR 4
#define REROCCMEM 6

#if NOC == 1
#if SET == 1
#define INTERVAL 50
#elif SET == 2
#define INTERVAL 30
#else 
#define INTERVAL 40
#endif

#else
#if SET == 1
#define INTERVAL 50
#elif SET == 2
#define INTERVAL 35
#else 
#define INTERVAL 40
#endif

#endif

#ifndef BAREMETAL
pthread_barrier_t barrier_global;
pthread_mutex_t ex_queue_mutex;
#endif

// core that need more resource set req bit to other core with higher score
//pthread_mutex_t realloc_req_mutex;

// single program (isolated run) cycles
#if NOC == 1
static uint64_t target_cycles[NUM_TYPE_WORKLOAD] = {1e9/30, 1e9/40, 1e9/80, 1e9/80, 1e9/80, 1e9/80, 1e9/100, 1e9/100, 1e9/25, 1e9/80, 1e9/45, 1e9/45, 1e9/50, 1e9/40, 1e9/35};
//static int rand_cycles[NUM_TYPE_WORKLOAD] = {11631637, 12890079, 7327589, 5607691, 8183652, 5227839, 2049409, 2184277, 24321810, 4568806};
#else
static uint64_t target_cycles[NUM_TYPE_WORKLOAD] = {1e9/40, 1e9/40, 1e9/80, 1e9/80, 1e9/80, 1e9/80, 1e9/100, 1e9/100, 1e9/25, 1e9/80, 1e9/35, 1e9/45, 1e9/50, 1e9/40, 1e9/30};
//static int rand_cycles[NUM_TYPE_WORKLOAD] = {9205806, 11384575, 6658393, 4671794, 7739816, 4868501, 1954311, 2059897, 23742640, 4445361};
//static int rand_cycles[NUM_TYPE_WORKLOAD] = {10048360, 11518812, 6730644, 4831530, 7484774, 4926520, 1894553, 2097425, 23820081, 4461869};
#endif
static int rand_cycles[NUM_TYPE_WORKLOAD] = {10048360, 11518812, 6730644, 4831530, 7484774, 4926520, 1894553, 2097425, 23820081, 4461869};
// 1, 2, 4 arrays
static int workload_num_layers[NUM_TYPE_WORKLOAD] = {0};
static uint64_t sp_layer_cycles[3][NUM_TYPE_WORKLOAD][MAX_LAYERS] = {0};
static uint64_t sp_cycles[3][NUM_TYPE_WORKLOAD] = {0};
static int sp_layer_alpha[3][NUM_TYPE_WORKLOAD][MAX_LAYERS] = {-1};
static int sp_layer_from_dram[3][NUM_TYPE_WORKLOAD][MAX_LAYERS] = {0};
static int sp_layer_compute_ideal[3][NUM_TYPE_WORKLOAD][MAX_LAYERS] = {0};
static int sp_layer_mem_ideal[3][NUM_TYPE_WORKLOAD][MAX_LAYERS] = {0};
//static int rand_cycles[NUM_TYPE_WORKLOAD] = {18273165, 22892262, 12823595, 7993957, 13703872, 9029720, 2136650, 1374090, 45212369, 8373603};
//static int rand_cycles[NUM_TYPE_WORKLOAD] = {10705806, 12984575, 7858393, 5871794, 8739816, 5968501, 3054311, 3329897, 26742640, 5845361};
//static int workload_group[NUM_TYPE_WORKLOAD] = {1, 4, 2, 2, 1, 2, 3, 1,
static int workload_blocks[NUM_TYPE_WORKLOAD] = {3, 5, 3, 4, 3, 3, 1, 1, 10, 4, 3, 4, 2, 6, 6};
static int layer_pointer[NUM_CORE] = {0};
static int last_queue_id = 0;
static int mode = 0;
static bool global_end = false;

static int total_queue_type[MAX_WORKLOAD] = {-1};
static bool total_queue_class[MAX_WORKLOAD] = {0}; // consecutive same type (prevent duplication) 
static uint64_t total_queue_dispatch[MAX_WORKLOAD] = {0}; // dispatched time (in order)
static uint64_t total_queue_finish[MAX_WORKLOAD] = {0}; // if kicked out before execution: 1, if kicked out during execution: 2
static int total_queue_status[MAX_WORKLOAD] = {-1}; // -1: not assigned, 0: in assigned queue, >= 1: layer 
//static int total_queue_priority[MAX_WORKLOAD] = {-1}; // 0 - 11
//static int total_queue_qos[MAX_WORKLOAD] = {-1}; // latency sensitivity of workload (target: (qos + 1) * 1.2 * sp_cycles)
static uint64_t total_queue_target[MAX_WORKLOAD] = {0};
static uint64_t total_queue_runtime[MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)
static int total_queue_core[MAX_WORKLOAD] = {0}; // which core executed it (for debugging)
// for debugging
static int total_queue_throttle[MAX_WORKLOAD] = {0};
static int total_queue_release[MAX_WORKLOAD] = {0};
static int total_queue_acquire[MAX_WORKLOAD] = {0};
static int total_queue_overhead[MAX_WORKLOAD] = {0};
static int total_queue_swap[MAX_WORKLOAD] = {0};
static bool total_queue_dropped[MAX_WORKLOAD] = {0};

//static int gemmini_workload_assigned[NUM_CORE][MAX_ITER][QUEUE_DEPTH] = {-1};
static int gemmini_status[NUM_ARRAY] = {0}; // to track occupied or not (-1: idle, >=0: occupied core), check both -1 and cid while running, check only -1 when starting NN 
static int gemmini_reserved_vel[NUM_CORE] = {0};
static int current_score[NUM_CORE] = {0}; // current dynamic score
static bool likely_miss_ddl_vel[NUM_CORE] = {0};
static int ex_queue[QUEUE_DEPTH] = {-1}; // execution queue (total_queue id)

static int core_gemmini[NUM_CORE][NUM_ARRAY] = {-1}; // to store which gemmini the core has (exclusive access per core) 
static int core_num_gemmini[NUM_CORE] = {0}; // to store how many gemmini are required (exclusive access per core), compare with core_gemmini to see it lacks gemmini
static int core_num_gemmini_share[NUM_CORE] = {0}; // globally seen
static uint64_t core_togo[3][NUM_CORE] = {0}; // track expected cycles per core (exclusive)

// current running workload
static int workload_running[NUM_CORE] = {-1};
static uint64_t global_start_time[NUM_CORE] = {0};

static int dram_util[NUM_CORE] = {0};
static int current_mem_score[NUM_CORE] = {0};

#define NUM_ARRAY_GROUP 5
//static int gemmini_noc_optim[NUM_TYPE_WORKLOAD][NUM_ARRAY_GROUP] = {{5, 4, 1, 8, 0}, {5, 4, 8, 1, 0}, {}, {5, 4, 1, 8, 0}, {}, {}, {5, 4, 1, 8, 0}, {5, 4, 8, 1, 0}, {}, {}};


#if QOS ==1
#if SET == 0
static int gemmini_noc_optim[NUM_TYPE_WORKLOAD][NUM_ARRAY_GROUP] = {{5, 1, 0, 8, 4}, {5, 8, 0, 4, 1}, {0, 1, 4, 8, 5}, {4, 8, 5, 1, 0}, {8, 0, 4, 1, 5}, {4, 8, 5, 1, 0}, {4, 8, 0, 5, 1}, {4, 8, 5, 1, 0}, {1, 4, 8, 0, 5}, {0, 1, 8, 4, 5}, {8, 1, 5, 0, 4}, {5, 1, 8, 4, 0}, {0, 8, 4, 1, 5}, {0, 1, 4, 8, 5}, {4, 5, 8, 0, 1}};
#else
static int gemmini_noc_optim[NUM_TYPE_WORKLOAD][NUM_ARRAY_GROUP] = {{5, 1, 0, 8, 4}, {4, 8, 5, 0, 1}, {0, 1, 4, 8, 5}, {4, 8, 5, 1, 0}, {8, 0, 4, 1, 5}, {0, 5, 4, 1, 8}, {5, 8, 1, 0, 4}, {4, 8, 5, 1, 0}, {1, 4, 8, 0, 5}, {0, 1, 8, 4, 5}, {8, 1, 5, 0, 4}, {5, 1, 8, 4, 0}, {0, 8, 4, 1, 5}, {0, 1, 4, 8, 5}, {4, 5, 8, 0, 1}};
#endif
#else
static int gemmini_noc_optim[NUM_TYPE_WORKLOAD][NUM_ARRAY_GROUP] = {{5, 4, 1, 8, 0}, {5, 4, 8, 1, 0}, {0, 8, 1, 4, 5}, {5, 4, 1, 8, 0}, {0, 8, 1, 4, 5}, {0, 8, 1, 4, 5}, {5, 4, 1, 8, 0}, {5, 4, 8, 1, 0}, {0, 8, 1, 4, 5}, {0, 8, 1, 4, 5}, {8, 1, 5, 0, 4}, {5, 1, 8, 4, 0}, {0, 8, 4, 1, 5}, {0, 1, 4, 8, 5}, {4, 5, 8, 0, 1}};
#endif
static int noc_optim_need_swap[NUM_CORE] = {0};

/*
#if SET == 1
#define RESNUM 6
#define ALEXNUM 7
#define YOLONUM 10
#define GOOGLENUM 13
#define KWSNUM 8
#define RES18NUM 13
#define SQUEEZENUM 32
#define YOLOLITENUM 30
#define BERTBASENUM 3
#define BERTSMALLNUM 14
#else
*/
#define RESNUM 5
#define ALEXNUM 5
#define YOLONUM 5
#define GOOGLENUM 5
#define KWSNUM 5
#define RES18NUM 5
#define SQUEEZENUM 5
#define YOLOLITENUM 5
#define BERTBASENUM 5
#define BERTSMALLNUM 5
//#endif

int hashed_gemmini_pair(int gemmini_id){
   if(gemmini_id == 0)
      return 3;
   else if (gemmini_id == 1)
      return 2;
   else if (gemmini_id == 4)
      return 7;
   else if (gemmini_id == 5)
      return 6;
   else if (gemmini_id == 8)
      return 9;
   else
      return -1;
}
