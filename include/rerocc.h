

#ifndef REROCC_H
#define REROCC_H

#include <stdint.h>

#define STR1(x) #x
#define STR(x) STR1(x)
#define EXTRACT(a, size, offset) (((~(~0 << size) << offset) & a) >> offset)

#define CUSTOMX_OPCODE(x) CUSTOM_ ## x
#define CUSTOM_0 0b0001011
#define CUSTOM_1 0b0101011
#define CUSTOM_2 0b1011011
#define CUSTOM_3 0b1111011

#define CUSTOMX(X, xd, xs1, xs2, rd, rs1, rs2, funct) \
  CUSTOMX_OPCODE(X)                     |             \
  (rd                 << (7))           |             \
  (xs2                << (7+5))         |             \
  (xs1                << (7+5+1))       |             \
  (xd                 << (7+5+2))       |             \
  (rs1                << (7+5+3))       |             \
  (rs2                << (7+5+3+5))     |             \
  (EXTRACT(funct, 7, 0) << (7+5+3+5+5))

// Standard macro that passes rd, rs1, and rs2 via registers
#define ROCC_INSTRUCTION_DSS(X, rd, rs1, rs2, funct) \
	ROCC_INSTRUCTION_R_R_R_R(X, rd, rs1, rs2, funct, 10, 11, 12)

#define ROCC_INSTRUCTION_DS(X, rd, rs1, funct) \
	ROCC_INSTRUCTION_R_R_I(X, rd, rs1, 0, funct, 10, 11)

#define ROCC_INSTRUCTION_D(X, rd, funct) \
	ROCC_INSTRUCTION_R_I_I(X, rd, 0, 0, funct, 10)

#define ROCC_INSTRUCTION_SS(X, rs1, rs2, funct) \
	ROCC_INSTRUCTION_I_R_R(X, 0, rs1, rs2, funct, 11, 12)

#define ROCC_INSTRUCTION_S(X, rs1, funct) \
	ROCC_INSTRUCTION_I_R_I(X, 0, rs1, 0, funct, 11)
// rd, rs1, and rs2 are data
// rd_n, rs_1, and rs2_n are the register numbers to use
#define ROCC_INSTRUCTION_R_R_R_R(X, rd, rs1, rs2, funct, rd_n, rs1_n, rs2_n) { \
    register uint64_t rd_  asm ("x" # rd_n);                                 \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;               \
    register uint64_t rs2_ asm ("x" # rs2_n) = (uint64_t) rs2;               \
    asm volatile (                                                           \
        ".word " STR(CUSTOMX(X, 1, 1, 1, rd_n, rs1_n, rs2_n, funct)) "\n\t"  \
        : "=r" (rd_)                                                         \
        : [_rs1] "r" (rs1_), [_rs2] "r" (rs2_));                             \
    rd = rd_;                                                                \
  }

#define ROCC_INSTRUCTION_R_R_I(X, rd, rs1, rs2, funct, rd_n, rs1_n) {     \
    register uint64_t rd_  asm ("x" # rd_n);                              \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;            \
    asm volatile (                                                        \
        ".word " STR(CUSTOMX(X, 1, 1, 0, rd_n, rs1_n, rs2, funct)) "\n\t" \
        : "=r" (rd_) : [_rs1] "r" (rs1_));                                \
    rd = rd_;                                                             \
  }

#define ROCC_INSTRUCTION_R_I_I(X, rd, rs1, rs2, funct, rd_n) {           \
    register uint64_t rd_  asm ("x" # rd_n);                             \
    asm volatile (                                                       \
        ".word " STR(CUSTOMX(X, 1, 0, 0, rd_n, rs1, rs2, funct)) "\n\t"  \
        : "=r" (rd_));                                                   \
    rd = rd_;                                                            \
  }

#define ROCC_INSTRUCTION_I_R_R(X, rd, rs1, rs2, funct, rs1_n, rs2_n) {    \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;            \
    register uint64_t rs2_ asm ("x" # rs2_n) = (uint64_t) rs2;            \
    asm volatile (                                                        \
        ".word " STR(CUSTOMX(X, 0, 1, 1, rd, rs1_n, rs2_n, funct)) "\n\t" \
        :: [_rs1] "r" (rs1_), [_rs2] "r" (rs2_));                         \
  }

#define ROCC_INSTRUCTION_I_R_I(X, rd, rs1, rs2, funct, rs1_n) {         \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;          \
    asm volatile (                                                      \
        ".word " STR(CUSTOMX(X, 0, 1, 0, rd, rs1_n, rs2, funct)) "\n\t" \
        :: [_rs1] "r" (rs1_));                                          \
  }

#define ROCC_INSTRUCTION_I_I_I(X, rd, rs1, rs2, funct) {                 \
    asm volatile (                                                       \
        ".word " STR(CUSTOMX(X, 0, 0, 0, rd, rs1, rs2, funct)) "\n\t" ); \
  }

#define REROCC_ACQUIRE (0)
#define REROCC_RELEASE (1)
#define REROCC_ASSIGN (2)
#define REROCC_INFO (3)
#define REROCC_FENCE (4)
#define REROCC_CFLUSH (5)
#define REROCC_CFG_READ_TRACKER (6)
#define REROCC_CFG_READ_ID (7)
#define REROCC_CFG_WRITE_TRACKER (8)
#define REROCC_CFG_WRITE_ID (8)

#define REROCC_CFG_EPOCH (0)
#define REROCC_CFG_RATE (1)
#define REROCC_CFG_LAST_REQS (2)
#define REROCC_CFG_EPOCHRATE (3)
#define REROCC_CFG_OFFSETTER_OFFSET (4)
#define REROCC_CFG_OFFSETTER_BASE0  (5)
#define REROCC_CFG_OFFSETTER_BASE1  (6)
#define REROCC_CFG_OFFSETTER_BASE2  (7)
#define REROCC_CFG_OFFSETTER_BASE3  (8)
#define REROCC_CFG_OFFSETTER_SIZE0  (9)
#define REROCC_CFG_OFFSETTER_SIZE1  (10)
#define REROCC_CFG_OFFSETTER_SIZE2  (11)
#define REROCC_CFG_OFFSETTER_SIZE3  (12)



// Attemps to assign a local tracker to one of the accelerators in the OH mask
// If no accelerators are available, return 0, else return the oh vector
// of the assigned accelerator
inline uint64_t rerocc_acquire(uint64_t tracker, uint64_t mask) {
  uint64_t op1 = tracker;
  uint64_t op2 = mask;
  uint64_t r;
  ROCC_INSTRUCTION_DSS(0, r, op1, op2, REROCC_ACQUIRE);
  return r;
}

// Releases the accelerator currently allocated to tracker
inline void rerocc_release(uint64_t tracker) {
  uint64_t op1 = tracker;
  ROCC_INSTRUCTION_S(0, op1, REROCC_RELEASE);
}

inline uint64_t rerocc_swap(uint64_t tracker_old, uint64_t tracker_new, uint64_t mask) {
  uint64_t op1 = tracker_old;
  uint64_t op2 = mask;
  uint64_t r;
  ROCC_INSTRUCTION_DSS(0, r, op1, op2, REROCC_ACQUIRE);
  if (!r)
    return 0; // failed to swap
  else{
    uint64_t op1 = tracker_new;
    ROCC_INSTRUCTION_S(0, op1, REROCC_RELEASE);
    return 1;
  }
  //return r;
}

// Assigns local opcode given by opcode to one of the trackers
inline void rerocc_assign(uint8_t opcode, uint64_t tracker) {
  uint64_t op1 = tracker;
  uint64_t op2 = opcode;
  ROCC_INSTRUCTION_SS(0, op1, op2, REROCC_ASSIGN);
}

// Gets the number of trackers on this hart
inline uint64_t rerocc_ntrackers() {
  uint64_t r;
  ROCC_INSTRUCTION_D(0, r, REROCC_INFO);
  return r;
}

// Fences a specific single tracker on this hart
inline void rerocc_fence(uint64_t tracker) {
  uint64_t op1 = tracker;
  ROCC_INSTRUCTION_S(0, op1, REROCC_FENCE);
  asm volatile("fence");
}

inline void rerocc_cflush(void* addr) {
  uint64_t op1 = (uint64_t)addr;
  ROCC_INSTRUCTION_S(0, op1, REROCC_CFLUSH);
}



inline uint64_t rerocc_read_cfg_mgr_id(uint64_t id, uint32_t cfg_id) {
  uint64_t op1 = ((uint64_t) cfg_id << 32) | (id & 0xffffffff);
  uint64_t r;
  ROCC_INSTRUCTION_DS(0, r, op1, REROCC_CFG_READ_ID);
  return r;
}

inline uint64_t rerocc_write_cfg_mgr_id(uint64_t id, uint64_t wdata, uint32_t cfg_id, bool read) {
  uint64_t op1 = ((uint64_t) cfg_id << 32) | (id & 0xffffffff);
  uint64_t op2 = wdata;
  uint64_t r = 0;
  if (read) {
    ROCC_INSTRUCTION_DSS(0, r, op1, op2, REROCC_CFG_WRITE_ID);
  } else {
    ROCC_INSTRUCTION_SS(0, op1, op2, REROCC_CFG_WRITE_ID);
  }
  return r;
}

inline uint64_t rerocc_write_cfg_tracker(uint64_t tracker, uint64_t wdata, uint32_t cfg_id, bool read) {
  uint64_t op1 = ((uint64_t) cfg_id << 32) | (tracker & 0xffffffff);
  uint64_t op2 = wdata;
  uint64_t r = 0;
  if (read) {
    ROCC_INSTRUCTION_DSS(0, r, op1, op2, REROCC_CFG_WRITE_TRACKER);
  } else {
    ROCC_INSTRUCTION_SS(0, op1, op2, REROCC_CFG_WRITE_TRACKER);
  }
  return r;
}

inline uint64_t rerocc_read_cfg_tracker(uint64_t tracker, uint32_t cfg_id) {
  uint64_t op1 = ((uint64_t) cfg_id << 32) | (tracker & 0xffffffff);
  uint64_t r;
  ROCC_INSTRUCTION_DS(0, r, op1, REROCC_CFG_READ_TRACKER);
  return r;
}


inline uint64_t rerocc_cfg_epochrate_by_tracker(uint64_t tracker, uint64_t epoch, uint64_t max_req, bool read) {
  max_req = max_req << 4;
  uint64_t wdata = (epoch << 32) | (max_req & 0xffffffff);
  return rerocc_write_cfg_tracker(tracker, wdata, REROCC_CFG_EPOCHRATE, read);
}

inline uint64_t rerocc_cfg_epochrate_by_mgr_id(uint64_t mgr_id, uint64_t epoch, uint64_t max_req, bool read) {
  uint64_t wdata = (epoch << 32) | (max_req & 0xffffffff);
  return rerocc_write_cfg_mgr_id(mgr_id, wdata, REROCC_CFG_EPOCHRATE, read);
}

inline void rerocc_cfg_offsetter_by_tracker(uint64_t tracker, void* base, size_t size, size_t offsetter_id) {
  uint32_t base_cfg_id = REROCC_CFG_OFFSETTER_BASE0 + offsetter_id;
  uint32_t size_cfg_id = REROCC_CFG_OFFSETTER_SIZE0 + offsetter_id;
  rerocc_write_cfg_tracker(tracker, (uint64_t)base, base_cfg_id, false);
  rerocc_write_cfg_tracker(tracker, (uint64_t)size, size_cfg_id, false);
}


/* // address falls into this range redirects to DRAM by bypassing */
/* // can have multiple configured bypass address range at the same time */
/* // initialize configured bypass range upon rerocc_fence */
/* inline void rerocc_bypass(uint64_t tracker, void* addr_start, void* addr_end){                     */
/*   uint64_t op1 = ((uint64_t) addr_end << 16) | tracker; */
/*   uint64_t op2 = (uint64_t) addr_start; */
/*   ROCC_INSTRUCTION_SS(0, op1, op2, REROCC_BYPASS); */
/* }     */


static uint64_t read_cycles_re() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
}
#endif
