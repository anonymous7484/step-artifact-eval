package Parameters;

typedef 8 NUM_PCUS;
typedef 8 NUM_PMUS;

typedef 4 NUM_INPUTS_PER_PCU;
typedef 4 NUM_OUTPUTS_PER_PCU;
typedef 4 NUM_INPUTS_PER_STAGE;
typedef 4 NUM_OUTPUTS_PER_STAGE;
typedef 4 NUM_INPUTS_PER_PMU;
typedef 4 NUM_OUTPUTS_PER_PMU;

typedef 4 NUM_STAGES;

typedef 8 FRAMES_PER_SET;
typedef TLog#(FRAMES_PER_SET) FRAMES_PER_SET_LOG;
typedef Bit#(FRAMES_PER_SET_LOG) FRAME_INDEX;
typedef 32 SETS;
typedef TLog#(SETS) SETS_LOG;
typedef Bit#(SETS_LOG) SET_INDEX;
typedef TMul#(FRAMES_PER_SET, SETS) MAX_ENTRIES;

// The ramulator context does not return requests in order. Therefore, we need a reorder buffer
// 

typedef 4096 RAMULATOR_BITS_PER_REQUEST; // Assuming 8 stacks, HBM2, DDR and Burst-Size 2
typedef 64 RAMULATOR_REORDER_WINDOW_SIZE;
typedef 0 RAMULATOR_PRINT_BUBBLES; // 0: no printing, 1: print bubbles

typedef 0 PRINT_DEBUG_OPERATION; // 0: no printing, 1: print debug operation

typedef 1 USE_RAMULATOR;

endpackage