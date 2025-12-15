# ================================================================
# BSV/Verilog build environment - Generic build infrastructure
# 
# This include file provides the core build system for Bluespec projects:
# - Bluesim compilation and linking
# - Verilog compilation and simulation
# - Build directory management
# - Cleanup targets
# 
# Project-specific configurations (DPI, custom flags, etc.) should be
# defined in the main Makefile that includes this file.
# ================================================================

TUTORIAL ?= .
RESOURCES_DIR ?= $(TUTORIAL)/res
TOPLANG ?= BSV

# Build directory structure
BUILD_DIR = build
B_SIM_BUILD_DIR = $(BUILD_DIR)/bluesim
V_BUILD_DIR = $(BUILD_DIR)/verilog
DPI_BUILD_DIR = $(BUILD_DIR)/dpi
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj

# Absolute path to the bundled Ramulator library for LD_LIBRARY_PATH.
RAMULATOR_LIB_DIR := $(abspath ramulator2)

ifeq ($(TOPLANG),BSV)
  SRC_EXT=bsv
else ifeq ($(TOPLANG),BH)
  SRC_EXT=bs
else
  SRC_EXT=TOPLANG_NOT_DEFINED
endif

TOPFILE ?= src/Top.$(SRC_EXT)
TOPMODULE ?= mkTop

BSC_COMP_FLAGS += -keep-fires -aggressive-conditions -no-warn-action-shadowing -check-assert -cpp \
	+RTS -K2G -RTS -show-range-conflict -show-schedule -Xc "lm" -steps-max-intervals 10000000 -v -show-stats -sched-dot

BSC_LINK_FLAGS += -keep-fires -Xc "lm" -show-stats
BSC_PATHS = -p $(BSC_PATH1)src:$(RESOURCES_DIR):+

V_SIM ?= iverilog

# ================================================================
# BDPI Support (static object compilation)

# Automatically generate object file list from source files
BDPI_OBJ = $(patsubst dpi/%.c,build/obj/%.o,$(filter %.c,$(BDPI_C_SRC))) \
           $(patsubst dpi/%.cpp,build/obj/%.o,$(filter %.cpp,$(BDPI_C_SRC)))

ifeq ($(BDPI_C_SRC),)
# No BDPI C source specified, do nothing
else
# Generic rule for .c files
$(OBJ_DIR)/%.o: dpi/%.c
	@echo Compiling $< to $@...
	mkdir -p $(OBJ_DIR)
	# Build position-independent code so it can be linked into the Bluesim shared object
	gcc -fPIC -c $< -o $@
	@echo Built: $@

# Generic rule for .cpp files (TODO: Ideally move the ramulator part into the outer file.)
$(OBJ_DIR)/%.o: dpi/%.cpp
	@echo Compiling $< to $@...
	mkdir -p $(OBJ_DIR)
	# Build position-independent code so it can be linked into the Bluesim shared object
	cd ramulator2/src && g++ -fPIC -c ../../$< -I. -I../ext/spdlog/include -I../ext/yaml-cpp/include -std=c++17 -o ../../$@
	@echo Built: $@
endif

# ================================================================
# Bluesim

B_SIM_DIRS = -simdir $(B_SIM_BUILD_DIR) -bdir $(B_SIM_BUILD_DIR) -info-dir $(B_SIM_BUILD_DIR)
B_SIM_EXE = $(BIN_DIR)/$(TOPMODULE)_b_sim

.PHONY: b_all
b_all: b_compile b_link b_sim

.PHONY: b_compile
b_compile:
	mkdir -p $(B_SIM_BUILD_DIR)
	mkdir -p $(BIN_DIR)
	@echo Compiling for Bluesim ...
	bsc -u -sim $(B_SIM_DIRS) $(BSC_COMP_FLAGS) $(BSC_PATHS) -g $(TOPMODULE) $(TOPFILE)
	@echo Compiling for Bluesim finished

.PHONY: b_link
b_link: $(BDPI_OBJ)
	@echo Linking for Bluesim ...
	bsc -e $(TOPMODULE) -sim \
		-o $(B_SIM_EXE) \
		$(B_SIM_DIRS) \
		$(BSC_LINK_FLAGS) \
		$(BSC_PATHS) \
		$(BDPI_OBJ)
	@echo Linking for Bluesim finished

.PHONY: b_sim
b_sim:
	@echo Bluesim simulation ...
	LD_LIBRARY_PATH=.:"$(RAMULATOR_LIB_DIR)" $(B_SIM_EXE)
	@echo Bluesim simulation finished

.PHONY: b_sim_vcd
b_sim_vcd:
	@echo Bluesim simulation and dumping VCD in $(BUILD_DIR)/dump.vcd ...
	LD_LIBRARY_PATH=.:"$(RAMULATOR_LIB_DIR)" $(B_SIM_EXE) -V
	@echo Bluesim simulation and dumping VCD in $(BUILD_DIR)/dump.vcd finished

# ================================================================
# Verilog

V_DIRS = -vdir $(V_BUILD_DIR)/rtl -bdir $(V_BUILD_DIR) -info-dir $(V_BUILD_DIR)
V_SIM_EXE = $(BIN_DIR)/$(TOPMODULE)_v_sim

.PHONY: v_all
v_all: v_compile v_link v_sim

.PHONY: v_compile
v_compile:
	mkdir -p $(V_BUILD_DIR)
	mkdir -p $(V_BUILD_DIR)/rtl
	mkdir -p $(BIN_DIR)
	@echo Compiling for Verilog ...
	bsc -u -verilog $(V_DIRS) $(BSC_COMP_FLAGS) $(BSC_PATHS) -g $(TOPMODULE) $(TOPFILE)
	@echo Compiling for Verilog finished

.PHONY: v_link
v_link:
	@echo Linking for Verilog sim ...
	bsc -e $(TOPMODULE) -verilog -o $(V_SIM_EXE) $(V_DIRS) -vsim $(V_SIM) $(V_BUILD_DIR)/rtl/$(TOPMODULE).v
	@echo Linking for Verilog sim finished

.PHONY: v_sim
v_sim:
	@echo Verilog simulation...
	$(V_SIM_EXE)
	@echo Verilog simulation finished

.PHONY: v_sim_vcd
v_sim_vcd:
	@echo Verilog simulation and dumping VCD in $(BUILD_DIR)/dump.vcd ...
	$(V_SIM_EXE) +bscvcd
	@echo Verilog simulation and dumping VCD in $(BUILD_DIR)/dump.vcd finished



# ================================================================
# Cleanup

.PHONY: clean
clean:
	rm -f $(B_SIM_BUILD_DIR)/* $(V_BUILD_DIR)/* $(OBJ_DIR)/* *~ src/*~ $(BDPI_OBJ)

.PHONY: full_clean
full_clean:
	rm -rf $(BUILD_DIR) *~ src/*~
	rm -f *.vcd

.PHONY: clean_build
clean_build:
	rm -rf $(BUILD_DIR)

.PHONY: clean_bin
clean_bin:
	rm -rf $(BIN_DIR)

.PHONY: clean_obj
clean_obj:
	rm -rf $(OBJ_DIR)