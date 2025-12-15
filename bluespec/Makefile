###  -*-Makefile-*-
# Copyright (c) 2018-2020 Bluespec, Inc. All Rights Reserved

# ================================================================

# Optional: override default BDPI C source
BDPI_C_SRC ?= dpi/matmul_tile.c dpi/ramulator_wrapper.cpp

# Add SMU directory to search path
BSC_PATH1 = src/SMU:

# ================================================================
# Ramulator flags for linking
#
# Use $ORIGIN for Linux runtime lookup; $$ escapes for make so the linker
# sees a single $ORIGIN token. Disable --as-needed so libramulator is kept
# even though BSC places linker flags before object files.
BSC_LINK_FLAGS += -Xl "-Wl,--no-as-needed" \
		-Xl "-L./ramulator2" \
		-Xl "-lramulator"    \
		-Xl "-lstdc++"       \
		-Xl "-Wl,-rpath,$$ORIGIN/../../ramulator2"\
		-Xc++ "-I./ramulator2/ext/spdlog/include" \
		-Xc++ "-std=c++17" \
		-Xc++ "-I./ramulator2/ext/yaml-cpp/include" \
		-Xc++ "-I./ramulator2/src"

include ./res/Include_Makefile.mk