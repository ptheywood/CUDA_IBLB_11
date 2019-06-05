# Objects required by executable
objects = main.o LatticeBoltzmann.o ImmersedBoundary.o seconds.o

NVCC=nvcc

# Gencode arguments to target different architectures
SMS_KEPLER = 30 35 37 
SMS_PASCAL = 60 
SMS_VOLTA  = 70


# Get version of CUDA to determine which device architectures to build for
NVCC_MAJOR = $(shell ""$(NVCC)"" --version | sed -n -r 's/.*(V([0-9]+).([0-9]+).([0-9]+))/\2/p')
NVCC_GE_8_0 = $(shell [ $(NVCC_MAJOR) -ge 8 ] && echo true)
NVCC_GE_9_0 = $(shell [ $(NVCC_MAJOR) -ge 9 ] && echo true)

# Start with Kepler cards only
SMS = $(SMS_KEPLER)
# If the version of NVCC is new enough, target Pascal
ifeq ($(NVCC_GE_8_0),true)
SMS += $(SMS_PASCAL)
endif
# If the version of NVCC is new enough, target Volta
ifeq ($(NVCC_GE_9_0),true)
SMS += $(SMS_VOLTA)
endif


# Convert list of SM numbers to nvcc arguments
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM := $(lastword $(sort $(SMS)))
# Add the last one again to support future architectures
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif


# Rule for building the executable
all: $(objects)
		$(NVCC) $(GENCODE_FLAGS) -lineinfo $(objects) -o IBLB

# Rule for compiling individual c++ files
%.o: %.cpp
		$(NVCC) -x cu $(GENCODE_FLAGS) -lineinfo -I. -dc $< -o $@

# Rule for compiling individual CUDA files
%.o: %.cu
		$(NVCC) -x cu $(GENCODE_FLAGS) -lineinfo -I. -dc $< -o $@

# Rule for cleaning the build directory
clean:
		rm -f *.o IBLB
