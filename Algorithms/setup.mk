# Common setup for the Algorithms directory that is included into each Makefile.

NVCC = nvcc
MGPU_PATH = ../../moderngpu2
NVCC_OPTS = -O3 -lineinfo -I. -I../../ -I$(MGPU_PATH)/include -L$(MGPU_PATH)
#NVCC_OPTS = -G -g -Xptxas -v -Xcompiler -fopenmp  -I. -I../../
#NVCC_ARCHS = -gencode=arch=compute_20,code=sm_20
#NVCC_ARCHS = -gencode arch=compute_35,code=sm_35

# Conditionally specify location of libcuda if not set and platform has odd default.
# Note: libcuda is under lib64/stubs for Linux and /usr/local/cuda/lib for OSX.
ifeq ($(OS),Windows_NT)
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
    endif
    ifeq ($(UNAME_S),Darwin)
    CUDALIB?=-L/usr/local/cuda/lib
    endif
endif

LD_LIBS = -lmgpu $(CUDALIB) -lcuda

# Note: atomicMin() for int64 requires compute 3.5+ (the others must be commented out)
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" 
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
SM_TARGETS = $(GEN_SM20) $(GEN_SM35) 

# Uncomment if you have	gcc 4.5	and would like to use its improved random number facility.
#RAND_OPTS=--compiler-options "-std=c++0x"
