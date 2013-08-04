NVCC = g++ -fopenmp
THRUST_INCLUDE_DIR=/usr/local/cuda-5.5/include

NVCC_OPTS = -O3 -DUSE_CPU -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -I $(THRUST_INCLUDE_DIR)
NVCC_ARCHS = 
#LD_LIBS = -lz -ltbb
LD_LIBS = -lz

vpath %.cpp ..
vpath %.h ..

include ../common.mk
