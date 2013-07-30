NVCC = icc -openmp
THRUST_INCLUDE_DIR=../../..

NVCC_OPTS = -O3 -DUSE_CPU -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB  -I $(THRUST_INCLUDE_DIR)
NVCC_ARCHS = 
LD_LIBS = -lz -ltbb

vpath %.cpp ..
vpath %.h ..

include ../common.mk

