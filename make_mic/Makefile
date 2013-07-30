NVCC = icc -openmp -mmic
THRUST_INCLUDE_DIR=../../..

NVCC_OPTS = -O3 -DUSE_CPU -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -I $(THRUST_INCLUDE_DIR) -I ../../../zlib-1.2.8
NVCC_ARCHS = 
LD_LIBS = ../../../zlib-1.2.8/libz.a

vpath %.cpp ..
vpath %.h ..

include ../common.mk

