NVCC = nvcc
NVCC_OPTS = -x cu -O3 -Xcompiler -fopenmp -DGPU_DEVICE_NUMBER=0
NVCC_ARCHS =  -arch=compute_35 -code=compute_35,sm_35
LD_LIBS = -lz -lgomp
vpath %.cpp ..
vpath %.h ..

include ../common.mk

