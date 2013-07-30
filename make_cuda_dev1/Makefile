NVCC = nvcc
NVCC_OPTS = -x cu -O3 -Xcompiler -fopenmp -DGPU_DEVICE_NUMBER=1
NVCC_ARCHS =  -arch=compute_20 -code=compute_20,sm_20,sm_35
LD_LIBS = -lz -lgomp
vpath %.cpp ..
vpath %.h ..

include ../common.mk

