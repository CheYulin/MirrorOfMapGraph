NVCC = nvcc
NVCC_OPTS = -x cu -O3 -Xcompiler -fopenmp -DGPU_DEVICE_NUMBER=0
NVCC_ARCHS =  -arch=compute_20 -code=compute_20,sm_20,sm_35 
#use the following for GK110 GPU (e.g., Titan, K20)
#NVCC_ARCHS =  -arch=compute_35 -code=compute_35,sm_35 
LD_LIBS = -lz -lgomp
vpath %.cpp ..
vpath %.h ..

include ../common.mk

