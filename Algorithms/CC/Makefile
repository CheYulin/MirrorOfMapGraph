NVCC = nvcc

NVCC_OPTS = -O3 -Xcompiler -fopenmp  -I. -I../..
#NVCC_OPTS = -G -g -Xptxas -v -Xcompiler -fopenmp  -I.
#NVCC_ARCHS = -gencode=arch=compute_20,code=sm_20
#NVCC_ARCHS = -gencode arch=compute_35,code=sm_35
LD_LIBS = -lgomp

GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" 
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
SM_TARGETS = $(GEN_SM20) $(GEN_SM35) 

# Uncomment if you have	gcc 4.5	and would like to use its improved random number facility.
#RAND_OPTS=--compiler-options "-std=c++0x"

all: config.o CC
	
config.o: config.cpp ../../getvalue.h config.h Makefile register.h
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) $(RAND_OPTS)

CC.o: cc.cu ../../GASEngine.h cc.h Makefile ../../GASengine/enactor_vertex_centric.cuh
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 

CC: CC.o config.o
	nvcc -o $@ $^ $(LD_LIBS)
	
clean:
	rm -f CC *.o

