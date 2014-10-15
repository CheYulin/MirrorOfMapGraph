include ../setup.mk

all: config.o SSSP

config.o: config.cpp ../../getvalue.h config.h Makefile register.h
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) $(RAND_OPTS)

SSSP.o: sssp.cu sssp.h ../../GASengine/enactor_vertex_centric.cuh Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 

SSSP: SSSP.o config.o
	nvcc -o $@ $^ $(NVCC_OPTS) $(LD_LIBS)
	
clean:
	rm -f SSSP *.o

