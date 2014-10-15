include ../setup.mk

all: config.o CC
	
config.o: config.cpp ../../getvalue.h config.h Makefile register.h
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) $(RAND_OPTS)

CC.o: cc.cu cc.h Makefile ../../GASengine/enactor_vertex_centric.cuh
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 

CC: CC.o config.o
	nvcc -o $@ $^ $(NVCC_OPTS) $(LD_LIBS)
	
clean:
	rm -f CC *.o

