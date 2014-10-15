include ../setup.mk

all: config.o PageRank
	
config.o: config.cpp ../../getvalue.h config.h Makefile register.h
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) $(RAND_OPTS)
	
PageRank.o: PageRank.cu PageRank.h Makefile ../../GASengine/enactor_vertex_centric.cuh
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 
	
PageRank: PageRank.o config.o
	nvcc -o $@ $^ $(NVCC_OPTS) $(LD_LIBS)
	
clean:
	rm -f PageRank *.o

