include ../setup.mk

all: config.o BFS

config.o: config.cpp ../../getvalue.h config.h Makefile register.h
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) $(RAND_OPTS)

BFS.o: bfs.cu bfs.h Makefile ../../GASengine/enactor_vertex_centric.cuh
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS)

BFS: BFS.o config.o
	nvcc -o $@ $^ $(NVCC_OPTS) $(LD_LIBS)

clean:
	rm -f BFS *.o
