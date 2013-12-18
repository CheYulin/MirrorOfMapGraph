NVCC = nvcc

NVCC_OPTS = -O3  -Xptxas -v  -Xcompiler -fopenmp  -I.
#NVCC_OPTS = -G -g -Xcompiler -fopenmp  -I.
#NVCC_ARCHS = -gencode=arch=compute_20,code=sm_20
#NVCC_ARCHS = -gencode arch=compute_35,code=sm_35
LD_LIBS = -lz -lgomp

GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" 
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
SM_TARGETS = $(GEN_SM20) $(GEN_SM35) 

# Uncomment if you have	gcc 4.5	and would like to use its improved random number facility.
#RAND_OPTS=--compiler-options "-std=c++0x"

all: graphio.o  simpleSSSP #sampleBC simpleBFS simplePageRank  simpleCC

graphio.o: graphio.cpp graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) $(RAND_OPTS)

simpleBFS.o: simpleBFS.cu GASEngine.h bfs.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS)

simplePageRank.o: simplePageRank.cu GASEngine.h pagerank.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 

simpleCC.o: simpleCC.cu GASEngine.h concomp.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 
				
sampleBC.o: sampleBC.cu sampleBC.h GASEngine.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 

samplePageRank.o: samplePageRank.cu GASEngine.h pagerank.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 

simpleSSSP.o: simpleSSSP.cu GASEngine.h sssp.h b40c/graph/GASengine/enactor_vertex_centric.cuh graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(SM_TARGETS) 

simpleBFS: simpleBFS.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simplePageRank: simplePageRank.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simpleSSSP: simpleSSSP.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simpleCC: simpleCC.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)
				
sampleBC: sampleBC.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)
	
clean:
	rm -f sampleBC simpleBFS simplePageRank simpleCC simpleSSSP samplePageRank *.o


regress:
	make -C regressions

