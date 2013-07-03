NVCC = nvcc

NVCC_OPTS = -O3 --restrict -Xptxas -dlcm=cg
NVCC_ARCHS = -gencode arch=compute_20,code=sm_20
LD_LIBS = -lz
# Uncomment if you have	gcc 4.5	and would like to use its improved random number facility.
#RAND_OPTS=--compiler-options "-std=c++0x"

all: graphio.o samplePageRank simpleBFS simplePageRank simpleSSSP  

graphio.o: graphio.cpp graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) $(RAND_OPTS)

simpleBFS.o: simpleBFS.cu GASEngine.h bfs.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cu GASEngine.h pagerank.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleCC.o: simpleCC.cu GASEngine.h concomp.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

samplePageRank.o: samplePageRank.cu GASEngine.h pagerank.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleSSSP.o: simpleSSSP.cu GASEngine.h sssp.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleBFS: simpleBFS.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simplePageRank: simplePageRank.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

samplePageRank: samplePageRank.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simpleSSSP: simpleSSSP.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simpleCC: simpleCC.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

clean:
	rm -f simpleBFS simplePageRank simpleCC simpleSSSP samplePageRank *.o


regress:
	make -C regressions

