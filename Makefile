NVCC = nvcc

#NVCC_OPTS = -O3 -Xptxas -dlcm=cg
NVCC_OPTS = -g -G
NVCC_ARCHS = -gencode arch=compute_20,code=sm_20
LD_LIBS = -lz

#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

all: graphio.o simpleBFS simplePageRank simpleSSSP samplePageRank

graphio.o: graphio.cpp graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) --compiler-options "-std=c++0x"

simpleBFS.o: simpleBFS.cu GASEngine.h bfs.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cu GASEngine.h pagerank.h graphio.h Makefile
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

clean:
	rm -f simpleBFS simplePageRank simpleSSSP samplePageRank *.o


regress:
	make -C regressions

