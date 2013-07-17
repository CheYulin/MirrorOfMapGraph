NVCC = nvcc

NVCC_OPTS = -O3 -Xptxas -dlcm=cg
#NVCC_OPTS = -g -G
NVCC_ARCHS = -gencode arch=compute_20,code=sm_20
LD_LIBS = -lz

# Uncomment if you have	gcc 4.5	and would like to use its improved random number facility.
#RAND_OPTS=--compiler-options "-std=c++0x"

all: graphio.o adaptiveBC_direct adaptiveBC_undirect simpleCC samplePageRank simpleBFS simplePageRank simpleSSSP  

graphio.o: graphio.cpp graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) $(RAND_OPTS)

simpleBFS.o: simpleBFS.cu GASEngine.h bfs.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cu GASEngine.h pagerank.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleCC.o: simpleCC.cu GASEngine.h concomp.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

adaptiveBC_undirect.o: adaptiveBC_undirect.cu adaptiveBC_undirect.h GASEngine.h graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

adaptiveBC_direct.o: adaptiveBC_direct.cu adaptiveBC_direct.h GASEngine.h graphio.h Makefile
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
	
adaptiveBC_undirect: adaptiveBC_undirect.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)
	
adaptiveBC_direct: adaptiveBC_direct.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

clean:
	rm -f adaptiveBC_direct adaptiveBC_undirect simpleBFS simplePageRank simpleCC simpleSSSP samplePageRank *.o


regress:
	make -C regressions

