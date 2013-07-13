NVCC = icc -openmp

NVCC_OPTS = -O3 -DUSE_CPU -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB  -I ../../..
NVCC_ARCHS = 
LD_LIBS = -lz -ltbb

vpath %.cpp ..
vpath %.h ..


#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

all: graphio.o simpleBFS.tbb simplePageRank.tbb simpleSSSP.tbb

graphio.o: graphio.cpp graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simpleBFS.o: simpleBFS.cpp GASEngine.h bfs.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cpp GASEngine.h pagerank.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleSSSP.o: simpleSSSP.cpp GASEngine.h sssp.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleBFS.tbb: simpleBFS.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simplePageRank.tbb: simplePageRank.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simpleSSSP.tbb: simpleSSSP.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

clean:
	rm -f simpleBFS.tbb simplePageRank.tbb simpleSSSP.tbb *.o


regress:
	make -C regressions

