NVCC = icc -openmp -mmic

NVCC_OPTS = -O3 -DUSE_CPU -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -I ../../.. -I ../../../zlib-1.2.8
NVCC_ARCHS = 
LD_LIBS = ../../../zlib-1.2.8/libz.a

vpath %.cpp ..
vpath %.h ..


#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

all: graphio.o simpleBFS.mic simplePageRank.mic simpleSSSP.mic

graphio.o: graphio.cpp graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simpleBFS.o: simpleBFS.cpp GASEngine.h bfs.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cpp GASEngine.h pagerank.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleSSSP.o: simpleSSSP.cpp GASEngine.h sssp.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleBFS.mic: simpleBFS.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simplePageRank.mic: simplePageRank.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simpleSSSP.mic: simpleSSSP.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

clean:
	rm -f simpleBFS.mic simplePageRank.mic simpleSSSP.mic *.o


regress:
	make -C regressions

