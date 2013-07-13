NVCC = nvcc
NVCC_OPTS = -O3 -Xcompiler -fopenmp -DGPU_DEVICE_NUMBER=0
NVCC_ARCHS = -arch=compute_20 -code=compute_20,sm_20,sm_35
LD_LIBS = -lz -lgomp
vpath %.cpp ..
vpath %.h ..


#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

all: graphio.o simpleBFS simplePageRank simpleSSSP

graphio.o: graphio.cpp graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simpleBFS.o: simpleBFS.cpp GASEngine.h bfs.h graphio.h Makefile
	nvcc -x cu -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cpp GASEngine.h pagerank.h graphio.h Makefile
	nvcc -x cu -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleSSSP.o: simpleSSSP.cpp GASEngine.h sssp.h graphio.h Makefile
	nvcc -x cu -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleBFS: simpleBFS.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simplePageRank: simplePageRank.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

simpleSSSP: simpleSSSP.o graphio.o
	nvcc -o $@ $^ $(LD_LIBS)

clean:
	rm -f simpleBFS simplePageRank simpleSSSP *.o


regress:
	make -C regressions

