NVCC = g++ -fopenmp
vpath %.cpp ..
vpath %.h ..

NVCC_OPTS = -O3 -DUSE_CPU -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -I /usr/local/cuda-5.5/include
NVCC_ARCHS = 
LD_LIBS = -lz

#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

all: graphio.o simpleBFS.omp simplePageRank.omp simpleSSSP.omp

graphio.o: graphio.cpp graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simpleBFS.o: simpleBFS.cpp GASEngine.h bfs.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cpp GASEngine.h pagerank.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleSSSP.o: simpleSSSP.cpp GASEngine.h sssp.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleBFS.omp: simpleBFS.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simplePageRank.omp: simplePageRank.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simpleSSSP.omp: simpleSSSP.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

clean:
	rm -f simpleBFS.omp simplePageRank.omp simpleSSSP.omp *.o


regress:
	make -C regressions

