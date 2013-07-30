
#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

all: graphio.o simpleBFS simplePageRank simpleSSSP simpleCC sampleBC

graphio.o: graphio.cpp graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simpleBFS.o: simpleBFS.cpp GASEngine.h bfs.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

simplePageRank.o: simplePageRank.cpp GASEngine.h pagerank.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleSSSP.o: simpleSSSP.cpp GASEngine.h sssp.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

simpleCC.o: simpleCC.cpp GASEngine.h concomp.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

sampleBC.o: sampleBC.cpp sampleBC.h GASEngine.h graphio.h Makefile
	$(NVCC) -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 


simpleBFS: simpleBFS.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simplePageRank: simplePageRank.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simpleSSSP: simpleSSSP.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)

simpleCC: simpleCC.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)
        
sampleBC: sampleBC.o graphio.o
	$(NVCC) -o $@ $^ $(LD_LIBS)
        
clean:
	rm -f simpleBFS simplePageRank simpleSSSP simpleCC *.o sampleBC

regress:
	make -C regressions

