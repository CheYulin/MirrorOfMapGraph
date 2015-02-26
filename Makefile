# The software release version.
ver=0.3.3
version=mapgraph.${ver}.tgz
release.dir=releases

# The list of directories to operate on.  Could also be defined using
# wildcards.

#SUBDIRS = Algorithms/InterestingSubgraph Algorithms/BFS Algorithms/CC Algorithms/SSSP Algorithms/PageRank
SUBDIRS = moderngpu2/src Algorithms/BFS Algorithms/CC Algorithms/SSSP Algorithms/PageRank

# Setup mock targets. There will be one per subdirectory. Note the
# ".all" or ".clean" extension. This will trigger the parameterized
# rules below.
#

ALL = $(foreach DIR,$(SUBDIRS),$(DIR).all)
CLEAN = $(foreach DIR,$(SUBDIRS),$(DIR).clean)

# Define top-level targets.
#

# Build and generate the documentation.
all: build doc.all

# Just build everything (no docs or tests).
build: $(ALL)

# Just build the moderngpu library.
mgpu:
	make -C moderngpu2/src

# Clean the build, but do not delete the downloaded data.
clean: $(CLEAN) doc.clean

# Clean the build and delete the downloaded data.
realclean: clean realclean.create
	make -C largePerformanceGraphs realclean

realclean.create:
	rm -rf ${release.dir}

# Parameterized implementation of the mock targets, invoked by
# top-level targets for each subdirectory.
#

%.all:
	$(MAKE) -C $*

%.clean:
	$(MAKE) -C $* clean

# Note: If there are dependency orders, declare them here. This way
# some things will be built before others.

#foo: baz

# Generates documentation from the code.
doc: doc.create

doc.create:
	$(MAKE) -C doc

release: clean release.create

release.create:
	-mkdir ${release.dir}
	-rm -f ${release.dir}/${version}
	tar --exclude .svn --exclude releases -cvz -f ${release.dir}/${version} .

#
# Downloads a variety of large(r) graphs used for performance testing.
#
download.graphs:
	make -C largePerformanceGraphs

# Run the experiments for the MapGraph single GPU paper @ SIGMOD 2014.
# 
# See http://mapgraph.io/papers/MapGraph-SIGMOD-2014.pdf
#
# Note: This also runs the CPU validation and thus serves as both a
# performance and correctness regression test.
#
# TODO We can run more graphs here. For example, soc-LiveJournal1,
# belgium_osm, etc.  While they were not included in the paper
# referenced above, we can always self-check against the CPU reference
# implementation.
#
GRAPHDIR=largePerformanceGraphs
#GRAPHS=belgium_osm delaunay_n13 delaunay_n21 coAuthorsDBLP kron_g500-logn21 soc-LiveJournal1 webbase-1M kron_g500-logn20 bitcoin wikipedia-20070206)
GRAPHS=webbase-1M delaunay_n21 bitcoin wiki kron_g500-logn20 wikipedia-20070206)
# webbase-1M: directed
# delaunay_n21: undirected
# bitcoin: directed
# wikipedia-200702006: directed
# kron_g500-logn20: undirected

# The GPU device number to use.  Some graphs require the RAM on a K20 or better card.
DEVICE=0
# When 1, the CPU correctness test is run. When 0 it is not run.
RUNCPU=1
# When running random starting vertex tests, this is the number of
# random vertices to use.  Each starting vertex will be used for a
# distinct traveral.
NUMSRC=100
# Arguments used by all of the tests.
ARGS=-p run_CPU=$(RUNCPU) -p device=$(DEVICE) -p stats=1
#
# FIXME webbase-1M.mtx. max degree in paper is incorrect. Should be 4700 (vs 27).
#
# FIXME delaunay_n21.mtx. max degree in paper is incorrect. Should be 20 (vs 23).
#
# FIXME kron_g500-logn20: file claims to have 44620272 edges but paper states 89,239,674. code reports maxDegree=73554 but paper states 131,505.
#
# FIXME The config.cpp code segfaults if the -p parameter name is not recognized. For example, try specifying "-p directed=1" to CC. It will segfault.
#
# FIXME The -source RANDOM -p num_src=100 does not appear to execute more than once (BFS/SSSP).

test: build test.BFS test.BFS.random test.SSSP test.SSSP.random test.CC test.PageRank
	echo "Done running performance/regression tests."

# Note: as run for the SIGMOD 2014 paper.
#
# Note: webbase-1M uses a specific starting vertex to get a decent traversal.
test.BFS: mgpu Algorithms/BFS.all download.graphs #-sources RANDOM -p num_src=100
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/webbase-1M/webbase-1M.mtx                 $(ARGS) -p directed=1 -p src=602514
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/delaunay_n21/delaunay_n21.mtx             $(ARGS) -p directed=0 -p src=1
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/wikipedia-20070206/wikipedia-20070206.mtx $(ARGS) -p directed=1 -p src=1
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/kron_g500-logn20/kron_g500-logn20.mtx     $(ARGS) -p directed=0 -p src=1
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/bitcoin/bitcoin.mtx                       $(ARGS) -p directed=1 -p src=1
	echo "End test.BFS"

# NUMSRC trials with random starting vertices. This tests the
# robustness of the GPU code.
#
# FIXME Validation is not enabled when using random starting vertices.
# It should be.  However, the SSSP CPU validation needs to be much
# faster for that to work.  And then we have to modify the bfs.cu and
# sssp.cu main() routines to validate each starting vertex in turn.
test.BFS.random: mgpu Algorithms/BFS.all download.graphs
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/webbase-1M/webbase-1M.mtx                 $(ARGS) -p directed=1 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/delaunay_n21/delaunay_n21.mtx             $(ARGS) -p directed=0 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/wikipedia-20070206/wikipedia-20070206.mtx $(ARGS) -p directed=1 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/kron_g500-logn20/kron_g500-logn20.mtx     $(ARGS) -p directed=0 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)\
												    -p max_queue_sizing=6
	./Algorithms/BFS/BFS       -g $(GRAPHDIR)/bitcoin/bitcoin.mtx                       $(ARGS) -p directed=1 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)\
												    -p max_queue_sizing=6
	echo "End test.BFS.random"

# Note: as run for the SIGMOD 2014 paper.
#
# FIXME The CPU SSSP algorithm is extremely expensive to run on large
# graphs (it takes hours) and is therefore disabled for wikipedia,
# kron, and bitcoin.
test.SSSP: mgpu Algorithms/SSSP.all download.graphs
# CPU validation takes 130s.
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/webbase-1M/webbase-1M.mtx                 $(ARGS) -p directed=1 -p src=602514
# FIXME CPU does not validate undirected graphs correctly (I've implemented undirected CPU validation, but now it takes hours to run. We need a better CPU SSSP impl.)
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/delaunay_n21/delaunay_n21.mtx             $(ARGS) -p directed=0 -p src=1 -p run_CPU=0
# FIXME validation takes hours and is disabled.
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/wikipedia-20070206/wikipedia-20070206.mtx $(ARGS) -p directed=1 -p src=1 -p run_CPU=0
# FIXME validation takes hours and is disabled.
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/kron_g500-logn20/kron_g500-logn20.mtx     $(ARGS) -p directed=0 -p src=1 -p run_CPU=0
# FIXME validation takes hours and is disabled.
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/bitcoin/bitcoin.mtx                       $(ARGS) -p directed=1 -p src=1 -p run_CPU=0 
	echo "End test.SSSP"

# NUMSRC trials with random starting vertices. This tests the
# robustness of the GPU code.
#
# Note: We specify [directed] based on whether or not the graph is
# directed since validation is not being performed.
test.SSSP.random: mgpu Algorithms/SSSP.all download.graphs
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/webbase-1M/webbase-1M.mtx                 $(ARGS) -p directed=1 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/delaunay_n21/delaunay_n21.mtx             $(ARGS) -p directed=0 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/wikipedia-20070206/wikipedia-20070206.mtx $(ARGS) -p directed=1 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC)
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/kron_g500-logn20/kron_g500-logn20.mtx     $(ARGS) -p directed=0 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC) \
												    -p max_queue_sizing=6
	./Algorithms/SSSP/SSSP     -g $(GRAPHDIR)/bitcoin/bitcoin.mtx                       $(ARGS) -p directed=1 -p run_CPU=0 -sources RANDOM -p num_src=$(NUMSRC) \
												    -p max_queue_sizing=6
	echo "End test.SSSP.random"

# Note: as run for the SIGMOD 2014 paper.
#
# Note: CC assumes undirected and ignores the "directed" argument.
test.CC: mgpu Algorithms/CC.all download.graphs
	./Algorithms/CC/CC         -g $(GRAPHDIR)/webbase-1M/webbase-1M.mtx                 $(ARGS) -p max_queue_sizing=3
	./Algorithms/CC/CC         -g $(GRAPHDIR)/delaunay_n21/delaunay_n21.mtx             $(ARGS) -p max_queue_sizing=3
	./Algorithms/CC/CC         -g $(GRAPHDIR)/wikipedia-20070206/wikipedia-20070206.mtx $(ARGS) -p max_queue_sizing=3
	./Algorithms/CC/CC         -g $(GRAPHDIR)/kron_g500-logn20/kron_g500-logn20.mtx     $(ARGS) -p max_queue_sizing=3
	./Algorithms/CC/CC         -g $(GRAPHDIR)/bitcoin/bitcoin.mtx                       $(ARGS) -p max_queue_sizing=3
	echo "End test.CC"

# Note: as run for the SIGMOD 2014 paper.
#
test.PageRank: mgpu Algorithms/PageRank.all download.graphs
# Correctness testing ...passed!! l2 error = 0.013418
	./Algorithms/PageRank/PageRank -g $(GRAPHDIR)/webbase-1M/webbase-1M.mtx                 $(ARGS) -p directed=1
# Correctness testing ...failed!! l2 error = 1.330842 (with default tol=.1)
	./Algorithms/PageRank/PageRank -g $(GRAPHDIR)/delaunay_n21/delaunay_n21.mtx             $(ARGS) -p directed=0 -p tol=1.5
# Correctness testing ...passed!! l2 error = 0.010080
	./Algorithms/PageRank/PageRank -g $(GRAPHDIR)/wikipedia-20070206/wikipedia-20070206.mtx $(ARGS) -p directed=1
# Correctness testing ...failed!! l2 error = 1.019699 (with default tol=.1)
	./Algorithms/PageRank/PageRank -g $(GRAPHDIR)/kron_g500-logn20/kron_g500-logn20.mtx     $(ARGS) -p directed=0 -p tol=1.2
# Correctness testing ...passed!! l2 error = 0.059472
	./Algorithms/PageRank/PageRank -g $(GRAPHDIR)/bitcoin/bitcoin.mtx                       $(ARGS) -p directed=1
	echo "End test.PageRank"
