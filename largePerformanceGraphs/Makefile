# Makefile to fetch and install graph data for regression testing

# Each graph lives in its own directory
SUBDIRS = ak2010 belgium_osm delaunay_n13 delaunay_n21 coAuthorsDBLP kron_g500-logn21 soc-LiveJournal1 webbase-1M kron_g500-logn20 bitcoin wikipedia-20070206

.PHONY: $(GRAPHS)

#fetches all graphs, extracts and sets up files for tests
all: recurse

#only download the graphs, but do not proceed further
fetch: recurse

# does not remove the data.
clean: recurse

# cleans everything including the downloaded graphs
realclean: recurse

#recurse into each subdirectory and try to build the provided targets
recurse:
	for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir $(MAKECMDGOALS); done

$(GRAPHS):
	$(MAKE) -C $@
