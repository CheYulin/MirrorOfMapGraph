# The list of directories to operate on.  Could also be defined using
# wildcards.

SUBDIRS = Algorithms/InterestingSubgraph Algorithms/BFS Algorithms/CC Algorithms/SSSP Algorithms/PageRank

# Setup mock targets. There will be one per subdirectory. Note the
# ".all" or ".clean" extension. This will trigger the parameterized
# rules below.
#

ALL = $(foreach DIR,$(SUBDIRS),$(DIR).all)
CLEAN = $(foreach DIR,$(SUBDIRS),$(DIR).clean)

# Define top-level targets.
#

all: $(ALL) doc.all

clean: $(CLEAN) doc.clean

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
