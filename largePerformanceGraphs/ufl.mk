#common make file fragment for ufl graph datasets
#just define GRAPH_NAME prior to including this fragment

GRAPH_TAR  = $(GRAPH_NAME).tar.gz

setup: $(GRAPH_NAME).mtx

$(GRAPH_NAME).mtx: $(GRAPH_TAR)
	tar xvfz $(GRAPH_TAR)
# Note: The original files have appropriate headers (#rows, #cols). We can use as is.
	cp $(GRAPH_NAME)/$(GRAPH_NAME).mtx .
# Note: If you want to use these files with graphlab you need to apply this script to
# transform them. However, that will wipe out the #of rows and #of columns in the header
# which means that you can't use the file with MapGraph.
#	$(MATRIX2SNAP) $(GRAPH_NAME)/$(GRAPH_NAME).mtx $(GRAPH_NAME).mtx
	rm -rf $(GRAPH_NAME)

# Do not remove the data.
clean:

# Remove the data.
realclean: clean
	rm $(GRAPH_NAME).mtx
	rm $(GRAPH_TAR)

