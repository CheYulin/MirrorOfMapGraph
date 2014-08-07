# Generate documentation from the code.
all: api

# Copy the papers into the html output directory.
# Note: This is intentionally avoiding the .svn directory.
# If you use a hierarchical directory for the papers, then
# be careful to include the directory structure but not the
# .svn files for each child.
papers.dir:
	-mkdir -p api/html/papers
	cp -r papers/* api/html/papers

# Build the documentation.
api: papers.dir
	doxygen Doxyfile

# Clean the output directory.
clean:
	rm -rf api

# Publish the documentation on the web site (requires authentication)
publish: clean api
	rsync -qrzPl --perms --chmod=ugo+X,ugo+r -e ssh api www.systap.com:/srv/www/htdocs/systap.com/mapgraph/
