GRAPH_NAME = bitcoin
GRAPH_URL = https://www.dropbox.com/s/994xui3sgk5pa4c/bitcoin.mtx

WGET := wget
GRAPH_FILE := $(notdir $(GRAPH_URL))

all: setup fetch

fetch: $(GRAPH_FILE)

$(GRAPH_FILE):
	$(WGET) -N $(GRAPH_URL)

setup: $(GRAPH_NAME).mtx

# Do not remove the data.
clean:

# Remove the data.
realclean: clean
	rm $(GRAPH_NAME).mtx
