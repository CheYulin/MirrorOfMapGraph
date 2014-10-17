CC = mpicc
CFLAGS = -std=c99 -O3 -DGRAPH_GENERATOR_MPI -DGRAPHGEN_DISTRIBUTED_MEMORY -DNDEBUG # -g -pg
# CFLAGS = -std=c99 -DGRAPH_GENERATOR_MPI -DGRAPHGEN_DISTRIBUTED_MEMORY -g
LDFLAGS = -O3
# LDFLAGS = -g
MPICC = mpicc

GENERATOR_SOURCES = graph_generator.c make_graph.c splittable_mrg.c utils.c generator_mpi_api.c
GENERATOR_OBJS = graph_generator.o make_graph.o splittable_mrg.o utils.o generator_mpi_api.o
GENERATOR_HEADERS = graph_generator.h make_graph.h mod_arith_32bit.h mod_arith_64bit.h mod_arith.h mod_arith_xmt.h splittable_mrg.h utils.h user_settings.h mrg_transitions.c generator_mpi_api.h

all: generator_test_mpi_objs libgen.a

generator_test_mpi_objs: $(GENERATOR_SOURCES) $(GENERATOR_HEADERS)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -c $(GENERATOR_SOURCES) -lm
main:
	mpicc -o main main.c -L./ -I./ -lgen

libgen.a:$(GENERATOR_OBJS)
	ar rcs $@ $^
	
clean:
	-rm -f *.o libgen.a 
