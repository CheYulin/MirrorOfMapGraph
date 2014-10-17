#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#define xfree free
#include <mpi.h>
#include "make_graph.h"

struct partition{
 unsigned long* edges;
 unsigned long numedges;
};
#ifdef __cplusplus
extern "C" {
#endif

int gen_2d(int log_numverts, int edges_per_vert, int ppr, struct partition* parts);

#ifdef __cplusplus
}
#endif
