/*
 Copyright (C) SYSTAP, LLC 2006-2014.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

typedef unsigned int uint;
#include <stdio.h> 
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <vector>
#include <sssp.h>
#include <iostream>
#include <omp.h>

#include <config.h>

// Utilities and correctness-checking
//#include <test/b40c_test_util.h>

// Graph construction utils

#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/random.cuh>

#include <GASengine/csr_problem.cuh>
#include <GASengine/enactor_vertex_centric.cuh>

using namespace b40c;
using namespace graph;
using namespace std;

/**
 * \brief Initialize the specified CUDA device.
 *
 * @param device The device number.
 */
void cudaInit(const int device)
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess)
  {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) error_id,
        cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit (EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
  {
    printf("There are no available device(s) that support CUDA\n");
  }
  else
  {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev)
  {
    if (dev == device)
    {
      cudaSetDevice(dev);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);

      printf("Running on this device:");
      printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

      // Console log
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      printf(
          "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
          driverVersion / 1000, (driverVersion % 100) / 10,
          runtimeVersion / 1000, (runtimeVersion % 100) / 10);
      printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
          deviceProp.major, deviceProp.minor);

      printf(
          "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
          (float) deviceProp.totalGlobalMem / 1048576.0f,
          (unsigned long long) deviceProp.totalGlobalMem);
    }
  }
}

void correctTest(int nodes, int* reference_dists, int* h_dists)
{
  bool pass = true;
  printf("Correctness testing ...");
  for (int i = 0; i < nodes; i++)
  {
    if (reference_dists[i] != h_dists[i])
    {
//      printf("Incorrect value for node %d: CPU value %d, GPU value %d\n", i, reference_dists[i], h_dists[i]);
      pass = false;
    }
  }
  if (pass)
    printf("passed\n");
  else
    printf("failed\n");
}

template<typename VertexId, typename Value, typename SizeT>
void CPUSSSP(CsrGraph<VertexId, Value, SizeT> const &graph, VertexId* dist, VertexId src)
{

// initialize dist[] and pred[] arrays. Start with vertex s by setting
// dist[] to 0.

  const int n = graph.nodes;
  for (int i = 0; i < n; i++)
    dist[i] = 100000000;
  vector<bool> visited(n);
  dist[src] = 0;

// find vertex in ever-shrinking set, V-S, whose dist value is smallest
// Recompute potential new paths to update all shortest paths

  double startTime = omp_get_wtime();
  while (true)
  {
// find shortest distance so far in unvisited vertices
    int u = -1;
    int sd = 100000000; // assume not reachable

    for (int i = 0; i < n; i++)
    {
      if (!visited[i] && dist[i] < sd)
      {
        sd = dist[i];
        u = i;
      }
    }
    if (u == -1)
    {
      break; // no more progress to be made
    }

// For neighbors of u, see if length of best path from s->u + weight
// of edge u->v is better than best path from s->v.
    visited[u] = true;

    for (int j = graph.row_offsets[u]; j < graph.row_offsets[u + 1]; ++j)
    {
      int v = graph.column_indices[j]; // the neighbor v
      long newLen = dist[u]; // compute as long
      newLen += graph.edge_values[j]; // sum with (u,v) weight

      if (newLen < dist[v])
      {
        dist[v] = newLen;
      }
    }
  }

  double EndTime = omp_get_wtime();

  std::cout << "CPU time took: " << (EndTime - startTime) * 1000 << " ms"
      << std::endl;
}

void printUsageAndExit(char* algo_name)
{
  std::cout
      << "Usage: " << algo_name
      << " [-graph (-g) graph_file] [-output (-o) output_file] [-sources src_file] [-SSSP \"variable1=value1 variable2=value2 ... variable3=value3\" -help ] [-c config_file]\n";
  std::cout << "     -help display the command options\n";
  std::cout << "     -graph or -g specify a sparse matrix in Matrix Market (.mtx) format\n";
  std::cout << "     -output or -o specify file for output result\n";
  std::cout << "     -sources or -s set the starting vertices file\n";
  std::cout << "     -c set the SSSP options from the configuration file\n";
  std::cout
      << "     -parameters (-p) set the options.  Options include the following:\n";
  Config::printOptions();

  exit(0);
}

int main(int argc, char **argv)
{

  const char* outFileName = 0;
//  int src[1];
//  bool g_undirected;
  const bool g_stream_from_host = false;
  const bool g_with_value = true;
  const bool g_mark_predecessor = false;
  bool g_verbose = false;
  typedef int VertexId; // Use as the node identifier type
  typedef int Value; // Use as the value type
  typedef typename sssp::DataType SizeT; // Use as the graph size type
  char* graph_file = NULL;
  CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);
  char source_file_name[1000] = "";
//  int device = 0;
//  double max_queue_sizing = 1.3;
  Config cfg;

  for (int i = 1; i < argc; i++)
  {

    if (strncmp(argv[i], "-help", 100) == 0) // print the usage information
      printUsageAndExit(argv[0]);

    else if (strncmp(argv[i], "-graph", 100) == 0
        || strncmp(argv[i], "-g", 100) == 0)
    { //input graph
      i++;

      graph_file = argv[i];

    }

    else if (strncmp(argv[i], "-output", 100) == 0 || strncmp(argv[i], "-o", 100) == 0)
    { //output file name
      i++;
      outFileName = argv[i];
    }

    else if (strncmp(argv[i], "-sources", 100) == 0 || strncmp(argv[i], "-s", 100) == 0)
    { //the file containing starting vertices
      i++;
      strcpy(source_file_name, argv[i]);
//			printf("source_file_name=%s\n", source_file_name);
    }

    else if (strncmp(argv[i], "-parameters", 100) == 0 || strncmp(argv[i], "-p", 100) == 0)
    { //The SSSP specific options
      i++;
      cfg.parseParameterString(argv[i]);
    }
    else if (strncmp(argv[i], "-c", 100) == 0)
    { //use a configuration file to specify the SSSP options instead of command line
      i++;
      cfg.parseFile(argv[i]);
    }
    else
    {
      printf("Wrong command!\n");
      exit(1);
    }
  }

  if (graph_file == NULL)
  {
    printUsageAndExit(argv[0]);
    exit(1);
  }

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);

  printf("Running on host: %s\n", hostname);

  cudaInit(cfg.getParameter<int>("device"));

  int directed = cfg.getParameter<int>("directed");
  int origin = cfg.getParameter<int>("origin");
  int iter_num = cfg.getParameter<int>("iter_num");
  if (builder::BuildMarketGraph<g_with_value>(graph_file, csr_graph,
      !directed) != 0)
    return 1;

  int run_CPU = cfg.getParameter<int>("run_CPU");

  int num_srcs = 0;
  int* srcs = NULL;

  const int max_src_num = 1000;

  if (strcmp(source_file_name, ""))
  {
    if (strcmp(source_file_name, "RANDOM") == 0)
    {
      printf("Using random starting vertices!\n");
      num_srcs = cfg.getParameter<int>("num_src");
      srcs = new int[num_srcs];
      printf("Using %d random starting vertices!\n", num_srcs);
      srand (time(NULL));int
      count = 0;
      while (count < num_srcs)
      {
        int tmp_src = rand() % csr_graph.nodes;
        if (csr_graph.row_offsets[tmp_src + 1]
            - csr_graph.row_offsets[tmp_src] > 0)
        {
          srcs[count++] = tmp_src;
        }
      }

    }
    else
    {
      printf("Using source file: %s!\n", source_file_name);
      FILE* src_file;
      if ((src_file = fopen(source_file_name, "r")) == NULL)
      {
        printf("Source file open error!\n");
        exit(0);
      }

      srcs = new int[max_src_num];
      for (num_srcs = 0; num_srcs < max_src_num; num_srcs++)
      {
        if (fscanf(src_file, "%d\n", &srcs[num_srcs]) != EOF)
        {
          if (origin == 1)
            srcs[num_srcs]--; //0-based index
        }
        else
          break;
      }
      printf("number of srcs used: %d\n", num_srcs);
    }

  }
  else
  {
    int src_node = cfg.getParameter<int>("src");
    int origin = cfg.getParameter<int>("origin");
    num_srcs = 1;
    srcs = new int[1];
    srcs[0] = src_node;
    if (origin == 1)
      srcs[0]--;
    printf("Single source vertex: %d\n", srcs[0]);
  }

  VertexId* reference_dists;
  if (strcmp(source_file_name, "") == 0 && run_CPU) //Do correctness test only with single starting vertex
  {
    reference_dists = (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
    int src = cfg.getParameter<int>("src");
//    int origin = cfg.getParameter<int>("origin");

    if (origin == 1)
      src--;

    CPUSSSP(
        csr_graph,
        reference_dists,
        src);

//    for (int i = 0; i < csr_graph.nodes; i++)
//    {
//      printf("%d\n", reference_dists[i]);
//    }
    //    return 0;
  }

// Allocate problem on GPU
  int num_gpus = 1;
  typedef GASengine::CsrProblem<sssp, VertexId, SizeT, Value,
      g_mark_predecessor, g_with_value> CsrProblem;
  CsrProblem csr_problem(cfg);
  if (csr_problem.FromHostProblem(g_stream_from_host, csr_graph.nodes,
      csr_graph.edges, csr_graph.column_indices,
      csr_graph.row_offsets, csr_graph.edge_values, csr_graph.row_indices,
      csr_graph.column_offsets, num_gpus, directed))
    exit(1);

  const bool INSTRUMENT = true;

  GASengine::EnactorVertexCentric<INSTRUMENT> vertex_centric(cfg, g_verbose);

  for (int i = 0; i < num_srcs; i++)
  {
    int tmpsrcs[1];
    tmpsrcs[0] = srcs[i];
    cudaError_t retval = cudaSuccess;

    retval = vertex_centric.EnactIterativeSearch<CsrProblem, sssp>(csr_problem,
        csr_graph.row_offsets, directed, 1, tmpsrcs, iter_num);

    if (retval && (retval != cudaErrorInvalidDeviceFunction))
    {
      exit(1);
    }
  }

  Value* h_values = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
  csr_problem.ExtractResults(h_values);

  if (strcmp(source_file_name, "") == 0 && run_CPU)
  {
    correctTest(csr_graph.nodes, reference_dists, h_values);
    free(reference_dists);
  }

  if (outFileName)
  {
    FILE* f = fopen(outFileName, "w");
    for (int i = 0; i < csr_graph.nodes; ++i)
    {
      fprintf(f, "%d\n", h_values[i]);
    }

    fclose(f);
  }

  return 0;
}
