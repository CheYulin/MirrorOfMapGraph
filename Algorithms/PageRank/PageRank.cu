/**
Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This work was (partially) funded by the DARPA XDATA program under
AFRL Contract #FA8750-13-C-0002.

This material is based upon work supported by the Defense Advanced
Research Projects Agency (DARPA) under Contract No. D14PC00029.
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
#include <PageRank.h>
#include <iostream>
#include <time.h>
#include <numeric>

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

void cudaInit(int device)
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

template<typename VertexId, typename Value, typename SizeT>
void CPUPR(CsrGraph<VertexId, Value, SizeT> const &graph, Value* dist)
{

  printf("Running CPU code...");fflush(stdout);

// initialize dist[] and pred[] arrays. Start with vertex s by setting
// dist[] to 0.

  const int n = graph.nodes;
  for (int i = 0; i < n; i++)
    dist[i] = 0.15;

  //compute d_num_out_edges
  VertexId* num_out_edge = (VertexId*) malloc((n + 1) * sizeof(VertexId));

  adjacent_difference(graph.row_offsets, graph.row_offsets + n + 1, num_out_edge);
  num_out_edge++;
//  for(int i=0; i<n; i++)
//    printf("%d %d\n", graph.row_offsets[i+1], num_out_edge[i]);

  bool changed = true;

// find vertex in ever-shrinking set, V-S, whose dist value is smallest
// Recompute potential new paths to update all shortest paths

  const time_t startTime = time(NULL);
  int iter_count = 0;
  while (changed)
  {
    changed = false;
    for (int v = 0; v < n; v++)
    {
      Value sumnb = 0.0;
      for (int j = graph.column_offsets[v]; j < graph.column_offsets[v + 1]; ++j)
      {
        int nb = graph.row_indices[j]; // the neighbor v
        sumnb += dist[nb] / (Value) num_out_edge[nb];
      }
      sumnb = 0.15 + 0.85 * sumnb;
      if (fabs(sumnb - dist[v]) >= 0.01)
      {
        changed = true;
      }

      dist[v] = sumnb;
    }
    iter_count++;
  }

  const time_t EndTime = time(NULL);

  std::cout << "CPU iterations: " << iter_count << std::endl;
  std::cout << "CPU time took: " << difftime(EndTime, startTime) * 1000 << " ms"
      << std::endl;
}

void printUsageAndExit(char* algo_name)
{
  std::cout
      << "Usage: " << algo_name << " [-graph (-g) graph_file] [-output (-o) output_file] [-PR \"variable1=value1 variable2=value2 ... variable3=value3\" -help ] [-c config_file]\n";
  std::cout << "     -help display the command options\n";
  std::cout << "     -graph specify a sparse matrix in Matrix Market (.mtx) format\n";
  std::cout << "     -output or -o specify file for output result\n";
  std::cout << "     -c set the PR options from the configuration file\n";
  std::cout
      << "     -parameters (-p) set the options.  Options include the following:\n";
  Config::printOptions();

  exit(0);
}

template<typename Value, typename SizeT>
Value l2norm(Value* v1, Value* v2, SizeT n)
{
  Value result = 0.0;
  for (unsigned int i = 0; i < n; ++i)
    result += (v1[i] - v2[i]) * (v1[i] - v2[i]);

  return sqrt(result);
}

template<typename Value, typename SizeT>
Value l2norm(Value* v, SizeT n)
{
  Value result = 0.0;
  for (unsigned int i = 0; i < n; ++i)
    result += v[i] * v[i];

  return sqrt(result);
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
  typedef typename pagerank::DataType Value; // Use as the value type
  typedef int SizeT; // Use as the graph size type
  char* graph_file = NULL;
  CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);
//  char source_file_name[100] = "";
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

    else if (strncmp(argv[i], "-parameters", 100) == 0 || strncmp(argv[i], "-p", 100) == 0)
    { //The PR specific options
      i++;
      cfg.parseParameterString(argv[i]);
    }
    else if (strncmp(argv[i], "-c", 100) == 0)
    { //use a configuration file to specify the PR options instead of command line
      i++;
      cfg.parseFile(argv[i]);
    }
  }

  if (graph_file == NULL)
  {
    printUsageAndExit(argv[0]);
    exit(1);
  }

  char hostname[1024] = "localhost";
#ifdef gethostname
  gethostname(hostname, 1023);
#endif

  printf("Running on host: %s\n", hostname);

  cudaInit(cfg.getParameter<int>("device"));
  const int directed = cfg.getParameter<int>("directed");

  if (builder::BuildMarketGraph<g_with_value>(graph_file, csr_graph,
      false) != 0)
    return 1;

  {
	  const int stats = cfg.getParameter<int>("stats");
	  if(stats) {
		  csr_graph.PrintHistogram();
	  }
  }

  Value* reference_dists;

  int run_CPU = cfg.getParameter<int>("run_CPU");
  if (run_CPU)
  {
    reference_dists = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
    CPUPR(csr_graph, reference_dists);
  }

// Allocate problem on GPU
  int num_gpus = 1;
  typedef GASengine::CsrProblem<pagerank, VertexId, SizeT, Value, g_mark_predecessor, g_with_value> CsrProblem;
  CsrProblem csr_problem(cfg);

  if (csr_problem.FromHostProblem(g_stream_from_host, csr_graph.nodes,
      csr_graph.edges, csr_graph.column_indices,
      csr_graph.row_offsets, csr_graph.edge_values, csr_graph.row_indices,
      csr_graph.column_offsets, num_gpus, directed))
    exit(1);

  const bool INSTRUMENT = true;

  GASengine::EnactorVertexCentric<CsrProblem, pagerank, INSTRUMENT> vertex_centric(cfg, g_verbose);

  cudaError_t retval = cudaSuccess;

  int iter_num = cfg.getParameter<int>("iter_num");
  int threshold = cfg.getParameter<int>("threshold");
  retval = vertex_centric.EnactIterativeSearch(csr_problem, csr_graph.row_offsets, directed, csr_graph.nodes, NULL, iter_num, threshold);

  if (retval && (retval != cudaErrorInvalidDeviceFunction))
  {
    exit(1);
  }

  Value* h_values = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
  csr_problem.ExtractResults(h_values);

  if (run_CPU)
  {
	const double tol = cfg.getParameter<double>("tol");
    printf("Correctness testing ...");fflush(stdout);
    const Value l2error = l2norm(reference_dists, h_values, csr_graph.nodes) / l2norm(reference_dists, csr_graph.nodes); // / sqrt((Value)csr_graph.nodes);
    const bool pass = l2error < tol;
    printf("%s! l2 error = %f\n", pass?"passed!":"failed!", l2error);
    free(reference_dists);
    if(!pass) {
		fprintf(stderr, "correctness test failed.");
		exit(1);
    }
  }

  if (outFileName)
  {
    FILE* f = fopen(outFileName, "w");
    for (int i = 0; i < csr_graph.nodes; ++i)
    {
      fprintf(f, "%f\n", h_values[i]);
    }

    fclose(f);
  }

  return 0;
}
