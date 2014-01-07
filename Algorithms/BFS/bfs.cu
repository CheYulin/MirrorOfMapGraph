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
#include <bfs.h>
#include <iostream>
#include <omp.h>

#include <config.h>

// Utilities and correctness-checking
//#include <test/b40c_test_util.h>

// Graph construction utils

#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/random.cuh>

#include <b40c/graph/GASengine/csr_problem.cuh>
#include <b40c/graph/GASengine/enactor_vertex_centric.cuh>

using namespace b40c;
using namespace graph;
using namespace std;

template<
    typename VertexId,
    typename Value,
    typename SizeT>
void CPUBFS(
    int test_iteration,
    const CsrGraph<VertexId, Value, SizeT> &csr_graph,
    VertexId *source_path,
    VertexId src)
{
  // (Re)initialize distances
  for (VertexId i = 0; i < csr_graph.nodes; i++)
  {
    source_path[i] = -1;
  }
  source_path[src] = 0;
  VertexId search_depth = 0;

  // Initialize queue for managing previously-discovered nodes
  std::deque<VertexId> frontier;
  frontier.push_back(src);

  double startTime = omp_get_wtime();
  //
  // Perform BFS on CPU
  //
  while (!frontier.empty())
  {
    // Dequeue node from frontier
    VertexId dequeued_node = frontier.front();
    frontier.pop_front();
    VertexId neighbor_dist = source_path[dequeued_node] + 1;

    // Locate adjacency list
    int edges_begin = csr_graph.row_offsets[dequeued_node];
    int edges_end = csr_graph.row_offsets[dequeued_node + 1];

    for (int edge = edges_begin; edge < edges_end; edge++)
    {

      // Lookup neighbor and enqueue if undiscovered
      VertexId neighbor = csr_graph.column_indices[edge];
      if (source_path[neighbor] == -1)
      {
        source_path[neighbor] = neighbor_dist;
        if (search_depth < neighbor_dist)
        {
          search_depth = neighbor_dist;
        }
        frontier.push_back(neighbor);
      }
    }
  }

  double EndTime = omp_get_wtime();

  std::cout << "CPU time took: " << (EndTime - startTime) * 1000 << " ms"
      << std::endl;
  search_depth++;
}

bool cudaInit(int device)
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
    return false;
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

      break;
    }
  }

  return true;
}

void correctTest(int nodes, int* reference_labels, int* h_labels)
{
  bool pass = true;
  printf("Correctness testing ...");
  for (int i = 0; i < nodes; i++)
  {
    if (reference_labels[i] != h_labels[i])
    {
      printf("Incorrect value for node %d: CPU value %d, GPU value %d\n", i, reference_labels[i], h_labels[i]);
      pass = false;
    }
  }
  if (pass)
    printf("passed\n");
}

void printUsageAndExit()
{
  std::cout
      << "Usage: ./BFS [-graph (-g) graph_file] [-sources src_file] [-BFS \"variable1=value1 variable2=value2 ... variable3=value3\" -help ] [-c config_file]\n";
  std::cout << "     -help display the command options\n";
  std::cout << "     -graph specify a sparse matrix in Matrix Market (.mtx) format\n";
  std::cout << "     -sources or -s set starting vertices file\n";
  std::cout << "     -c set the BFS options from the configuration file\n";
  std::cout
      << "     -BFS set the options.  Options include the following:\n";
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
  typedef int SizeT; // Use as the graph size type
  char* graph_file = NULL;
  CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);
  char source_file_name[100] = "";
//  int device = 0;
//  double max_queue_sizing = 1.3;
  Config cfg;

  for (int i = 1; i < argc; i++)
  {
    if (strncmp(argv[i], "-help", 100) == 0) // print the usage information
      printUsageAndExit();
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
    }

    else if (strncmp(argv[i], "-BFS", 100) == 0)
    { //The BFS specific options
      i++;
      cfg.parseParameterString(argv[i]);
    }
    else if (strncmp(argv[i], "-c", 100) == 0)
    { //use a configuration file to specify the BFS options instead of command line
      i++;
      cfg.parseFile(argv[i]);
    }
  }

  if (graph_file == NULL)
  {
    printUsageAndExit();
    exit(1);
  }

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);

  printf("Running on host: %s\n", hostname);

  int directed = cfg.getParameter<int>("directed");

  if (builder::BuildMarketGraph<g_with_value>(graph_file, csr_graph,
      !directed) != 0)
    exit(1);

  bool cudaEnabled = cudaInit(cfg.getParameter<int>("device"));
  VertexId* reference_labels = (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
  if (strcmp(source_file_name, "") == 0)//Do correctness test only with single starting vertex
  {
    int test_iteration = 1;
    int src = cfg.getParameter<int>("src");
    int origin = cfg.getParameter<int>("origin");

    if(origin == 1)
      src--;

    CPUBFS(
        test_iteration,
        csr_graph,
        reference_labels,
        src);
    //    return 0;
  }

  if (!cudaEnabled)
    return 0;

  VertexId* h_labels = (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
  int* h_dists = (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
//    VertexId* reference_check = (g_quick) ? NULL : reference_labels;
//
//    //Allocate host-side node_value array (both ref and gpu-computed results)
//    Value* ref_node_values = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
  Value* h_node_values = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
//    Value* ref_node_value_check = (g_quick) ? NULL : ref_node_values;
//
//    //Allocate host-side sigma value array (both ref and gpu-computed results)
//    Value* ref_sigmas = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
  Value* h_sigmas = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
//    Value* ref_sigmas_check = (g_quick) ? NULL : ref_sigmas;
  Value* h_deltas = (Value*) malloc(sizeof(Value) * csr_graph.nodes);

// Allocate problem on GPU
  int num_gpus = 1;
  typedef GASengine::CsrProblem<bfs, VertexId, SizeT, Value,
      g_mark_predecessor, g_with_value> CsrProblem;
  CsrProblem csr_problem(cfg);
  if (csr_problem.FromHostProblem(source_file_name, g_stream_from_host, csr_graph.nodes,
      csr_graph.edges, csr_graph.column_indices,
      csr_graph.row_offsets, csr_graph.edge_values, csr_graph.row_indices,
      csr_graph.column_offsets, csr_graph.node_values, num_gpus))
    exit(1);

  const bool INSTRUMENT = true;

  GASengine::EnactorVertexCentric<INSTRUMENT> vertex_centric(cfg, g_verbose);

  cudaError_t retval = cudaSuccess;

  retval = vertex_centric.EnactIterativeSearch<CsrProblem, bfs>(csr_problem, source_file_name,
      csr_graph.row_offsets);

  if (retval && (retval != cudaErrorInvalidDeviceFunction))
  {
    exit(1);
  }

  csr_problem.ExtractResults(h_dists, h_labels, h_sigmas, h_deltas);

  if (strcmp(source_file_name, "") == 0)
    correctTest(csr_graph.nodes, reference_labels, h_dists);

  if (outFileName)
  {
    FILE* f = fopen(outFileName, "w");
    for (int i = 0; i < csr_graph.nodes; ++i)
    {
      fprintf(f, "%d\n", h_dists[i]);
    }

    fclose(f);
  }

  return 0;
}
