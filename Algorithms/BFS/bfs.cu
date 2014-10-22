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
#include <bfs.h>
#include <iostream>
#include <time.h>

#include <config.h>


// Utilities and correctness-checking
//#include <test/b40c_test_util.h>

// Graph construction utils

#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/random.cuh>
#include <b40c/graph/builder/rmat.cuh>

#include <GASengine/csr_problem.cuh>
#include <GASengine/enactor_vertex_centric.cuh>
#include <predextract.cuh>

using namespace b40c;
using namespace graph;
using namespace std;

template<
typename VertexId,
typename Value,
typename SizeT>
void CPUBFS(
            int directed,
            int test_iteration,
            const CsrGraph<VertexId, Value, SizeT> &csr_graph,
            VertexId *source_path,
            VertexId src)
{
  printf("Running CPU code...");
  fflush(stdout);

  // (Re)initialize distances
  for (VertexId i = 0; i < csr_graph.nodes; i++)
  {
    source_path[i] = -1;
  }
  source_path[src] = 0;
  VertexId search_depth = 0;

  const time_t startTime = time(NULL);

  // Initialize queue for managing previously-discovered nodes
  std::deque<VertexId> frontier;
  frontier.push_back(src);

  //
  // Perform BFS on CPU
  //

  int iter = 0;
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

    if (directed == 0)
    {
      edges_begin = csr_graph.column_offsets[dequeued_node];
      edges_end = csr_graph.column_offsets[dequeued_node + 1];
      for (int edge = edges_begin; edge < edges_end; edge++)
      {

        // Lookup neighbor and enqueue if undiscovered
        VertexId neighbor = csr_graph.row_indices[edge];
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
    iter++;
  }

  const time_t EndTime = time(NULL);

  std::cout << "CPU time took: " << difftime(EndTime, startTime) * 1000 << " ms"
          << std::endl;
  search_depth++;
}

bool cudaInit(int device)
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess)
  {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
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
             (float)deviceProp.totalGlobalMem / 1048576.0f,
             (unsigned long long)deviceProp.totalGlobalMem);

      break;
    }
  }

  return true;
}

bool correctTest(int nodes, int* reference_labels, int* h_labels)
{
  bool pass = true;
  printf("Correctness testing ...");
  fflush(stdout);
  for (int i = 0; i < nodes; i++)
  {
    if (reference_labels[i] != h_labels[i])
    {
      //      printf("Incorrect value for node %d: CPU value %d, GPU value %d\n", i, reference_labels[i], h_labels[i]);
      pass = false;
    }
  }
  if (pass)
    printf("passed\n");
  else
    printf("failed\n");
  return pass;
}

void printUsageAndExit(char *algo_name)
{
  std::cout
          << "Usage: " << algo_name
          << " [-graph (-g) graph_file] [-output (-o) output_file] [-sources src_file] [-BFS \"variable1=value1 variable2=value2 ... variable3=value3\" -help ] [-c config_file]\n";
  std::cout << "     -help display the command options\n";
  std::cout << "     -graph specify a sparse matrix in Matrix Market (.mtx) format\n";
  std::cout << "     -output or -o specify file for output result\n";
  std::cout << "     -sources or -s set starting vertices file\n";
  std::cout << "     -c set the BFS options from the configuration file\n";
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
    }

    else if (strncmp(argv[i], "-parameters", 100) == 0 || strncmp(argv[i], "-p", 100) == 0)
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
    printUsageAndExit(argv[0]);
    exit(1);
  }

  int device_id = cfg.getParameter<int>("device");
  bool cudaEnabled = cudaInit(device_id);
  if (!cudaEnabled)
    return 0;

  char hostname[1024] = "localhost";
#ifdef gethostname
  gethostname(hostname, 1023);
#endif

  printf("Running on host: %s\n", hostname);

  int directed = cfg.getParameter<int>("directed");

  /*
    if(strcmp(graph_file, "RMAT") == 0)
    {
      int tmpn = pow(2,21);
      int tmpe = tmpn * 16;
      if(builder::BuildRmatGraph<g_with_value>(tmpn, tmpe, csr_graph, true, 0.45, 0.15, 0.15) != 0)
        exit(1);
    }
    else*/
  {
    if (builder::BuildMarketGraph<g_with_value > (graph_file, csr_graph,
                                                  false) != 0)
      exit(1);
  }
  //  csr_graph.DisplayGraph();
  int num_srcs = 0;
  int* srcs = NULL;
  int origin = cfg.getParameter<int>("origin");
  int iter_num = cfg.getParameter<int>("iter_num");
  int threshold = cfg.getParameter<int>("threshold");

  const int max_src_num = 1000;

  if (strcmp(source_file_name, ""))
  {
    if (strcmp(source_file_name, "RANDOM") == 0)
    {
      printf("Using random starting vertices!\n");
      num_srcs = cfg.getParameter<int>("num_src");
      srcs = new int[num_srcs];
      printf("Using %d random starting vertices!\n", num_srcs);
      srand(time(NULL));
      int
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

  {
    const int stats = cfg.getParameter<int>("stats");
    if (stats)
    {
      csr_graph.PrintHistogram();
    }
  }

  VertexId* reference_labels;

  int run_CPU = cfg.getParameter<int>("run_CPU");
  if (strcmp(source_file_name, "") == 0 && run_CPU) //Do correctness test only with single starting vertex
  {
    reference_labels = (VertexId*)malloc(sizeof (VertexId) * csr_graph.nodes);
    int test_iteration = 1;
    int src = cfg.getParameter<int>("src");

    if (origin == 1)
      src--;

    CPUBFS(
           directed,
           test_iteration,
           csr_graph,
           reference_labels,
           src);
  }

  // Allocate problem on GPU
  int num_gpus = 1;
  typedef GASengine::CsrProblem<bfs, VertexId, SizeT, Value, g_mark_predecessor, g_with_value> CsrProblem;
  CsrProblem csr_problem(cfg);

  if (csr_problem.FromHostProblem(g_stream_from_host, csr_graph.nodes,
                                  csr_graph.edges, csr_graph.column_indices,
                                  csr_graph.row_offsets, csr_graph.edge_values, csr_graph.row_indices,
                                  csr_graph.column_offsets, num_gpus, directed))
    exit(1);

  const bool INSTRUMENT = true;

  GASengine::EnactorVertexCentric<CsrProblem, bfs, INSTRUMENT> vertex_centric(cfg, g_verbose);

  for (int i = 0; i < num_srcs; i++)
  {
    int tmpsrcs[1];
    tmpsrcs[0] = srcs[i];
    printf("num_srcs=%d, src=%d, iter_num=%d\n", num_srcs, tmpsrcs[i], iter_num);

    cudaError_t retval = cudaSuccess;

    retval = vertex_centric.EnactIterativeSearch(csr_problem, csr_graph.row_offsets, directed, 1, tmpsrcs, iter_num, threshold);

    if (retval && (retval != cudaErrorInvalidDeviceFunction))
    {
      exit(1);
    }
  }

  int* d_predlist;
  cudaMalloc((void**)&d_predlist, csr_graph.nodes * sizeof (int));
  
  //extract the predecessor list: the predecessor computed for the src vertex is arbitrary and should be ignored. 
//  predextract<bfs, VertexId, SizeT, Value, g_mark_predecessor, g_with_value > (csr_problem, device_id, d_predlist);
  pred_extract(csr_problem.graph_slices[0]->nodes, 
              csr_problem.graph_slices[0]->d_column_offsets, 
              csr_problem.graph_slices[0]->d_row_indices, 
              csr_problem.graph_slices[0]->vertex_list.d_labels,
              device_id, 
              d_predlist);

//  int* test_vid = new int[csr_graph.nodes];
//  cudaMemcpy(test_vid, d_predlist, csr_graph.nodes * sizeof (int), cudaMemcpyDeviceToHost);
//  printf("d_predlist: ");
//  for (int i = 0; i < (csr_graph.nodes); ++i)
//  {
//    printf("%d, ", test_vid[i]);
//  }
//  printf("\n");
//  delete[] test_vid;

  Value* h_values = (Value*)malloc(sizeof (Value) * csr_graph.nodes);
  csr_problem.ExtractResults(h_values);

  //compute GTEPS
  long long te = 0;
  for (int i = 0; i < csr_graph.nodes; i++)
  {
    if (h_values[i] != -1)
    {
      te += csr_graph.row_offsets[i + 1] - csr_graph.row_offsets[i];
    }
  }

  printf("Traversed edge: %lld\n", te);

  if (strcmp(source_file_name, "") == 0 && run_CPU)
  {
    const bool ok = correctTest(csr_graph.nodes, reference_labels, h_values);
    free(reference_labels);
    if (!ok)
    {
      fprintf(stderr, "correctness test failed.");
      exit(1);
    }
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
