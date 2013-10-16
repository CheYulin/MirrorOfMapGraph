/*********************************************************

 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.

 **********************************************************/

/* Written by Erich Elsen and Vishal Vaidyanathan
 of Royal Caliber, LLC
 Contact us at: info@royal-caliber.com
 */

typedef unsigned int uint;

//#include <thrust/random/linear_congruential_engine.h>
//#include <thrust/random/normal_distribution.h>
//#include <thrust/random/uniform_int_distribution.h>
#include "graphio.h"
#include <stdio.h> 
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <vector>

//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/transform.h>
//#include <thrust/iterator/transform_iterator.h>
//#include <thrust/iterator/permutation_iterator.h>
//#include <thrust/iterator/discard_iterator.h>
//#include <thrust/sort.h>
//#include <thrust/reduce.h>
//#include <thrust/copy.h>
//#include <thrust/execution_policy.h>
#include <iostream>
#include <omp.h>

// Utilities and correctness-checking
//#include <test/b40c_test_util.h>

// Graph construction utils

#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/random.cuh>

// BFS includes
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/bfs/enactor_contract_expand.cuh>
#include <b40c/graph/bfs/enactor_expand_contract.cuh>
#include <b40c/graph/bfs/enactor_two_phase.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>
#include <b40c/graph/bfs/enactor_multi_gpu.cuh>

#include <b40c/graph/GASengine/csr_problem.cuh>
#include <b40c/graph/GASengine/enactor_vertex_centric.cuh>

using namespace b40c;
using namespace graph;
using namespace std;

enum Strategy
{
  HOST = -1, EXPAND_CONTRACT, CONTRACT_EXPAND, TWO_PHASE, HYBRID, MULTI_GPU, VERTEX_CENTRIC
};

bool g_verbose;
bool g_verbose2;
bool g_undirected; // Whether to add a "backedge" for every edge parsed/generated
bool g_quick; // Whether or not to perform CPU traversal as reference
bool g_stream_from_host; // Whether or not to stream CSR representation from host mem
bool g_with_value; // Whether or not to load value

void cudaInit(int device)
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) error_id, cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  }
  else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {
    if (dev == device) {
      cudaSetDevice(dev);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);


      printf("Running on this device:");
      printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

      // Console log
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
      printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

      printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
             (float) deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
    }
  }
}

int main(int argc, char **argv)
{

  const char* outFileName = 0;
  int src = 0;
  bool g_undirected;
  const bool g_stream_from_host = false;
  const bool g_with_value = true;
  const bool g_mark_predecessor = false;
  bool g_verbose = false;
  typedef int VertexId; // Use as the node identifier type
  typedef int Value; // Use as the value type
  typedef int SizeT; // Use as the graph size type
  CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);
  char* source_file_name = "";
  int device = 0;
  double max_queue_sizing = 1.3;

  if (argc == 7) {
    //    ispattern = loadGraph(argv[1], numVertices, h_edge_src_vertex, h_edge_dst_vertex, &h_edge_data);
    outFileName = argv[2];
    src = atoi(argv[3]) - 1;
    g_undirected = atoi(argv[4]);
    device = atoi(argv[5]);
    max_queue_sizing = atof(argv[6]);
    //    source_file_name = argv[6];


    char *market_filename = argv[1];
    if (builder::BuildMarketGraph<g_with_value>(market_filename, csr_graph, g_undirected) != 0) {
      return 1;
    }
    //    csr_graph.DisplayGraph();
  }
  else {
    std::cerr << "Usage: ./simpleSSSP market_graph_file output_file src undirected (0 or 1) device max_queue_sizing" << std::endl;
    exit(1);
  }

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);

  printf("Running on host: %s\n", hostname);

  cudaInit(device);
  
  printf("max_queue_sizing = %f\n", max_queue_sizing );

  VertexId* h_labels = (VertexId*) malloc(sizeof (VertexId) * csr_graph.nodes);
  int* h_dists = (VertexId*) malloc(sizeof (VertexId) * csr_graph.nodes);
  //    VertexId* reference_check = (g_quick) ? NULL : reference_labels;
  //
  //    //Allocate host-side node_value array (both ref and gpu-computed results)
  //    Value* ref_node_values = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
  Value* h_node_values = (Value*) malloc(sizeof (Value) * csr_graph.nodes);
  //    Value* ref_node_value_check = (g_quick) ? NULL : ref_node_values;
  //
  //    //Allocate host-side sigma value array (both ref and gpu-computed results)
  //    Value* ref_sigmas = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
  Value* h_sigmas = (Value*) malloc(sizeof (Value) * csr_graph.nodes);
  //    Value* ref_sigmas_check = (g_quick) ? NULL : ref_sigmas;
  Value* h_deltas = (Value*) malloc(sizeof (Value) * csr_graph.nodes);

  // Allocate problem on GPU
  int num_gpus = 1;
  typedef GASengine::CsrProblem<VertexId, SizeT, Value, g_mark_predecessor, g_with_value> CsrProblem;
  CsrProblem csr_problem;
  if (csr_problem.FromHostProblem(g_stream_from_host, csr_graph.nodes, csr_graph.edges, csr_graph.column_indices, csr_graph.row_offsets, csr_graph.edge_values, csr_graph.row_indices,
                                  csr_graph.column_offsets, csr_graph.node_values, num_gpus, device)) exit(1);

  const bool INSTRUMENT = true;

  int max_grid_size = 0; // Maximum grid size (0: leave it up to the enactor)

  std::vector<int> strategies(1, VERTEX_CENTRIC);

  // Allocate BFS enactor map
  GASengine::EnactorVertexCentric<INSTRUMENT> vertex_centric(g_verbose);

  cudaError_t retval = cudaSuccess;

  retval = vertex_centric.EnactIterativeSearch(csr_problem, src, source_file_name, csr_graph.row_offsets, max_grid_size, max_queue_sizing);

  if (retval && (retval != cudaErrorInvalidDeviceFunction)) {
    exit(1);
  }

  //  // Allocate problem on GPU
  //  int num_gpus = 1;
  //  typedef GASengine::CsrProblem<int, int, int, false, false> CsrProblem;
  //  CsrProblem csr_problem;
  //  if (csr_problem.FromHostProblem(false, csr_graph.nodes, csr_graph.edges, csr_graph.column_indices, csr_graph.row_offsets, csr_graph.edge_values, csr_graph.row_indices, csr_graph.column_offsets,
  //      csr_graph.node_values, num_gpus)) exit(1);
  //
  //  std::vector<int> ret(2);
  //  GASEngine<sssp, sssp::VertexType, int, int, int> engine;
  //
  //  cudaEvent_t start, stop;
  //  cudaEventCreate(&start);
  //  cudaEventCreate(&stop);
  //
  //  cudaEventRecord(start);
  //
  ////  ret = engine.run(csr_problem, d_edge_dst_vertex, d_edge_src_vertex, d_vertex_data, d_edge_data, d_active_vertex_flags, INT_MAX);
  //  ret = engine.run(csr_problem);
  //
  //  cudaEventRecord(stop);
  //  cudaEventSynchronize(stop);
  //
  //  int diameter = ret[0];
  //  float elapsed;
  //  cudaEventElapsedTime(&elapsed, start, stop);
  //  std::cout << "Took: " << elapsed << " ms" << std::endl;
  //  std::cout << "Iterations to convergence: " << diameter << std::endl;

  csr_problem.ExtractResults(h_dists, h_labels, h_sigmas, h_deltas);
  
  if (outFileName) {
    FILE* f = fopen(outFileName, "w");
    for (int i = 0; i < csr_graph.nodes; ++i) {
      fprintf(f, "%d\n", h_labels[i]);
    }

    fclose(f);
  }

  return 0;
}
