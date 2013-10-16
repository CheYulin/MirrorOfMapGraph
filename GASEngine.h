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

#ifndef GASENGINE_H__
#define GASENGINE_H__

#define INDEBUG

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <omp.h>

// Utilities and correctness-checking
#include <test/b40c_test_util.h>

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

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool g_verbose;
bool g_verbose2;
const bool g_undirected = true; // Whether to add a "backedge" for every edge parsed/generated
bool g_quick; // Whether or not to perform CPU traversal as reference
const bool g_stream_from_host = false; // Whether or not to stream CSR representation from host mem
const bool g_with_value = true; // Whether or not to load value

class GASEngine
{
public:

  std::vector<int> run(CsrGraph<int, EdgeType, int>& csr_graph)
  {
    thrust::device_vector<int> d_dummy_edge_vals;

    return run(csr_graph);
  }

  void run(GASengine::CsrProblem<int, int, float, false, false>& csr_problem, int src)
  {
    const bool INSTRUMENT = true;

    int max_grid_size = 0; // Maximum grid size (0: leave it up to the enactor)
    double max_queue_sizing = 6240.5;
    std::vector<int> strategies(1, VERTEX_CENTRIC);

    // Allocate BFS enactor map
    GASengine::EnactorVertexCentric<INSTRUMENT> vertex_centric(g_verbose);

    cudaError_t retval = cudaSuccess;

    retval = vertex_centric.EnactIterativeSearch(csr_problem, src, max_grid_size, max_queue_sizing);

    if (retval && (retval != cudaErrorInvalidDeviceFunction))
    {
      exit(1);
    }
  }
};

#endif
