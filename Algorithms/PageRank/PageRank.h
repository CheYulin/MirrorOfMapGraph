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

#ifndef PR_H_
#define PR_H_

#include <GASengine/csr_problem.cuh>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

struct pagerank
{

  typedef float DataType;
  typedef DataType MiscType;
  typedef float GatherType;
  typedef int VertexId;
  typedef int SizeT;

  static const DataType INIT_VALUE = 0.0;
  static const bool allow_duplicates = false;

  struct VertexType
  {
    int nodes; // #of nodes.
    int edges; // #of edges.
//    int* d_changed; // 1 iff dists_out was changed in apply.
    DataType* d_dists; // the actual distance computed by post_apply
    int* d_num_out_edge;
    //    int* d_visited_flag;

    VertexType() :
        d_dists(NULL),
            d_num_out_edge(NULL),
            nodes(0), edges(0)
    {
    }
  };

  struct EdgeType
  {
    int nodes; // #of nodes.
    int edges; // #of edges.

    EdgeType() :
        nodes(0), edges(0)
    {
    }
  };

  static void Initialize(const int directed, const int nodes, const int edges, int num_srcs,
      int* srcs, int* d_row_offsets, int* d_column_indices, int* d_column_offsets, int* d_row_indices, DataType* d_edge_values,
      VertexType &vertex_list, EdgeType &edge_list, int* d_frontier_keys[3],
      MiscType* d_frontier_values[3])
  {
    vertex_list.nodes = nodes;
    vertex_list.edges = edges;

    b40c::util::B40CPerror(
        cudaMalloc((void**) &vertex_list.d_dists,
            nodes * sizeof(DataType)),
        "cudaMalloc VertexType::d_dists failed", __FILE__, __LINE__);

//    b40c::util::B40CPerror(
//        cudaMalloc((void**) &vertex_list.d_dists_out,
//            nodes * sizeof(DataType)),
//        "cudaMalloc VertexType::d_dists_out failed", __FILE__,
//        __LINE__);

//    b40c::util::B40CPerror(
//        cudaMalloc((void**) &vertex_list.d_changed,
//            nodes * sizeof(int)),
//        "cudaMalloc VertexType::d_changed failed", __FILE__, __LINE__);

//    b40c::util::B40CPerror(
//        cudaMalloc((void**) &vertex_list.d_min_dists,
//            nodes * sizeof(DataType)),
//        "cudaMalloc VertexType::d_min_dists failed", __FILE__,
//        __LINE__);

//    b40c::util::B40CPerror(
//        cudaMalloc((void**) &vertex_list.d_visited_flag,
//            nodes * sizeof(int)),
//        "cudaMalloc VertexType::d_visited_flag failed", __FILE__, __LINE__);

    int memset_block_size = 256;
    int memset_grid_size_max = 32 * 1024;   // 32K CTAs
    int memset_grid_size;

    // Initialize d_dists elements to 100000000
    memset_grid_size = B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);

    b40c::util::MemsetKernel<DataType><<<memset_grid_size, memset_block_size, 0,
    0>>>(vertex_list.d_dists, 0.15, nodes);

//    // Initialize d_dists_out elements
//    cudaMemcpy(vertex_list.d_dists_out,
//        vertex_list.d_dists,
//        nodes * sizeof(DataType),
//        cudaMemcpyDeviceToDevice);

//    b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0,
//        0>>>(vertex_list.d_changed, 0, nodes);

//    b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0,
//        0>>>(vertex_list.d_visited_flag, 0, nodes);

    b40c::util::SequenceKernel<int><<<memset_grid_size,
    memset_block_size, 0, 0>>>(d_frontier_keys[0],
        nodes);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[1], d_frontier_keys[0], nodes * sizeof(int),
            cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__,
        __LINE__))
      exit(0);

    //compute d_num_out_edges
    b40c::util::B40CPerror(
        cudaMalloc((void**) &vertex_list.d_num_out_edge,
            (nodes + 1) * sizeof(int)),
        "cudaMalloc d_num_out_edges failed", __FILE__, __LINE__);

    thrust::device_ptr<int> d_row_offsets_ptr(d_row_offsets);
    thrust::device_ptr<int> d_num_out_edge_ptr(vertex_list.d_num_out_edge);

    thrust::device_vector<int> d_num_out_edge_tmp(nodes + 1);

    thrust::adjacent_difference(thrust::device, d_row_offsets_ptr, d_row_offsets_ptr + nodes + 1, d_num_out_edge_ptr);
    vertex_list.d_num_out_edge++;

    if (directed == 0)
    {
      thrust::device_ptr<int> d_column_offsets_ptr(d_column_offsets);
      thrust::adjacent_difference(thrust::device, d_column_offsets_ptr, d_column_offsets_ptr + nodes + 1, d_num_out_edge_tmp.begin());

      thrust::transform(d_num_out_edge_tmp.begin() + 1, d_num_out_edge_tmp.end(), d_num_out_edge_ptr + 1, d_num_out_edge_ptr + 1, thrust::plus<int>());
    }

//    int max_degree = thrust::reduce(d_num_out_edge_ptr + 1, d_num_out_edge_ptr + nodes + 1, -1, thrust::maximum<int>());
//    int min_degree = thrust::reduce(d_num_out_edge_ptr + 1, d_num_out_edge_ptr + nodes + 1, 100000000, thrust::minimum<int>());
//    printf("Max degree is: %d\n", max_degree);
//    printf("Min degree is: %d\n", min_degree);


//    int *tmp = new int[nodes];
//    cudaMemcpy(tmp, vertex_list.d_num_out_edge, nodes * sizeof(int),
//        cudaMemcpyDeviceToHost);
//    printf("d_num_out_edge: ");
//    for (int i = 0; i < nodes; i++)
//    {
//      printf("%d, ", tmp[i]);
//    }
//    printf("\n");
//    delete[] tmp;
//
//    tmp = new int[nodes + 1];
//    cudaMemcpy(tmp, d_column_offsets, (nodes + 1) * sizeof(int),
//        cudaMemcpyDeviceToHost);
//    printf("d_column_offsets: ");
//    for (int i = 0; i < nodes + 1; i++)
//    {
//      printf("%d, ", tmp[i]);
//    }
//    printf("\n");
//    delete[] tmp;
  }

  static SrcVertex srcVertex()
  {
    return ALL;
  }

  static GatherEdges gatherOverEdges()
  {
    return GATHER_IN_EDGES;
  }

  static ApplyVertices applyOverEdges()
  {
    return APPLY_FRONTIER;
  }

  static PostApplyVertices postApplyOverEdges()
  {
    return POST_APPLY_FRONTIER;
  }

  static ExpandEdges expandOverEdges()
  {
    return EXPAND_OUT_EDGES;
  }

//  /**
//   * For each vertex in the frontier,
//   */
//  struct gather_vertex
//  {
//    __device__
//    void operator()(const int vertex_id, const GatherType final_value,
//            VertexType &vertex_list, EdgeType &edge_list)
//    {
////      vertex_list.d_min_dists[vertex_id] += final_value;
//    }
//  };

  /**
   * For each vertex in the frontier,
   */
  struct gather_edge
  {
    __device__
    void operator()(const int vertex_id, const int edge_id, const int neighbor_id_in,
        VertexType &vertex_list, EdgeType &edge_list, GatherType& new_value)
    {
      DataType nb_dist = vertex_list.d_dists[neighbor_id_in];
      new_value = nb_dist / (DataType) vertex_list.d_num_out_edge[neighbor_id_in];
//      printf("vertex_id=%d, d_num_out_edge[%d]=%d\n", vertex_id, neighbor_id_in, vertex_list.d_num_out_edge[neighbor_id_in]);
    }
  };

  /**
   * the binary operator
   */
  struct gather_sum
  {
    __device__ GatherType operator()(const GatherType &left, const GatherType &right)
    {
      return left + right;
    }
  };

  /** Update the vertex state given the gather results. */
  struct apply
  {
    __device__
    /**
     *
     */
    void operator()(const int vertex_id, const int iteration, GatherType gathervalue,
        VertexType& vertex_list, EdgeType& edge_list, char& changed)
    {

      const DataType oldvalue = vertex_list.d_dists[vertex_id];
//      const DataType gathervalue = vertex_list.d_min_dists[active_vertex_id];
      const DataType newvalue = 0.15f + (1.0f - 0.15f) * gathervalue;

      if (fabs(oldvalue - newvalue) < 0.01f)
        changed = 0;
      else
      {
//        if(vertex_id < 200) printf("(%d %.3f) ", vertex_id, newvalue);
        changed = 1;
      }
//      if(vertex_id < 100)
//      printf("vertex_id=%d, oldvalue=%f, gathervalue=%f, newvalue=%f, changed=%d\n", vertex_id, oldvalue, gathervalue, newvalue, changed);

      vertex_list.d_dists[vertex_id] = newvalue;
    }
  };

  /** post-apply function (invoked after threads in apply() synchronize at a memory barrier). */
  struct post_apply
  {
    __device__
    void operator()(const int vertex_id, VertexType& vertex_list, EdgeType& edge_list, GatherType* gather_tmp)
    {
//      gather_tmp[vertex_id] = 0.0;
    }
  };

  /** The return value of this device function will be passed into the
   expand_edge device function as [bool:change]. For example, this
   can check the state of the vertex to decide whether it has been
   updated and only expand its neighbors if it has be updated. */
  struct expand_vertex
  {
    __device__
    /**
     * @param vertex_id The vertex identifier of the source
     * vertex.
     *
     * @param vertex_list The vertices in the graph.
     */
    bool operator()(const int vertex_id, const char changed, VertexType &vertex_list, EdgeType& edge_list)
    {
      return changed;
    }
  };

  /** Expand stage creates a new frontier. The frontier can have a lot
   of duplicates.  The contract stage will eliminate (some of)
   those duplicates.  There are two outputs for expand.  One is the
   new frontier.  The other is a "predecessor" array.  These arrays
   have a 1:1 correspondence.  The predecessor array is available
   for user data, depending on the algorithm.  For example, for BFS
   it is used to store the vertex_id of the vertex from which this
   vertex was reached by a one step traversal along some edge.

   TODO: add edge_list and edge_id

   TODO: Potentially make the predecessor[] into a used-defined
   type, but this will change the shared memory size calculations.
   */
  struct expand_edge
  {
    __device__
    /**
     * @param changed true iff the device function
     * expand_vertex evaluated to true for this vertex.
     *
     * @param row_id The vertex identifier of the source
     * vertex.
     *
     * @param vertex_list The vertices in the graph.
     *
     * @param neighbor_id_in The vertex identifier of
     * the target vertex.
     *
     * @param neighbor_id_out DEPRECATED.
     *
     * @param frontier The vertex identifier to be added
     * into the new frontier. Set to neighbor_id_in if
     * you want to visit that vertex and set to -1 if
     * you do not want to visit that vertex along this
     * edge.
     *
     * @param precedessor_out The optional value to be
     * written into the predecessor array. This array
     * has a 1:1 correspondence with the frontier array.
     */
    void operator()(const bool changed, const int iteration,
        const int vertex_id, const int neighbor_id_in, const int edge_id,
        VertexType& vertex_list, EdgeType& edge_list, int& frontier, MiscType& misc_value)
    {
//      printf("expand: vertex_id=%d, neighbor_id_in=%d\n", vertex_id, neighbor_id_in);
      if (changed)
        frontier = neighbor_id_in;
    }
  };

  /** The contract stage is used to reduce the duplicates in the
   frontier created by the Expand stage.

   TODO: Replace iteration with a struct for some engine state.
   Pass this into more methods.
   */
  struct contract
  {
    __device__
    /**
     * @param row_id The vertex identifier of the source
     * vertex.
     *
     * @param vertex_list The vertices in the graph.
     *
     * @param iterator The iteration number.
     *
     * @param vertex_id If you do not want to visit this
     * vertex, then write a -1 on this parameter.
     *
     * @param predecessor The value from the
     * predecessor_out array in the expand_edge device
     * function.
     */
    void operator()(const int iteration, int &vertex_id,
        VertexType &vertex_list, EdgeType &edge_list, GatherType* gather_tmp, MiscType& misc_value)
    {

      /**
       * Note: predecessor is source dist + edge weight
       * for SSSP.  This writes on d_min_dists[] to find
       * the minimum distinct for this vertex.
       */

//			  printf("vertex_id=%d, misc_value=%d\n", vertex_id, misc_value);
//      atomicMin(&vertex_list.d_min_dists[vertex_id], misc_value);
    }
  };

  static void extractResult(VertexType& vertex_list, DataType* h_output)
  {
    cudaMemcpy(h_output, vertex_list.d_dists, sizeof(DataType) * vertex_list.nodes, cudaMemcpyDeviceToHost);
  }

};

#endif /* SSSP_H_ */
