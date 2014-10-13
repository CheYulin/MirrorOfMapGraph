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

#ifndef CC_H_
#define CC_H_

#include <GASengine/csr_problem.cuh>

/* Single Source Shortest Path.
 */

//TODO: edge data not currently represented
struct cc
{

  typedef int DataType;
  typedef DataType MiscType;
  typedef DataType GatherType;
  typedef int VertexId;
  typedef int SizeT;

  static const int INIT_VALUE = INT_MAX;//100000000;
  static const bool allow_duplicates = true;

  struct VertexType
  {
    int nodes; // #of nodes.
    int edges; // #of edges.
    DataType* d_dists_out; // new value computed by apply.
    DataType* d_dists; // the actual distance computed by post_apply

    VertexType() :
        d_dists(NULL), d_dists_out(NULL), nodes(0), edges(0)
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

  static SrcVertex srcVertex()
  {
    return ALL;
  }

  static GatherEdges gatherOverEdges()
  {
    return NO_GATHER_EDGES;
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

  static void Initialize(const int directed, const int nodes, const int edges, int num_srcs,
      int* srcs, int* d_row_offsets, int* d_column_indices, int* d_column_offsets, int* d_row_indices, int* d_edge_values,
      VertexType &vertex_list, EdgeType &edge_list, int* d_frontier_keys[3],
      MiscType* d_frontier_values[3])
  {
    vertex_list.nodes = nodes;
    vertex_list.edges = edges;

    b40c::util::B40CPerror(
        cudaMalloc((void**) &vertex_list.d_dists,
            nodes * sizeof(DataType)),
        "cudaMalloc VertexType::d_dists failed", __FILE__, __LINE__);

    b40c::util::B40CPerror(
        cudaMalloc((void**) &vertex_list.d_dists_out,
            nodes * sizeof(DataType)),
        "cudaMalloc VertexType::d_dists_out failed", __FILE__,
        __LINE__);

    int memset_block_size = 256;
    int memset_grid_size_max = 32 * 1024;	// 32K CTAs
    int memset_grid_size;

    memset_grid_size = B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);

    b40c::util::SequenceKernel<DataType><<<memset_grid_size,
    memset_block_size, 0, 0>>>(vertex_list.d_dists,
        nodes);

    // Initialize d_dists_out elements
    cudaMemcpy(vertex_list.d_dists_out,
        vertex_list.d_dists,
        nodes * sizeof(DataType),
        cudaMemcpyDeviceToDevice);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[0], vertex_list.d_dists, nodes * sizeof(int),
            cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__,
        __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[1], vertex_list.d_dists, nodes * sizeof(int),
            cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__,
        __LINE__))
      exit(0);

    b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0,
    0>>>(d_frontier_values[0], INIT_VALUE, nodes);

    b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0,
    0>>>(d_frontier_values[1], INIT_VALUE, nodes);

  }

  /**
   * For each vertex in the frontier,
   */
  struct gather_vertex
  {
    __device__
    void operator()(const int vertex_id, const GatherType final_value,
        VertexType &vertex_list, EdgeType &edge_list)
    {

    }
  };

  /**
   * For each vertex in the frontier,
   */
  struct gather_edge
  {
    __device__
    void operator()(const int vertex_id, const int edge_id, const int neighbor_id_in,
        VertexType &vertex_list, EdgeType &edge_list, GatherType& new_value)
    {

    }
  };

  /**
   * the binary operator
   */
  struct gather_sum
  {
    __device__ GatherType operator()(const GatherType &left,
        const GatherType &right)
    {
      return 0;
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

      const int oldvalue = vertex_list.d_dists[vertex_id];
      const int newvalue = min(oldvalue, gathervalue);

      if (iteration == 0)
      {
        changed = 1;
      }
      else
      {
        if (oldvalue == newvalue)
        changed = 0;
        else
        changed = 1;
      }
      vertex_list.d_dists_out[vertex_id] = newvalue;
//      printf("Apply: oldvalue=%d, newvalue=%d, gathervalue=%d\n", oldvalue, newvalue, gathervalue);
    }
  };

  /** post-apply function (invoked after threads in apply() synchronize at a memory barrier). */
  struct post_apply
  {
    __device__
    void operator()(const int vertex_id, VertexType& vertex_list, EdgeType& edge_list, GatherType* gather_tmp)
    {
      vertex_list.d_dists[vertex_id] = vertex_list.d_dists_out[vertex_id];
      gather_tmp[vertex_id] = INIT_VALUE;
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
      return changed == 1;
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
        VertexType& vertex_list, EdgeType& edge_list, int& frontier, int& misc_value)
    {
      const int src_dist = vertex_list.d_dists[vertex_id];
      const int dst_dist = vertex_list.d_dists[neighbor_id_in];
      if ((changed || iteration == 0) && dst_dist > src_dist)
        frontier = neighbor_id_in;
      else
        frontier = -1;
      misc_value = src_dist;
//      printf("vertex_id=%d, neighbor_id=%d, src_dist=%d, dst_dist=%d, edge_id = %d, frontier=%d, misc_value=%d\n", vertex_id, neighbor_id_in, src_dist, dst_dist, edge_id, frontier, misc_value);
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
        VertexType &vertex_list, EdgeType &edge_list, GatherType* gather_tmp, int& misc_value)
    {

      /**
       * Note: predecessor is source dist + edge weight
       * for SSSP.  This writes on d_min_dists[] to find
       * the minimum distinct for this vertex.
       */

//			  printf("vertex_id=%d, misc_value=%d\n", vertex_id, misc_value);
      atomicMin(&gather_tmp[vertex_id], misc_value);

    }
  };

  static void extractResult(VertexType& vertex_list, DataType* h_output)
  {
    cudaMemcpy(h_output, vertex_list.d_dists, sizeof(DataType) * vertex_list.nodes, cudaMemcpyDeviceToHost);
  }

};

#endif /* CC_H_ */
