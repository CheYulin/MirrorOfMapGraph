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

#ifndef SSSP_H_
#define SSSP_H_

#include <GASengine/csr_problem.cuh>

/* Single Source Shortest Path.
 */

//TODO: edge data not currently represented
//TODO: initialize frontier
struct sssp
{

	/*
	 * Note: You must modify the code here and in the Gather device function to
	 * switch between int32, int64, and float.  double could probably also be
	 * supported with a re-interpretation cast to unsigned long long (in664).
	 */
  // int32
  typedef int DataType;
  static const DataType INIT_VALUE = INT_MAX;
  // int64 (must also edit ../setup.mk to specify only cuda compute 3.5+)
//  typedef unsigned long long DataType;
//  static const DataType INIT_VALUE = ULLONG_MAX;
  // float
//  typedef float DataType;
//  static const DataType INIT_VALUE = FLT_MAX;

  typedef DataType MiscType;
  typedef DataType GatherType;
  typedef int VertexId;
  typedef int SizeT;

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
    DataType* d_weights;

    EdgeType() :
        d_weights(NULL), nodes(0), edges(0)
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

    DataType* h_init_dists = new DataType[nodes];

    for (int i = 0; i < nodes; i++)
    {
      h_init_dists[i] = INIT_VALUE;
    }
    for (int i = 0; i < num_srcs; i++)
    {
      h_init_dists[srcs[i]] = 0;
    }

    if (vertex_list.d_dists == NULL)
      b40c::util::B40CPerror(
          cudaMalloc((void**) &vertex_list.d_dists,
              nodes * sizeof(DataType)),
          "cudaMalloc VertexType::d_dists failed", __FILE__, __LINE__);

    if (vertex_list.d_dists_out == NULL)
      b40c::util::B40CPerror(
          cudaMalloc((void**) &vertex_list.d_dists_out,
              nodes * sizeof(DataType)),
          "cudaMalloc VertexType::d_dists_out failed", __FILE__,
          __LINE__);

    if (edge_list.d_weights == NULL)
      b40c::util::B40CPerror(
          cudaMalloc((void**) &edge_list.d_weights,
              edges * sizeof(DataType)),
          "cudaMalloc edge_list.d_weights failed", __FILE__,
          __LINE__);

    if (b40c::util::B40CPerror(
        cudaMemcpy(edge_list.d_weights, d_edge_values,
            edges * sizeof(DataType), cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy edge d_weights failed", __FILE__, __LINE__))
      exit(0);

    //Initialize
    if (b40c::util::B40CPerror(
        cudaMemcpy(vertex_list.d_dists, h_init_dists,
            nodes * sizeof(DataType), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy edge d_dists failed", __FILE__, __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(vertex_list.d_dists_out, vertex_list.d_dists,
            nodes * sizeof(DataType), cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy edge d_weights failed", __FILE__, __LINE__))
      exit(0);

    printf("Starting vertex: ");
    for (int i = 0; i < num_srcs; i++)
      printf("%d ", srcs[i]);
    printf("\n");

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[0], srcs, num_srcs * sizeof(int),
            cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys[0] failed", __FILE__,
        __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[1], d_frontier_keys[0], num_srcs * sizeof(int),
            cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys[1] failed", __FILE__,
        __LINE__))
      exit(0);

    int init_value[1] = { 0 };
    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_values[0], init_value,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
        __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_values[1], d_frontier_values[0],
            num_srcs * sizeof(int), cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
        __LINE__))
      exit(0);

    delete[] h_init_dists;
  }

  static SrcVertex srcVertex()
  {
    return SINGLE;
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
    void operator()(const VertexId vertex_id, const int iteration, GatherType gathervalue,
        VertexType& vertex_list, EdgeType& edge_list, char& changed)
    {

      const DataType oldvalue = vertex_list.d_dists[vertex_id];
      const DataType newvalue = min(oldvalue, gathervalue);

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
//      printf("Apply: vertex_id=%d, oldvalue=%d, gathervalue=%d, changed=%d\n", vertex_id, oldvalue, gathervalue, changed);
    }
  };

  /** post-apply function (invoked after threads in apply() synchronize at a memory barrier). */
  struct post_apply
  {
    __device__
    void operator()(const VertexId vertex_id, VertexType& vertex_list, EdgeType& edge_list, GatherType* gather_tmp)
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
    bool operator()(const VertexId vertex_id, const char changed, VertexType &vertex_list, EdgeType& edge_list)
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
        const VertexId vertex_id, const VertexId neighbor_id_in, const VertexId edge_id,
        VertexType& vertex_list, EdgeType& edge_list, int& frontier, MiscType& misc_value)
    {
      const int src_dist = vertex_list.d_dists[vertex_id];
      const int dst_dist = vertex_list.d_dists[neighbor_id_in];
      DataType edge_value = edge_list.d_weights[edge_id];

//      if ((changed || iteration == 0) && dst_dist > src_dist + edge_value)
      if (dst_dist > src_dist + edge_value)
        frontier = neighbor_id_in;
      else
        frontier = -1;
      misc_value = src_dist + edge_value; // source dist + edge weight
//      printf("vertex_id=%d, neighbor_id=%d, src_dist=%d, dst_dist=%d, edge_id = %d, edge_value=%d, frontier=%d, misc_value=%d\n", vertex_id, neighbor_id_in, src_dist, dst_dist, edge_id, edge_value, frontier, misc_value);
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
       * for SSSP.  This writes on gather_tmp[] to find
       * the minimum distinct for this vertex.
       */

//			  printf("vertex_id=%d, misc_value=%d\n", vertex_id, misc_value);
      // Works for int32
      // Works for int64 (CUDA compute 3.5+, but you must also edit the ../setup.mk to restrict the target architectures)
      GatherType old = atomicMin(&gather_tmp[vertex_id], misc_value);
      // Works for float.
      // GatherType old = atomicMin(reinterpret_cast<int *>( &gather_tmp[vertex_id] ), reinterpret_cast<int&>(misc_value));
//      printf("Contract: vertex_id=%d, old=%d, gather_tmp=%d, misc_value=%d\n", vertex_id, old, gather_tmp[vertex_id], misc_value);
    }
  };

  static void extractResult(VertexType& vertex_list, DataType* h_output)
  {
    cudaMemcpy(h_output, vertex_list.d_dists, sizeof(DataType) * vertex_list.nodes, cudaMemcpyDeviceToHost);
  }

};

#endif /* SSSP_H_ */
