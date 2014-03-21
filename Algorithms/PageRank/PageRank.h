/*
 * pagerank.h
 *
 *  Created on: Dec 9, 2013
 *      Author: zhisong
 */

#ifndef PR_H_
#define PR_H_

#include <GASengine/csr_problem.cuh>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

struct pagerank
{

  typedef float DataType;
  typedef int MiscType;
  typedef float GatherType;
  typedef int VertexId;
  typedef int SizeT;

  static const DataType INIT_VALUE = 0.0;

  struct VertexType
  {
    int nodes; // #of nodes.
    int edges; // #of edges.
//    DataType* d_dists_out; // new value computed by apply.
    int* d_changed; // 1 iff dists_out was changed in apply.
    DataType* d_dists; // the actual distance computed by post_apply
//    DataType* d_dists_out;
    DataType* d_min_dists; // intermediate value for global synchronization computed in contract. used by the next apply()
    int* d_num_out_edge;
    int* d_visited_flag;

    VertexType() :
        d_dists(NULL),
            //        d_dists_out(NULL),
            d_changed(NULL), d_num_out_edge(NULL), d_visited_flag(NULL), d_min_dists(
                NULL), nodes(0), edges(0)
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

  static void freeall(VertexType &vertex_list, EdgeType &edge_list)
  {
    cudaFree(vertex_list.d_dists);
    cudaFree(vertex_list.d_changed);
    cudaFree(vertex_list.d_min_dists);
  }

  static void Initialize(const int nodes, const int edges, int num_srcs,
      int* srcs, int* d_row_offsets, int* d_column_indices,
      int* d_column_offsets, int* d_row_indices, DataType* d_edge_values,
      VertexType &vertex_list, EdgeType &edge_list,
      int* d_frontier_keys[3], MiscType* d_frontier_values[3])
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

    b40c::util::B40CPerror(
        cudaMalloc((void**) &vertex_list.d_changed,
            nodes * sizeof(int)),
        "cudaMalloc VertexType::d_changed failed", __FILE__, __LINE__);

    b40c::util::B40CPerror(
        cudaMalloc((void**) &vertex_list.d_min_dists,
            nodes * sizeof(DataType)),
        "cudaMalloc VertexType::d_min_dists failed", __FILE__,
        __LINE__);

    b40c::util::B40CPerror(
        cudaMalloc((void**) &vertex_list.d_visited_flag,
            nodes * sizeof(int)),
        "cudaMalloc VertexType::d_visited_flag failed", __FILE__,
        __LINE__);

    int memset_block_size = 256;
    int memset_grid_size_max = 32 * 1024;   // 32K CTAs
    int memset_grid_size;

    memset_grid_size =
        B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);

    b40c::util::MemsetKernel<DataType><<<memset_grid_size,
    memset_block_size, 0, 0>>>(vertex_list.d_dists, 0.15, nodes);

//    // Initialize d_dists_out elements
//    cudaMemcpy(vertex_list.d_dists_out,
//        vertex_list.d_dists,
//        nodes * sizeof(DataType),
//        cudaMemcpyDeviceToDevice);

    b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0,
    0>>>(vertex_list.d_changed, 0, nodes);

    b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0,
    0>>>(vertex_list.d_visited_flag, 0, nodes);

    b40c::util::MemsetKernel<DataType><<<memset_grid_size,
    memset_block_size, 0, 0>>>(vertex_list.d_min_dists, 0.0, nodes);

    b40c::util::SequenceKernel<int><<<memset_grid_size, memset_block_size,
    0, 0>>>(d_frontier_keys[0], nodes);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[1], d_frontier_keys[0],
            nodes * sizeof(int), cudaMemcpyDeviceToDevice),
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

    thrust::adjacent_difference(thrust::device, d_row_offsets_ptr,
        d_row_offsets_ptr + nodes + 1, d_num_out_edge_ptr);
    vertex_list.d_num_out_edge++;

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
    return POST_APPLY_ALL;
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
      vertex_list.d_min_dists[vertex_id] += final_value;
    }
  };

  /**
   * For each vertex in the frontier,
   */
  struct gather_edge
  {
    __device__
    void operator()(const int vertex_id, const int neighbor_id_in,
        VertexType &vertex_list, EdgeType &edge_list,
        GatherType& new_value)
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
    void operator()(const int vertex_id, const int iteration,
        VertexType& vertex_list, EdgeType& edge_list)
    {

      const DataType oldvalue = vertex_list.d_dists[vertex_id];
      const DataType gathervalue = vertex_list.d_min_dists[vertex_id];
      const DataType newvalue = 0.15f + (1.0f - 0.15f) * gathervalue;

      if (fabs(oldvalue - newvalue) < 0.01f)
        vertex_list.d_changed[vertex_id] = 0;
      else
      {
//        if(vertex_id < 200) printf("(%d %.3f) ", vertex_id, newvalue);
        vertex_list.d_changed[vertex_id] = 1;
      }

      vertex_list.d_dists[vertex_id] = newvalue;
    }
  };

  /** post-apply function (invoked after threads in apply() synchronize at a memory barrier). */
  struct post_apply
  {
    __device__
    void operator()(const int vertex_id, VertexType& vertex_list,
        EdgeType& edge_list)
    {
      vertex_list.d_visited_flag[vertex_id] = 0;
      vertex_list.d_min_dists[vertex_id] = 0.0;
//      vertex_list.d_dists[vertex_id] = vertex_list.d_dists_out[vertex_id];
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
    bool operator()(const int vertex_id, VertexType &vertex_list,
        EdgeType& edge_list)
    {
      return vertex_list.d_changed[vertex_id];
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
        const int vertex_id, const int neighbor_id_in,
        const int edge_id, VertexType& vertex_list, EdgeType& edge_list,
        int& frontier, int& misc_value)
    {
//      const int src_dist = vertex_list.d_dists[vertex_id];
//      const int dst_dist = vertex_list.d_dists[neighbor_id_in];
      if ((changed
          && atomicCAS(&vertex_list.d_visited_flag[neighbor_id_in], 0,
              1) == 0))
        frontier = neighbor_id_in;
      else
        frontier = -1;

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
        VertexType &vertex_list, EdgeType &edge_list, int& misc_value)
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
    cudaMemcpy(h_output, vertex_list.d_dists,
        sizeof(DataType) * vertex_list.nodes, cudaMemcpyDeviceToHost);
  }

//  /**
//   * destructor: free device memory
//   */
//  ~pagerank()
//  {
//    if (d_changed) util::B40CPerror(cudaFree(d_changed), "GpuSlice cudaFree d_changed failed", __FILE__, __LINE__);
//    if (d_dists) util::B40CPerror(cudaFree(d_dists), "GpuSlice cudaFree d_dists failed", __FILE__, __LINE__);
//    if (d_min_dists) util::B40CPerror(cudaFree(d_min_dists), "GpuSlice cudaFree d_min_dists failed", __FILE__, __LINE__);
//    if (d_num_out_edge) util::B40CPerror(cudaFree(d_num_out_edge), "GpuSlice cudaFree d_num_out_edge failed", __FILE__, __LINE__);
//    if (d_visited_flag) util::B40CPerror(cudaFree(d_visited_flag), "GpuSlice cudaFree d_visited_flag failed", __FILE__, __LINE__);
//  }
};

#endif /* SSSP_H_ */
