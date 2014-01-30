/*
 * sssp.h
 *
 *  Created on: Dec 9, 2013
 *      Author: zhisong
 */

#ifndef SSSP_H_
#define SSSP_H_

#include <GASengine/csr_problem.cuh>

/**
 * \brief Single Source Shortest Path.
 *
 * Given a starting vertex, label the reachable vertices of the graph
 * such that the shortest distance from the starting vertex is marked
 * on each vertex.  Vertices that are not reachable will report a
 * label of -1.
 */
struct sssp
{

  /**
   * The data type for the edge weights, the distance labels on the
   * vertices, etc.
   */
  typedef int DataType;
  /**
   * The data type for the used defined scratch column that is 1:1
   * with the frontier.
   */
  typedef DataType MiscType;
  /**
   * The data type for the intermediate results of the GATHER kernel.
   */
  typedef DataType GatherType;

  /**
   * The initial distance for all vertices.  If a vertex is never
   * visited. it will be marked with this distance. */
  static const DataType INIT_VALUE = 100000000;

  /**
   * \brief The VertexType is the data type for the vertex list.
   *
   * The VertexType is the data type for the vertex list.  The
   * VertexType is a structure of arrays to provide coalesced access
   * to the data.  The index into the array is the vertex identifier
   * (vertexId).
   */
  struct VertexType
  {
    int nodes; /**< #of nodes. */
    int edges; /**< #of edges. */
    DataType* d_dists_out; /**< new value computed by apply. */
    int* d_changed; /**< 1 iff dists_out was changed in apply. */
    DataType* d_dists; /**< the actual distance computed by post_apply. */
    DataType* d_min_dists; /**< The intermediate value for global synchronization computed in contract. used by the next apply(). */

    VertexType() :
        d_dists(NULL), d_dists_out(NULL), d_changed(NULL), d_min_dists(
            NULL), nodes(0), edges(0)
    {
    }
  };

  /**
   * \brief The EdgeType is the data type for the edge list.
   *
   * The EdgeType is the data type for the edge list.  The EdgeType
   * is a structure of arrays to provide coalesced access to the data.
   * The index into the array is the edge identifier (edgeId).
   */
  struct EdgeType
  {
    int nodes; /**< #of nodes. */
    int edges; /**< #of edges. */
    DataType* d_weights; /**< The weights for the edges. */

    EdgeType() :
        d_weights(NULL), nodes(0), edges(0)
    {
    }
  };

  /**
   * \brief Initialize the device memory.
   *
   * This function is invoked from main().  You can supply any
   * necessary arguments.  The function should only perform allocation
   * for device memory.  If you need to allocate host memory, do that
   * somewhere else.
   */
  static void Initialize
  ( const int nodes, /**< The number of vertices in the graph. */
    const int edges, /**< The number of edges in the graph. */
    int num_srcs, /**< The number of vertices in the initial frontier. */
    int* srcs, /**< The vertices in the initial frontier. */
    int* d_row_offsets,
    int* d_column_indices,
    int* d_column_offsets,
    int* d_row_indices,
    int* d_edge_values,
    VertexType &vertex_list,
    EdgeType &edge_list,
    int* d_frontier_keys[3], /**< The frontier arrays. */
    MiscType* d_frontier_values[3] /**< User data arrays that are 1:1 with the frontier. */
    )
  {
    // save the #of nodes (vertices) and #of edges in the graph.
    vertex_list.nodes = nodes;
    vertex_list.edges = edges;

    // Allocate data for vertex_list.
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

    if (vertex_list.d_changed == NULL)
      b40c::util::B40CPerror(
          cudaMalloc((void**) &vertex_list.d_changed,
              nodes * sizeof(int)),
          "cudaMalloc VertexType::d_changed failed", __FILE__, __LINE__);

    if (vertex_list.d_min_dists == NULL)
      b40c::util::B40CPerror(
          cudaMalloc((void**) &vertex_list.d_min_dists,
              nodes * sizeof(DataType)),
          "cudaMalloc VertexType::d_min_dists failed", __FILE__,
          __LINE__);

    // Allocate data for edge_list.
    if (edge_list.d_weights == NULL)
      b40c::util::B40CPerror(
          cudaMalloc((void**) &edge_list.d_weights,
              edges * sizeof(DataType)),
          "cudaMalloc edge_list.d_weights failed", __FILE__,
          __LINE__);

    // Parameters for parallel initialization.
    int memset_block_size = 256;
    int memset_grid_size_max = 32 * 1024;	// 32K CTAs
    int memset_grid_size;

    // Initialize d_dists elements to 100000000
    memset_grid_size =
        B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);
    b40c::util::MemsetKernel<DataType><<<memset_grid_size,
        memset_block_size, 0, 0>>>(vertex_list.d_dists, INIT_VALUE,
        nodes);

    // Initialize d_labels elements to -1
    memset_grid_size =
        B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);
    b40c::util::MemsetKernel<DataType><<<memset_grid_size,
        memset_block_size, 0, 0>>>(vertex_list.d_dists_out, INIT_VALUE,
        nodes);

    // Initialize d_labels elements to -1
    memset_grid_size =
        B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);
    b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0,
        0>>>(vertex_list.d_changed, 0, nodes);

    // Initialize d_min_dists elements to -1
    memset_grid_size =
        B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);
    b40c::util::MemsetKernel<DataType><<<memset_grid_size,
        memset_block_size, 0, 0>>>(vertex_list.d_min_dists, INIT_VALUE,
        nodes);

    // Initialize edge data
    if (b40c::util::B40CPerror(
        cudaMemcpy(edge_list.d_weights, d_edge_values,
            edges * sizeof(DataType), cudaMemcpyDeviceToDevice),
        "CsrProblem cudaMemcpy edge d_weights failed", __FILE__, __LINE__))
      exit(0);

    printf("Starting vertex: ");
        for(int i=0; i<num_srcs; i++)
          printf("%d ", srcs[i]);
        printf("\n");

    int init_dists[1];
    init_dists[0] = 0;

    if (b40c::util::B40CPerror(
        cudaMemcpy(vertex_list.d_dists + srcs[0], init_dists,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_dists failed", __FILE__, __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(vertex_list.d_dists_out + srcs[0], init_dists,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_dists_out failed", __FILE__, __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[0], srcs, num_srcs * sizeof(int),
            cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__,
        __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[1], srcs, num_srcs * sizeof(int),
            cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__,
        __LINE__))
      exit(0);

    int init_value[1] = { INIT_VALUE };
    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_values[0], init_value,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
        __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_values[1], init_value,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
        __LINE__))
      exit(0);

    int init_changed[1] = { 1 };
    if (b40c::util::B40CPerror(
        cudaMemcpy(vertex_list.d_changed, init_changed,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_changed failed", __FILE__, __LINE__))
      exit(0);
  }

  /**
   * \brief The initial frontier for the algorithm is a single starting vertex.
   */
  static SrcVertex srcVertex()
  {
    return SINGLE;
  }

  /**
   * \brief The GATHER kernel is not used.
   */
  static GatherEdges gatherOverEdges()
  {
    return NO_GATHER_EDGES;
  }

  /**
   * \brief The APPLY kernel is executed over the vertices in the frontier.
   */
  static ApplyVertices applyOverEdges()
  {
    return APPLY_FRONTIER;
  }

  /**
   * \brief The POST-APPLY kernel is executed over the vertices in the frontier.
   */
  static PostApplyVertices postApplyOverEdges()
  {
    return POST_APPLY_FRONTIER;
  }

  /**
   * \brief The APPLY kernel is over the out-edges of the graph.
   */
  static ExpandEdges expandOverEdges()
  {
    return EXPAND_OUT_EDGES;
  }

  /**
   * \brief For each vertex in the frontier, ... (Not used by SSSP).
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
   * \brief For each vertex in the frontier, ... (Not used by SSSP).
   */
  struct gather_edge
  {
    __device__
    /**
     * Compute an intermediate gather result for the specified vertex
     * and edge.
     * 
     * @param vertex_id The vertex identifier of the source vertex.
     *
     * @param neighbor_id_in The vertex identifier of the target
     * vertex.
     *
     * @param edge_id The index of the edge in the edge-list.
     *
     * @param vertex_list The vertices in the graph.
     *
     * @param edge_list The edges in the graph.
     *
     * @param new_value The intermediate gather result (set by
     * side-effect).
     */
    void operator()(const int vertex_id, const int neighbor_id_in,
        VertexType &vertex_list, EdgeType &edge_list, GatherType& new_value)
    {

    }
  };

  /**
   * \brief The binary operator used to combine the intermediate
   * results during a GATHER.
   *
   * This is SUM(left,right) for SSSP.
   */
  struct gather_sum
  {
    /**
     * Combine and return two intermediate gather results.
     * 
     * @param left An intermediate gather result.
     *
     * @param right Another intermediate gather result.
     *
     * @return An intermediate gather result that combines the impact
     * of both arguments.
     */
    __device__ GatherType operator()(const GatherType &left,
        const GatherType &right)
    {
      return left + right;
    }
  };

  /** 
   * \brief The APPLY kernel invokes this device function to update
   * the vertex state given the gather results.
   */
  struct apply
  {
    __device__
    /**
     * \brief Update the vertex state given the gather results.
     *
     * @param vertex_id The vertex identifier.
     *
     * @param iteration The current iteration number.
     
     * @param vertex_list The vertex list.  Index into this using the
     * vertex identifier.
     */
    void operator()(const int vertex_id, const int iteration,
        VertexType& vertex_list, EdgeType& edge_list)
    {

      const int oldvalue = vertex_list.d_dists[vertex_id];
      const int gathervalue = vertex_list.d_min_dists[vertex_id];
      const int newvalue = min(oldvalue, gathervalue);

      if (oldvalue == newvalue)
        vertex_list.d_changed[vertex_id] = 0;
      else
        vertex_list.d_changed[vertex_id] = 1;

      vertex_list.d_dists_out[vertex_id] = newvalue;
    }
  };

  /** 
   * \brief The post-apply device function.
   * 
   * The post-apply device function is invoked by the POST-APPLY
   * kernel invoked after threads in APPLY kernel synchronize at a
   * memory barrier.
   */
  struct post_apply
  {
    __device__
    /**
     * \brief Update the vertex state given the gather results
     * (invoked after thread synchronize at the end of the APPLY
     * kernel).
     *
     * @param vertex_id The vertex identifier.
     *
     * @param iteration The current iteration number.
     *
     * @param vertex_list The vertex list.  Index into this using the vertex identifier.
     */
    void operator()(const int vertex_id, VertexType& vertex_list, EdgeType& edge_list)
    {
      vertex_list.d_dists[vertex_id] = vertex_list.d_dists_out[vertex_id];
      vertex_list.d_min_dists[vertex_id] = INIT_VALUE;
    }
  };

  /**
   * \brief Return true iff this vertex has changed state.
   * 
   * The return value of this device function will be passed into the
   * expand_edge device function as [bool:change]. For example, this
   * can check the state of the vertex to decide whether it has been
   * updated and only expand its neighbors if it has be updated.
   */
  struct expand_vertex
  {
    __device__
    /**
     * Return true iff this vertex has changed state.
     *
     * @param vertex_id The vertex identifier of the source vertex.
     *
     * @param vertex_list The vertices in the graph.
     */
    bool operator()(const int vertex_id, VertexType &vertex_list, EdgeType& edge_list)
    {
      return vertex_list.d_changed[vertex_id];
    }
  };

  /**
   * \brief Generate the new frontier.
   * 
   * The expand stage creates a new frontier. The frontier can have a
   * lot of duplicates.  The contract stage will eliminate (some of)
   * those duplicates.  There are two outputs for expand.  One is the
   * new frontier.  The other is a "predecessor" array.  These arrays
   * have a 1:1 correspondence.  The predecessor array is available
   * for user data, depending on the algorithm.  For example, for BFS
   * it is used to store the vertex_id of the vertex from which this
   * vertex was reached by a one step traversal along some edge.
   */
  struct expand_edge
  {
    __device__
    /**
     * Generate the new frontier.
     *
     * @param changed true iff the device function expand_vertex
     * evaluated to true for this vertex.
     *
     * @param iteration The current iteration number.
     *
     * @param vertex_id The vertex identifier of the source vertex.
     *
     * @param neighbor_id_in The vertex identifier of the target
     * vertex.
     *
     * @param edge_id The index of the edge in the edge-list.
     *
     * @param vertex_list The vertices in the graph.
     *
     * @param edge_list The edges in the graph.
     *
     * @param frontier The vertex identifier to be added into the new
     * frontier. Set to neighbor_id_in if you want to visit that
     * vertex and set to -1 if you do not want to visit that vertex
     * along this edge.
     *
     * @param misc_value The optional value to be written into the
     * scratch array. This array has a 1:1 correspondence with the
     * frontier array.
     */
    void operator()(const bool changed, const int iteration,
        const int vertex_id, const int neighbor_id_in, const int edge_id,
        VertexType& vertex_list, EdgeType& edge_list, int& frontier, int& misc_value)
    {
      const int src_dist = vertex_list.d_dists[vertex_id];
      const int dst_dist = vertex_list.d_dists[neighbor_id_in];
      DataType edge_value = edge_list.d_weights[edge_id];
//      printf("vertex_id=%d, edge_id=%d, neighbor_id_in=%d, edge_value=%d\n", vertex_id, edge_id, neighbor_id_in, edge_value);
      if ((changed || iteration == 0) && dst_dist > src_dist + edge_value)
//      if ((changed || iteration == 0) && dst_dist > src_dist + 1)
        frontier = neighbor_id_in;
      else
        frontier = -1;
      misc_value = src_dist + edge_value; // source dist + edge weight
//      misc_value = src_dist + 1;
    }
  };

  /**
   * \brief Reduce or eliminate duplicate vertices in the frontier.
   *
   * The contract stage is used to reduce the duplicates in the
   * frontier created by the Expand stage.
   */
  struct contract
  {
    __device__
    /**
     * Reduce or eliminate duplicate vertices in the frontier.
     *
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
       * Note: predecessor is source dist + edge weight for SSSP.
       * This writes on d_min_dists[] to find the minimum distinct for
       * this vertex.
       */

//			  printf("vertex_id=%d, misc_value=%d\n", vertex_id, misc_value);
      atomicMin(&vertex_list.d_min_dists[vertex_id], misc_value);

    }
  };

  /**
   * \brief Extract the SSSP labels from the vertices.
   *
   * This device function copies the SSSP results from the device to
   * the host.  It is invoked from the CsrProblem.
   *
   * @param vertex_list The vertex list.
   * 
   * @param h_output The host array to which the data will be
   * transferred.
   */
  static void extractResult(VertexType& vertex_list, DataType* h_output)
  {
    cudaMemcpy(h_output, vertex_list.d_dists, sizeof(DataType) * vertex_list.nodes, cudaMemcpyDeviceToHost);
  }

};

#endif /* SSSP_H_ */
