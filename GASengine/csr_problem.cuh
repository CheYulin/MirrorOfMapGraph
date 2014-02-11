/******************************************************************************
 * GPU CSR storage management structure for problem data
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>
#include <GASengine/problem_type.cuh>
#include <config.h>

#include <vector>

using namespace b40c;

enum SrcVertex
{
  SINGLE, ALL
};

enum GatherEdges
{
  NO_GATHER_EDGES, GATHER_IN_EDGES, GATHER_OUT_EDGES, GATHER_ALL_EDGES
};

enum ExpandEdges
{
  NO_EXPAND_EDGES, EXPAND_IN_EDGES, EXPAND_OUT_EDGES, EXPAND_ALL_EDGES
};

enum ApplyVertices
{
  NO_APPLY_VERTICES, APPLY_ALL, APPLY_FRONTIER
};

enum PostApplyVertices
{
  NO_POST_APPLY_VERTICES, POST_APPLY_ALL, POST_APPLY_FRONTIER
};

namespace GASengine
{

  /**
   * Enumeration of global frontier queue configurations
   */
  enum FrontierType
  {
    VERTEX_FRONTIERS,		// O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,			// O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS,			// O(n) global vertex frontier, O(m) global edge frontier
    MULTI_GPU_FRONTIERS,			// O(MULTI_GPU_VERTEX_FRONTIER_SCALE * n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier

    MULTI_GPU_VERTEX_FRONTIER_SCALE = 2,
  };

  /**
   * CSR storage management structure for BFS problems.
   */
  template<typename Program, typename _VertexId, typename _SizeT, typename _EValue, bool MARK_PREDECESSORS, // Whether to mark predecessors (vs. mark distance from source)
      bool WITH_VALUE>
  // Whether to include edge/ndoe value computation with BFS
  struct CsrProblem
  {
    //---------------------------------------------------------------------
    // Typedefs and constants
    //---------------------------------------------------------------------

    typedef ProblemType<Program, // vertex type
        _VertexId,				// VertexId
        _SizeT,					// SizeT
        _EValue,				// Edge Value
        unsigned char,			// VisitedMask
        unsigned char, 			// ValidFlag
        MARK_PREDECESSORS,		// MARK_PREDECESSORS
        WITH_VALUE>             // WITH_VALUE
    ProblemType;

    typedef typename Program::VertexType VertexType;
    typedef typename Program::EdgeType EdgeType;
    typedef typename Program::MiscType MiscType;
    typedef typename Program::VertexId VertexId;
    typedef typename Program::SizeT SizeT;
    typedef typename ProblemType::VisitedMask VisitedMask;
    typedef typename ProblemType::ValidFlag ValidFlag;
    typedef typename Program::DataType EValue;

    //---------------------------------------------------------------------
    // Helper structures
    //---------------------------------------------------------------------

    /**
     * Graph slice per GPU
     */
    struct GraphSlice
    {
      // GPU index
      int gpu;

      // Standard CSR device storage arrays
//      VertexId *d_vertex_ids; // Just plain vertex id array for backward contract kernel
      VertexId *d_column_indices;
      SizeT *d_row_offsets;
      VertexId *d_row_indices;
      SizeT *d_column_offsets;
      EValue* d_edge_values;

      VertexId *d_preds;               // Predecessor values
      int num_src;
      int *srcs;
      int init_num_elements;
      int outer_iter_num;
      int directed;

      VertexType vertex_list;
      EdgeType edge_list;
      // Best-effort mask for keeping track of which vertices we've seen so far
      VisitedMask *d_visited_mask;
      SizeT *d_visit_flags; // Track if same vertex is being expanded inside the same frontier-queue

      // Frontier queues.  Keys track work, values optionally track predecessors.  Only
      // multi-gpu uses triple buffers (single-GPU only uses ping-pong buffers).
      util::TripleBuffer<VertexId, MiscType> frontier_queues;
      SizeT frontier_elements[3];
      SizeT predecessor_elements[3];

      // Flags for filtering duplicates from the edge-frontier queue when partitioning during multi-GPU BFS.
      ValidFlag *d_filter_mask;

      // Number of nodes and edges in slice
      VertexId nodes;
      SizeT edges;

      // CUDA stream to use for processing this slice
      cudaStream_t stream;

      /**
       * Constructor
       */
      GraphSlice(int gpu, int directed, cudaStream_t stream) :
          gpu(gpu), directed(directed), d_column_indices(NULL), d_row_offsets(NULL), d_row_indices(NULL), d_column_offsets(NULL), d_edge_values(NULL),
          d_preds(NULL), d_visited_mask(NULL), d_filter_mask(NULL), d_visit_flags(NULL), nodes(0), edges(0), stream(stream)
      {
        // Initialize triple-buffer frontier queue lengths
        for (int i = 0; i < 3; i++)
        {
          frontier_elements[i] = 0;
          predecessor_elements[i] = 0;
        }
      }

      /**
       * Destructor
       */
      virtual ~GraphSlice()
      {
        // Set device
        util::B40CPerror(cudaSetDevice(gpu), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

        // Free pointers
//        if (d_vertex_ids) util::B40CPerror(cudaFree(d_vertex_ids), "GpuSlice cudaFree d_vertex_ids failed", __FILE__, __LINE__);
        if (d_column_indices) util::B40CPerror(cudaFree(d_column_indices), "GpuSlice cudaFree d_column_indices failed", __FILE__, __LINE__);
        if (d_row_offsets) util::B40CPerror(cudaFree(d_row_offsets), "GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
        if (directed == 1)
        {
          if (d_row_indices) util::B40CPerror(cudaFree(d_row_indices), "GpuSlice cudaFree d_row_indices failed", __FILE__, __LINE__);
          if (d_column_offsets) util::B40CPerror(cudaFree(d_column_offsets), "GpuSlice cudaFree d_column_offsets failed", __FILE__, __LINE__);
        }
        if (d_edge_values) util::B40CPerror(cudaFree(d_edge_values), "GpuSlice cudaFree d_edge_values", __FILE__, __LINE__);
        if (d_visited_mask) util::B40CPerror(cudaFree(d_visited_mask), "GpuSlice cudaFree d_visited_mask failed", __FILE__, __LINE__);
        if (d_filter_mask) util::B40CPerror(cudaFree(d_filter_mask), "GpuSlice cudaFree d_filter_mask failed", __FILE__, __LINE__);
        if (d_visit_flags) util::B40CPerror(cudaFree(d_visit_flags), "GpuSlice cudaFree d_visit_flags failed", __FILE__, __LINE__);
        for (int i = 0; i < 3; i++)
        {
          if (frontier_queues.d_keys[i]) util::B40CPerror(cudaFree(frontier_queues.d_keys[i]), "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__);
          if (frontier_queues.d_values[i]) util::B40CPerror(cudaFree(frontier_queues.d_values[i]), "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__);
        }

        // Destroy stream
        if (stream)
        {
          util::B40CPerror(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
        }
      }
    };

    //---------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------

    // Number of GPUS to be sliced over
    int num_gpus;

    // Size of the graph
    SizeT nodes;
    SizeT edges;
    Config cfg;

    // Set of graph slices (one for each GPU)
    std::vector<GraphSlice*> graph_slices;

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    CsrProblem(Config cfg) :
        num_gpus(0), nodes(0), edges(0), cfg(cfg)
    {
    }

    /**
     * Destructor
     */
    virtual ~CsrProblem()
    {
      // Cleanup graph slices on the heap
      for (typename std::vector<GraphSlice*>::iterator itr = graph_slices.begin(); itr != graph_slices.end(); itr++)
      {
        if (*itr) delete (*itr);
      }
    }

    /**
     * Returns index of the gpu that owns the neighbor list of
     * the specified vertex
     */
    template<typename VertexId>
    int GpuIndex(VertexId vertex)
    {
      if (graph_slices.size() == 1)
      {

        // Special case for only one GPU, which may be set as with
        // an ordinal other than 0.
        return graph_slices[0]->gpu;

      }
      else
      {

        return vertex % num_gpus;
      }
    }

    /**
     * Returns the row within a gpu's GraphSlice row_offsets vector
     * for the specified vertex
     */
    template<typename VertexId>
    VertexId GraphSliceRow(VertexId vertex)
    {
      return vertex / num_gpus;
    }

    /**
     * Extract into a single host vector the BFS results disseminated across
     * all GPUs
     */
    cudaError_t ExtractResults(EValue *h_values)
    {
      cudaError_t retval = cudaSuccess;

      do
      {
        if (graph_slices.size() == 1)
        {
          // Set device
          if (util::B40CPerror(cudaSetDevice(graph_slices[0]->gpu), "CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

          Program::extractResult(graph_slices[0]->vertex_list, h_values);

        }
        else
        {
          printf("Multi GPU is not supported yet!\n");
          exit(0);
        }
      }
      while (0);

      return retval;
    }

    /**
     * Initialize from host CSR problem
     */
    cudaError_t FromHostProblem(bool stream_from_host,		// Only meaningful for single-GPU BFS
        SizeT nodes,
        SizeT edges,
        VertexId *h_column_indices,
        SizeT *h_row_offsets,
        EValue *h_edge_values,
        VertexId *h_row_indices,
        SizeT *h_column_offsets,
        int num_gpus,
        int directed)
    {
      int device = cfg.getParameter<int>("device");
      cudaError_t retval = cudaSuccess;
      this->nodes = nodes;
      this->edges = edges;

      this->num_gpus = num_gpus;

      do
      {
        if (num_gpus <= 1)
        {

          // Create a single GPU slice for the currently-set gpu
          int gpu = device;
          if (retval = util::B40CPerror(cudaSetDevice(gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__)) break;

          graph_slices.clear();
          graph_slices.push_back(new GraphSlice(gpu, directed, 0));
          graph_slices[0]->nodes = nodes;
          graph_slices[0]->edges = edges;

          printf("GPU %d column_indices: %lld elements (%lld bytes)\n", graph_slices[0]->gpu, (unsigned long long) (graph_slices[0]->edges),
              (unsigned long long) (graph_slices[0]->edges * sizeof(VertexId) * sizeof(SizeT)));

          if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_column_indices, graph_slices[0]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_column_indices failed",
              __FILE__, __LINE__)) break;

          if (directed)
          {
            if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_row_indices, graph_slices[0]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_row_indices failed", __FILE__,
                __LINE__)) break;
          }

//          if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_vertex_ids, graph_slices[0]->nodes * sizeof(VertexId)), "CsrProblem cudaMalloc d_vertex_ids failed", __FILE__,
//              __LINE__)) break;

          if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_column_indices, h_column_indices, graph_slices[0]->edges * sizeof(VertexId), cudaMemcpyHostToDevice),
              "CsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

          if (directed)
          {
            if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_row_indices, h_row_indices, graph_slices[0]->edges * sizeof(VertexId), cudaMemcpyHostToDevice),
                "CsrProblem cudaMemcpy d_row_indices failed", __FILE__, __LINE__)) break;
          }

//          if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_vertex_ids, h_vertex_ids, graph_slices[0]->nodes * sizeof(VertexId), cudaMemcpyHostToDevice),
//              "CsrProblem cudaMemcpy d_vertex_ids failed", __FILE__, __LINE__)) break;

          // Allocate and initialize d_edge_values
          if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_edge_values, graph_slices[0]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_edge_values failed", __FILE__,
              __LINE__)) break;

//                if (WITH_VALUE)
          {
            if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_edge_values, h_edge_values, graph_slices[0]->edges * sizeof(VertexId), cudaMemcpyHostToDevice),
                "CsrProblem cudaMemcpy d_edge_values failed", __FILE__, __LINE__)) break;
          }

          // Allocate and initialize d_row_offsets

          printf("GPU %d row_offsets: %lld elements (%lld bytes)\n", graph_slices[0]->gpu, (unsigned long long) (graph_slices[0]->nodes + 1),
              (unsigned long long) (graph_slices[0]->nodes + 1) * sizeof(SizeT));

          if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_row_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT)), "CsrProblem cudaMalloc d_row_offsets failed",
              __FILE__, __LINE__)) break;

          if (directed)
          {
            if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_column_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT)), "CsrProblem cudaMalloc d_column_offsets failed",
                __FILE__, __LINE__))
              break;
          }

          if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_row_offsets, h_row_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT), cudaMemcpyHostToDevice),
              "CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

          if (directed)
          {
            if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_column_offsets, h_column_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT), cudaMemcpyHostToDevice),
                "CsrProblem cudaMemcpy d_column_offsets failed", __FILE__, __LINE__)) break;
          }
        }
        else //TODO: multiple GPU
        {
        }

      }
      while (0);

//      delete[] h_vertex_ids;

      return retval;
    }

    /**
     * Performs any initialization work needed for this problem type.  Must be called
     * prior to each search
     */
    cudaError_t Reset(FrontierType frontier_type,	// The frontier type (i.e., edge/vertex/mixed)
        double queue_sizing) //starting vertex
    {
      cudaError_t retval = cudaSuccess;

      for (int gpu = 0; gpu < num_gpus; gpu++)
      {

        // Set device
        if (retval = util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu), "CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

        //
        // Allocate visited masks for the entire graph if necessary
        //

        int visited_mask_bytes = ((nodes * sizeof(VisitedMask)) + 8 - 1) / 8;				// round up to the nearest VisitedMask
        int visited_mask_elements = visited_mask_bytes * sizeof(VisitedMask);
        if (!graph_slices[gpu]->d_visited_mask)
        {

//              printf("GPU %d visited mask: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) visited_mask_elements, (unsigned long long) visited_mask_bytes);

          if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_visited_mask, visited_mask_bytes), "CsrProblem cudaMalloc d_visited_mask failed", __FILE__, __LINE__))
            return retval;
        }

        //
        // Allocate frontier queues if necessary
        //

        // Determine frontier queue sizes
        SizeT new_frontier_elements[3] = { 0, 0, 0 };
        SizeT new_predecessor_elements[3] = { 0, 0, 0 };

        switch (frontier_type)
        {
          case VERTEX_FRONTIERS:
            // O(n) ping-pong global vertex frontiers
            new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
            new_frontier_elements[1] = new_frontier_elements[0];
            break;

          case EDGE_FRONTIERS:
            // O(m) ping-pong global edge frontiers
            new_frontier_elements[0] = double(graph_slices[gpu]->edges) * queue_sizing;
            new_frontier_elements[1] = new_frontier_elements[0];
            new_frontier_elements[2] = new_frontier_elements[0];
            new_predecessor_elements[0] = new_frontier_elements[0];
            new_predecessor_elements[1] = new_frontier_elements[1];
            new_predecessor_elements[2] = new_frontier_elements[2];
            break;

          case MIXED_FRONTIERS:
            // O(n) global vertex frontier, O(m) global edge frontier
            new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
            new_frontier_elements[1] = double(graph_slices[gpu]->edges) * queue_sizing;
            new_predecessor_elements[1] = new_frontier_elements[1];
            break;

          case MULTI_GPU_FRONTIERS:
            // O(n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier
            new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * MULTI_GPU_VERTEX_FRONTIER_SCALE * queue_sizing;
            new_frontier_elements[1] = double(graph_slices[gpu]->edges) * queue_sizing;
            new_frontier_elements[2] = new_frontier_elements[1];
            new_predecessor_elements[1] = new_frontier_elements[1];
            new_predecessor_elements[2] = new_frontier_elements[2];
            break;
        }

        // Iterate through global frontier queue setups
        for (int i = 0; i < 3; i++)
        {

          // Allocate frontier queue if not big enough
          if (graph_slices[gpu]->frontier_elements[i] < new_frontier_elements[i])
          {

            // Free if previously allocated
            if (graph_slices[gpu]->frontier_queues.d_keys[i])
            {
              if (retval = util::B40CPerror(cudaFree(graph_slices[gpu]->frontier_queues.d_keys[i]), "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
            }

            graph_slices[gpu]->frontier_elements[i] = new_frontier_elements[i];

            if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->frontier_queues.d_keys[i], graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)),
                "CsrProblem cudaMalloc frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
          }

          // Allocate predecessor queue if not big enough
          if (graph_slices[gpu]->predecessor_elements[i] < new_predecessor_elements[i])
          {

            // Free if previously allocated
            if (graph_slices[gpu]->frontier_queues.d_values[i])
            {
              if (retval = util::B40CPerror(cudaFree(graph_slices[gpu]->frontier_queues.d_values[i]), "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
            }

            graph_slices[gpu]->predecessor_elements[i] = new_predecessor_elements[i];

            if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->frontier_queues.d_values[i], graph_slices[gpu]->predecessor_elements[i] * sizeof(MiscType)),
                "CsrProblem cudaMalloc frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
          }
        }

        //
        // Allocate duplicate filter mask if necessary (for multi-gpu)
        //

        if ((frontier_type == MULTI_GPU_FRONTIERS) && (!graph_slices[gpu]->d_filter_mask))
        {

          if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_filter_mask, graph_slices[gpu]->frontier_elements[1] * sizeof(ValidFlag)),
              "CsrProblem cudaMalloc d_filter_mask failed", __FILE__, __LINE__)) return retval;
        }

        int memset_block_size = 256;
        int memset_grid_size_max = 32 * 1024;	// 32K CTAs
        int memset_grid_size;

        // Initialize d_visited_mask elements to 0
        memset_grid_size = B40C_MIN(memset_grid_size_max, (visited_mask_elements + memset_block_size - 1) / memset_block_size);

        util::MemsetKernel<VisitedMask><<<memset_grid_size, memset_block_size, 0,
        graph_slices[gpu]->stream>>>(graph_slices[gpu]->d_visited_mask, 0, visited_mask_elements);

        if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;
      }

      return retval;
    }
  };

} // namespace GASengine

