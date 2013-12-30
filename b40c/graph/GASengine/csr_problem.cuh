/******************************************************************************
 * Copyright 2010-2012 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 ******************************************************************************/

/******************************************************************************
 * GPU CSR storage management structure for BFS problem data
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>
#include <b40c/graph/GASengine/problem_type.cuh>
#include <config.h>

#include <vector>

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

namespace b40c
{
  namespace graph
  {
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
      template<typename _Program, typename _VertexId, typename _SizeT, typename _EValue, bool MARK_PREDECESSORS, // Whether to mark predecessors (vs. mark distance from source)
          bool WITH_VALUE>
      // Whether to include edge/ndoe value computation with BFS
      struct CsrProblem
      {
        //---------------------------------------------------------------------
        // Typedefs and constants
        //---------------------------------------------------------------------

        typedef ProblemType<_Program, // vertex type
            _VertexId,				// VertexId
            _SizeT,					// SizeT
            _EValue,				// Edge Value
            unsigned char,			// VisitedMask
            unsigned char, 			// ValidFlag
            MARK_PREDECESSORS,		// MARK_PREDECESSORS
            WITH_VALUE>             // WITH_VALUE
        ProblemType;

        typedef typename ProblemType::Program::VertexType VertexType;
        typedef typename ProblemType::Program::MiscType MiscType;
        typedef typename ProblemType::VertexId VertexId;
        typedef typename ProblemType::SizeT SizeT;
        typedef typename ProblemType::VisitedMask VisitedMask;
        typedef typename ProblemType::ValidFlag ValidFlag;
        typedef typename ProblemType::EValue EValue;

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
          VertexId *d_vertex_ids; // Just plain vertex id array for backward contract kernel
          VertexId *d_column_indices;
          SizeT *d_row_offsets;
          VertexId *d_row_indices;
          SizeT *d_column_offsets;

//          VertexId *d_row_indices;
//          SizeT *d_column_offsets;

          VertexId *d_labels;				// Source distance
          VertexId *d_preds;               // Predecessor values
          EValue *d_edge_values; // Weight attached to each edge, size equals to the size of d_column_indices
          EValue *d_node_values; // BC values attached to each node, size equals to the size of d_row_offsets
          EValue *d_sigmas;              // Sigma values attached to each node
          EValue *d_deltas;              // Delta values attached to each node
          int *d_dists;
          int *d_changed;
          int num_src;
          int *srcs;
          int init_num_elements;

          VertexType vertex_list;
          //          VertexType gather_list;
//          VertexType gather_list_out;

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
          GraphSlice(int gpu, cudaStream_t stream) :
              gpu(gpu), d_vertex_ids(NULL), d_column_indices(NULL), d_row_offsets(NULL), d_edge_values(NULL), d_node_values(NULL), d_labels(NULL), d_preds(NULL), d_sigmas(NULL), d_deltas(NULL), d_dists(
                  NULL), d_changed(NULL), d_visited_mask(NULL), d_filter_mask(NULL), d_visit_flags(NULL), nodes(0), edges(0), stream(stream)
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
            if (d_vertex_ids) util::B40CPerror(cudaFree(d_vertex_ids), "GpuSlice cudaFree d_vertex_ids failed", __FILE__, __LINE__);
            if (d_column_indices) util::B40CPerror(cudaFree(d_column_indices), "GpuSlice cudaFree d_column_indices failed", __FILE__, __LINE__);
            if (d_row_offsets) util::B40CPerror(cudaFree(d_row_offsets), "GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
            if (d_edge_values) util::B40CPerror(cudaFree(d_edge_values), "GpuSlice cudaFree d_edge_values", __FILE__, __LINE__);
            if (d_node_values) util::B40CPerror(cudaFree(d_node_values), "GpuSlice cudaFree d_node_values", __FILE__, __LINE__);
//            if (d_labels) util::B40CPerror(cudaFree(d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
//            if (d_preds) util::B40CPerror(cudaFree(d_preds), "GpuSlice cudaFree d_preds failed", __FILE__, __LINE__);
//            if (d_sigmas) util::B40CPerror(cudaFree(d_sigmas), "GpuSlice cudaFree d_sigmas failed", __FILE__, __LINE__);
//            if (d_deltas) util::B40CPerror(cudaFree(d_deltas), "GpuSlice cudaFree d_deltas failed", __FILE__, __LINE__);
//            if (d_dists) util::B40CPerror(cudaFree(d_dists), "GpuSlice cudaFree d_dists failed", __FILE__, __LINE__);
//            if (d_changed) util::B40CPerror(cudaFree(d_changed), "GpuSlice cudaFree d_changed failed", __FILE__, __LINE__);
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
//          for (typename std::vector<GraphSlice*>::iterator itr = graph_slices.begin(); itr != graph_slices.end(); itr++)
//          {
//            if (*itr) delete (*itr);
//          }
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
        cudaError_t ExtractResults(VertexId *h_dists, VertexId *h_labels, EValue *h_sigmas, EValue *h_deltas)
        {
          cudaError_t retval = cudaSuccess;

          do
          {
            if (graph_slices.size() == 1)
            {

              // Set device
              if (util::B40CPerror(cudaSetDevice(graph_slices[0]->gpu), "CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

              _Program::extractResult(graph_slices[0]->vertex_list, h_dists);

              // Special case for only one GPU, which may be set as with
              // an ordinal other than 0.
//              if (retval = util::B40CPerror(cudaMemcpy(h_dists, graph_slices[0]->vertex_list.d_dists, sizeof(VertexId) * graph_slices[0]->nodes, cudaMemcpyDeviceToHost),
//                  "CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

//              //copy node values out from GPU
//              if (retval = util::B40CPerror(cudaMemcpy(h_labels, graph_slices[0]->d_labels, sizeof(VertexId) * graph_slices[0]->nodes, cudaMemcpyDeviceToHost),
//                  "CsrProblem cudaMemcpy d_node_values failed", __FILE__, __LINE__)) break;
//
//              //copy sigma values out from GPU
//              if (retval = util::B40CPerror(cudaMemcpy(h_sigmas, graph_slices[0]->d_sigmas, sizeof(EValue) * graph_slices[0]->nodes, cudaMemcpyDeviceToHost), "CsrProblem cudaMemcpy d_sigmas failed",
//                  __FILE__, __LINE__)) break;
//
//              //copy delta values out from GPU
//              if (retval = util::B40CPerror(cudaMemcpy(h_deltas, graph_slices[0]->d_deltas, sizeof(EValue) * graph_slices[0]->nodes, cudaMemcpyDeviceToHost), "CsrProblem cudaMemcpy d_deltas failed",
//                  __FILE__, __LINE__)) break;

            }
            else
            {

//              VertexId **gpu_labels = new VertexId*[num_gpus];
//              EValue **gpu_node_values = new EValue*[num_gpus];
//              EValue **gpu_sigmas = new EValue*[num_gpus];
//              EValue **gpu_deltas = new EValue*[num_gpus];
//
//              // Copy out
//              for (int gpu = 0; gpu < num_gpus; gpu++)
//              {
//
//                // Set device
//                if (util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu), "CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;;
//
//                // Allocate and copy out
//                gpu_labels[gpu] = new VertexId[graph_slices[gpu]->nodes];
//                gpu_node_values[gpu] = new EValue[graph_slices[gpu]->nodes];
//                gpu_sigmas[gpu] = new EValue[graph_slices[gpu]->nodes];
//                gpu_deltas[gpu] = new EValue[graph_slices[gpu]->nodes];
//
//                if (retval = util::B40CPerror(cudaMemcpy(gpu_labels[gpu], graph_slices[gpu]->d_labels, sizeof(VertexId) * graph_slices[gpu]->nodes, cudaMemcpyDeviceToHost),
//                    "CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
//
//                //copy node values out from GPU
//                if (retval = util::B40CPerror(cudaMemcpy(gpu_node_values[gpu], graph_slices[gpu]->d_node_values, sizeof(EValue) * graph_slices[0]->nodes, cudaMemcpyDeviceToHost),
//                    "CsrProblem cudaMemcpy d_node_value failed", __FILE__, __LINE__)) break;
//
//                //copy sigma values out from GPU
//                if (retval = util::B40CPerror(cudaMemcpy(gpu_sigmas[gpu], graph_slices[gpu]->d_sigmas, sizeof(EValue) * graph_slices[0]->nodes, cudaMemcpyDeviceToHost),
//                    "CsrProblem cudaMemcpy d_sigma_value failed", __FILE__, __LINE__)) break;
//
//                //copy delta values out from GPU
//                if (retval = util::B40CPerror(cudaMemcpy(gpu_deltas[gpu], graph_slices[gpu]->d_deltas, sizeof(EValue) * graph_slices[0]->nodes, cudaMemcpyDeviceToHost),
//                    "CsrProblem cudaMemcpy d_delta_value failed", __FILE__, __LINE__)) break;
//
//              }
//              if (retval) break;
//
//              // Combine
//              for (VertexId node = 0; node < nodes; node++)
//              {
//                int gpu = GpuIndex(node);
//                VertexId slice_row = GraphSliceRow(node);
//                h_label[node] = gpu_labels[gpu][slice_row];
//                h_node_values[node] = gpu_node_values[gpu][slice_row];
//                h_sigmas[node] = gpu_sigmas[gpu][slice_row];
//                h_deltas[node] = gpu_deltas[gpu][slice_row];
//
//                switch (h_label[node])
//                {
//                  case -1:
//                  case -2:
//                    break;
//                  default:
//                    h_label[node] &= ProblemType::VERTEX_ID_MASK;
//                };
//              }
//
//              // Clean up
//              for (int gpu = 0; gpu < num_gpus; gpu++)
//              {
//                if (gpu_labels[gpu]) delete gpu_labels[gpu];
//                if (gpu_node_values[gpu]) delete gpu_node_values[gpu];
//                if (gpu_sigmas[gpu]) delete gpu_sigmas[gpu];
//                if (gpu_deltas[gpu]) delete gpu_deltas[gpu];
//              }
//              delete gpu_labels;
//              delete gpu_node_values;
//              delete gpu_sigmas;
//              delete gpu_deltas;
            }
          }
          while (0);

          return retval;
        }

        /**
         * Initialize from host CSR problem
         */
        cudaError_t FromHostProblem(char* source_file_name, bool stream_from_host,		// Only meaningful for single-GPU BFS
            SizeT nodes,
            SizeT edges,
            VertexId *h_column_indices,
            SizeT *h_row_offsets,
            EValue *h_edge_values,
            VertexId *h_row_indices,
            SizeT *h_column_offsets,
            EValue *h_node_values,
            int num_gpus)
        {
          int device = cfg.getParameter<int>("device");
          cudaError_t retval = cudaSuccess;
          this->nodes = nodes;
          this->edges = edges;

          this->num_gpus = num_gpus;
          VertexId *h_vertex_ids = new VertexId[nodes];
          for (int i = 0; i < nodes; ++i)
          {
            h_vertex_ids[i] = i;
          }

          do
          {
            if (num_gpus <= 1)
            {

              // Create a single GPU slice for the currently-set gpu
              int gpu = device;
//              if (retval = util::B40CPerror(cudaGetDevice(&gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
              if (retval = util::B40CPerror(cudaSetDevice(gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
//              printf("Running on device %d\n", device);
              graph_slices.push_back(new GraphSlice(gpu, 0));
              graph_slices[0]->nodes = nodes;
              graph_slices[0]->edges = edges;

              graph_slices[0]->num_src = 1;
              printf("source_file_name=%s\n", source_file_name);
              if (_Program::srcVertex() == SINGLE)
              {
                graph_slices[0]->init_num_elements = 1;
                int src_node = cfg.getParameter<int>("src");
                int origin = cfg.getParameter<int>("origin");
                printf("origin: %d\n", origin);
                const int max_src_num = 100;

                if (strcmp(source_file_name, ""))
                {
                  //TODO random sources
                  if (strcmp(source_file_name, "RANDOM") == 0)
                  {
                    graph_slices[0]->num_src = cfg.getParameter<int>("num_src");
                    graph_slices[0]->srcs = new int[graph_slices[0]->num_src];
                    printf("Using %d random starting vertices!\n", graph_slices[0]->num_src);
                    srand (time(NULL));
                    int count = 0;
                    while (count < graph_slices[0]->num_src)
                    {
                      int tmp_src = rand() % graph_slices[0]->nodes;
                      if (h_row_offsets[tmp_src + 1] - h_row_offsets[tmp_src] > 0)
                      {
                        graph_slices[0]->srcs[count++] = tmp_src;
                      }
                    }

                  }
                  else
                  {
                    printf("Using source file!\n");
                    FILE* src_file;
                    if ((src_file = fopen(source_file_name, "r")) == NULL)
                    {
                      printf("Source file open error!\n");
                      exit(0);
                    }

                    graph_slices[0]->srcs = new int[max_src_num];
                    for (graph_slices[0]->num_src = 0; graph_slices[0]->num_src < max_src_num; graph_slices[0]->num_src++)
                    {
                      if ((fscanf(src_file, "%d\n", &graph_slices[0]->srcs[graph_slices[0]->num_src]) != EOF))
                      {
                        if (origin == 1)
                          graph_slices[0]->srcs[graph_slices[0]->num_src]--; //0-based index
                      }
                      else
                        break;
                    }
                  }

                }
                else
                {
                  printf("Single source vertex!\n");
                  graph_slices[0]->num_src = 1;
                  graph_slices[0]->srcs = new int[1];
                  graph_slices[0]->srcs[0] = src_node;
                  if (origin == 1)
                    graph_slices[0]->srcs[0]--;
                }
              }

              else if (_Program::srcVertex() == ALL)
              {
                graph_slices[0]->srcs = new int[1]; //dummy not used
                graph_slices[0]->srcs[0] = -1; // dummy not used
                graph_slices[0]->num_src = 1;
                graph_slices[0]->init_num_elements = nodes;
              }
              else
              {
                printf("Invalid src vertex!\n");
                exit(0);
              }
//              graph_slices[0]->num_srcs = _num_srcs;
//              graph_slices[0]->srcs = new int[_num_srcs];
//              int origin = cfg.getParameter<int>("origin");
//              int src = cfg.getParameter<int>("src");
//			  if(origin == 0)
//				  graph_slices[0]->srcs[0] = src;
//			  else if(origin == 1)
//				  graph_slices[0]->srcs[0] = src - 1;
//			  else
//				  printf("Error: only 0-based or 1-based indices are supported\n");
//              graph_slices[0]->vertex_list.size = nodes;
//              graph_slices[0]->gather_list.size = nodes;

              if (stream_from_host)
              {

                printf("Streaming from host\n");
                // Map the pinned graph pointers into device pointers
                if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &graph_slices[0]->d_column_indices, (void *) h_column_indices, 0),
                    "CsrProblem cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &graph_slices[0]->d_row_offsets, (void *) h_row_offsets, 0),
                    "CsrProblem cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &graph_slices[0]->d_row_indices, (void *) h_row_indices, 0),
                    "CsrProblem cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &graph_slices[0]->d_column_offsets, (void *) h_column_offsets, 0),
                    "CsrProblem cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &graph_slices[0]->d_edge_values, (void *) h_edge_values, 0),
                    "CsrProblem cudaHostGetDevicePointer d_edge_values failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &graph_slices[0]->d_node_values, (void *) h_node_values, 0),
                    "CsrProblem cudaHostGetDevicePointer d_node_values failed", __FILE__, __LINE__)) break;

              }
              else
              {
                printf("NOT streaming from host\n");
                // Allocate and initialize d_column_indices

                printf("GPU %d column_indices: %lld elements (%lld bytes)\n", graph_slices[0]->gpu, (unsigned long long) (graph_slices[0]->edges),
                    (unsigned long long) (graph_slices[0]->edges * sizeof(VertexId) * sizeof(SizeT)));

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_column_indices, graph_slices[0]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_column_indices failed",
                    __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_row_indices, graph_slices[0]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_row_indices failed", __FILE__,
                    __LINE__)) break;

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_vertex_ids, graph_slices[0]->nodes * sizeof(VertexId)), "CsrProblem cudaMalloc d_vertex_ids failed", __FILE__,
                    __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_column_indices, h_column_indices, graph_slices[0]->edges * sizeof(VertexId), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_row_indices, h_row_indices, graph_slices[0]->edges * sizeof(VertexId), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy d_row_indices failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_vertex_ids, h_vertex_ids, graph_slices[0]->nodes * sizeof(VertexId), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy d_vertex_ids failed", __FILE__, __LINE__)) break;

                // Allocate and initialize d_edge_values
                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_edge_values, graph_slices[0]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_edge_values failed", __FILE__,
                    __LINE__)) break;

                if (WITH_VALUE)
                {
                  if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_edge_values, h_edge_values, graph_slices[0]->edges * sizeof(VertexId), cudaMemcpyHostToDevice),
                      "CsrProblem cudaMemcpy d_edge_values failed", __FILE__, __LINE__)) break;
                }

                // Allocate and initialize d_row_offsets

                printf("GPU %d row_offsets: %lld elements (%lld bytes)\n", graph_slices[0]->gpu, (unsigned long long) (graph_slices[0]->nodes + 1),
                    (unsigned long long) (graph_slices[0]->nodes + 1) * sizeof(SizeT));

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_row_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT)), "CsrProblem cudaMalloc d_row_offsets failed",
                    __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_column_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT)), "CsrProblem cudaMalloc d_column_offsets failed",
                    __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_row_offsets, h_row_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[0]->d_column_offsets, h_column_offsets, (graph_slices[0]->nodes + 1) * sizeof(SizeT), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy d_column_offsets failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[0]->d_node_values, (graph_slices[0]->nodes) * sizeof(EValue)), "CsrProblem cudaMalloc d_node_values failed", __FILE__,
                    __LINE__)) break;

                int memset_block_size = 256;
                int memset_grid_size_max = 32 * 1024;	// 32K CTAs
                int memset_grid_size;

                // Initialize d_node_values elements to 0
                memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[0]->nodes + memset_block_size - 1) / memset_block_size);
                util::MemsetKernel<EValue><<<memset_grid_size, memset_block_size, 0, graph_slices[0]->stream>>>(graph_slices[0]->d_node_values, 0, graph_slices[0]->nodes);

                if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;

              }

            }
            else
            {

              // Create multiple GPU graph slices
              for (int gpu = 0; gpu < num_gpus; gpu++)
              {

                // Set device
                if (retval = util::B40CPerror(cudaSetDevice(gpu), "CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                // Create stream
                cudaStream_t stream;
                if (retval = util::B40CPerror(cudaStreamCreate(&stream), "CsrProblem cudaStreamCreate failed", __FILE__, __LINE__)) break;

                // Create slice
                graph_slices.push_back(new GraphSlice(gpu, stream));
              }
              if (retval) break;

              // Count up nodes and edges for each gpu
              for (VertexId node = 0; node < nodes; node++)
              {
                int gpu = GpuIndex(node);
                graph_slices[gpu]->nodes++;
                graph_slices[gpu]->edges += h_row_offsets[node + 1] - h_row_offsets[node];
              }

              // Allocate data structures for gpu on host
              VertexId **slice_vertex_ids = new VertexId*[num_gpus];
              SizeT **slice_row_offsets = new SizeT*[num_gpus];
              VertexId **slice_column_indices = new VertexId*[num_gpus];
              EValue **slice_edge_values = new EValue*[num_gpus];
              EValue **slice_node_values = new EValue*[num_gpus];
              for (int gpu = 0; gpu < num_gpus; gpu++)
              {

                printf("GPU %d gets %lld vertices and %lld edges\n", gpu, (long long) graph_slices[gpu]->nodes, (long long) graph_slices[gpu]->edges);
                fflush (stdout);

                slice_row_offsets[gpu] = new SizeT[graph_slices[gpu]->nodes + 1];
                slice_row_offsets[gpu][0] = 0;

                slice_vertex_ids[gpu] = new VertexId[graph_slices[gpu]->nodes];
                slice_column_indices[gpu] = new VertexId[graph_slices[gpu]->edges];
                slice_edge_values[gpu] = new EValue[graph_slices[gpu]->edges];
                slice_node_values[gpu] = new EValue[graph_slices[gpu]->nodes];

                // Reset for construction
                graph_slices[gpu]->edges = 0;
              }

              printf("Done allocating gpu data structures on host\n");
              fflush (stdout);

              // Construct data structures for gpus on host
              for (VertexId node = 0; node < nodes; node++)
              {

                int gpu = GpuIndex(node);
                VertexId slice_row = GraphSliceRow(node);
                SizeT row_edges = h_row_offsets[node + 1] - h_row_offsets[node];

                memcpy(slice_column_indices[gpu] + slice_row_offsets[gpu][slice_row], h_column_indices + h_row_offsets[node], row_edges * sizeof(VertexId));

                memcpy(slice_vertex_ids[gpu] + slice_row_offsets[gpu][slice_row], h_vertex_ids + h_row_offsets[node], row_edges * sizeof(VertexId));

                if (WITH_VALUE)
                {
                  memcpy(slice_edge_values[gpu] + slice_row_offsets[gpu][slice_row], h_edge_values + h_row_offsets[node], row_edges * sizeof(EValue));
                  slice_node_values[gpu][slice_row] = h_node_values[node];
                }

                graph_slices[gpu]->edges += row_edges;
                slice_row_offsets[gpu][slice_row + 1] = graph_slices[gpu]->edges;

                // Mask in owning gpu
                for (int edge = 0; edge < row_edges; edge++)
                {
                  VertexId *ptr = slice_column_indices[gpu] + slice_row_offsets[gpu][slice_row] + edge;
                  VertexId owner = GpuIndex(*ptr);
                  (*ptr) |= (owner << ProblemType::GPU_MASK_SHIFT);
                }
              }

              printf("Done constructing gpu data structures on host\n");
              fflush(stdout);

              // Initialize data structures on GPU
              for (int gpu = 0; gpu < num_gpus; gpu++)
              {

                // Set device
                if (util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu), "CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                // Allocate and initialize d_row_offsets: copy and adjust by gpu offset

                printf("GPU %d row_offsets: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) (graph_slices[gpu]->nodes + 1),
                    (unsigned long long) (graph_slices[gpu]->nodes + 1) * sizeof(SizeT));

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_row_offsets, (graph_slices[gpu]->nodes + 1) * sizeof(SizeT)), "CsrProblem cudaMalloc d_row_offsets failed",
                    __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_node_values, (graph_slices[gpu]->nodes + 1) * sizeof(EValue)), "CsrProblem cudaMalloc d_row_offsets failed",
                    __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[gpu]->d_row_offsets, slice_row_offsets[gpu], (graph_slices[gpu]->nodes + 1) * sizeof(SizeT), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

                // Allocate and initialize d_column_indices

                printf("GPU %d column_indices: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) (graph_slices[gpu]->edges),
                    (unsigned long long) (graph_slices[gpu]->edges * sizeof(VertexId) * sizeof(SizeT)));

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_column_indices, graph_slices[gpu]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_column_indices failed",
                    __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[gpu]->d_column_indices, slice_column_indices[gpu], graph_slices[gpu]->edges * sizeof(VertexId), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_vertex_ids, graph_slices[gpu]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_vertex_ids failed", __FILE__,
                    __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(graph_slices[gpu]->d_vertex_ids, slice_vertex_ids[gpu], graph_slices[gpu]->nodes * sizeof(VertexId), cudaMemcpyHostToDevice),
                    "CsrProblem cudaMemcpy dverte_ids failed", __FILE__, __LINE__)) break;

                // Allocate and initialize d_edge_values
                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_edge_values, graph_slices[gpu]->edges * sizeof(VertexId)), "CsrProblem cudaMalloc d_edge_values failed",
                    __FILE__, __LINE__)) break;

                if (WITH_VALUE)
                {
                  if (retval = util::B40CPerror(cudaMemcpy(graph_slices[gpu]->d_edge_values, slice_edge_values[gpu], graph_slices[gpu]->edges * sizeof(EValue), cudaMemcpyHostToDevice),
                      "CsrProblem cudaMemcpy d_edge_values failed", __FILE__, __LINE__)) break;
                  if (retval = util::B40CPerror(cudaMemcpy(graph_slices[gpu]->d_node_values, slice_node_values[gpu], graph_slices[gpu]->nodes * sizeof(EValue), cudaMemcpyHostToDevice),
                      "CsrProblem cudaMemcpy d_node_values failed", __FILE__, __LINE__)) break;
                }
                // Cleanup host construction arrays
                if (slice_row_offsets[gpu]) delete slice_row_offsets[gpu];
                if (slice_vertex_ids[gpu]) delete slice_vertex_ids[gpu];
                if (slice_column_indices[gpu]) delete slice_column_indices[gpu];
                if (slice_edge_values[gpu]) delete slice_edge_values[gpu];
                if (slice_node_values[gpu]) delete slice_node_values[gpu];
              }
              if (retval) break;

              printf("Done initializing gpu data structures on gpus\n");
              fflush(stdout);

              if (slice_row_offsets) delete slice_row_offsets;
              if (slice_vertex_ids) delete slice_vertex_ids;
              if (slice_column_indices) delete slice_column_indices;
              if (slice_edge_values) delete slice_edge_values;
              if (slice_node_values) delete slice_node_values;
            }

          }
          while (0);

          delete[] h_vertex_ids;

          return retval;
        }

        /**
         * Performs any initialization work needed for this problem type.  Must be called
         * prior to each search
         */
        cudaError_t Reset(FrontierType frontier_type,	// The frontier type (i.e., edge/vertex/mixed)
            double queue_sizing,	// Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).  0.0 is unspecified.
            int src) //starting vertex
        {
          cudaError_t retval = cudaSuccess;
          printf("Starting vertex: %d\n", src);

          for (int gpu = 0; gpu < num_gpus; gpu++)
          {

            // Set device
            if (retval = util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu), "CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            //
            // Allocate output labels, preds, sigmas and deltas if necessary
            //
//            typename Program::Initialize init_functor;
//            _Program::Initialize(graph_slices[gpu]->nodes, graph_slices[gpu]->edges, graph_slices[gpu]->num_srcs, graph_slices[gpu]->srcs, graph_slices[gpu]->vertex_list, graph_slices[gpu]->frontier_queues.d_keys, graph_slices[gpu]->frontier_queues.d_values);
//            graph_slices[gpu]->vertex_list.init(graph_slices[gpu]->nodes, graph_slices[gpu]->edges, graph_slices[gpu]->frontier_queues.d_keys);
//            graph_slices[gpu]->gather_list.init();

//            if (!graph_slices[gpu]->d_labels)
//            {
//
//              printf("GPU %d labels: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->nodes,
//                  (unsigned long long) graph_slices[gpu]->nodes * sizeof(VertexId));
//
//              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_labels, graph_slices[gpu]->nodes * sizeof(VertexId)), "CsrProblem cudaMalloc d_labels failed", __FILE__,
//                  __LINE__)) return retval;
//            }
//
//            if (!graph_slices[gpu]->d_preds)
//            {
//
//              printf("GPU %d preds: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->nodes,
//                  (unsigned long long) graph_slices[gpu]->nodes * sizeof(VertexId));
//
//              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_preds, graph_slices[gpu]->nodes * sizeof(VertexId)), "CsrProblem cudaMalloc d_preds failed", __FILE__, __LINE__))
//                return retval;
//            }
//
//            if (!graph_slices[gpu]->d_sigmas)
//            {
//
//              printf("GPU %d sigmas: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->nodes,
//                  (unsigned long long) graph_slices[gpu]->nodes * sizeof(EValue));
//
//              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_sigmas, graph_slices[gpu]->nodes * sizeof(EValue)), "CsrProblem cudaMalloc d_sigmas failed", __FILE__, __LINE__))
//                return retval;
//            }
//
//            if (!graph_slices[gpu]->d_deltas)
//            {
//
//              printf("GPU %d labels: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->nodes,
//                  (unsigned long long) graph_slices[gpu]->nodes * sizeof(EValue));
//
//              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_deltas, graph_slices[gpu]->nodes * sizeof(EValue)), "CsrProblem cudaMalloc d_deltas failed", __FILE__, __LINE__))
//                return retval;
//            }
//
//            if (!graph_slices[gpu]->d_dists)
//            {
//
//              printf("GPU %d dists: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->nodes, (unsigned long long) graph_slices[gpu]->nodes * sizeof(int));
//
//              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_dists, graph_slices[gpu]->nodes * sizeof(int)), "CsrProblem cudaMalloc d_deltas failed", __FILE__, __LINE__))
//                return retval;
//            }
//
//            if (!graph_slices[gpu]->d_changed)
//            {
//
//              printf("GPU %d changed: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->nodes,
//                  (unsigned long long) graph_slices[gpu]->nodes * sizeof(int));
//
//              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_changed, graph_slices[gpu]->nodes * sizeof(int)), "CsrProblem cudaMalloc d_deltas failed", __FILE__, __LINE__))
//                return retval;
//            }

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

//            //
//            // Allocate visit flag for the entire graph if necessary
//            //
//
//            if (!graph_slices[gpu]->d_visit_flags)
//            {
//              printf("GPU %d visit flag: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->nodes,
//                  (unsigned long long) graph_slices[gpu]->nodes * sizeof(SizeT));
//
//              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_visit_flags, graph_slices[gpu]->nodes * sizeof(SizeT)), "CsrProblem cudaMalloc d_visit_flags failed", __FILE__,
//                  __LINE__)) return retval;
//
//            }

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
                new_predecessor_elements[0] = new_frontier_elements[0];
                new_predecessor_elements[1] = new_frontier_elements[1];
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

//                printf("GPU %d frontier queue[%d] (queue-sizing factor %.2fx): %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, i, queue_sizing,
//                    (unsigned long long) graph_slices[gpu]->frontier_elements[i], (unsigned long long) graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId));
//                fflush (stdout);

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

//                printf("GPU %d predecessor queue[%d] (queue-sizing factor %.2fx): %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, i, queue_sizing,
//                    (unsigned long long) graph_slices[gpu]->predecessor_elements[i], (unsigned long long) graph_slices[gpu]->predecessor_elements[i] * sizeof(VertexId));
//                fflush (stdout);

                if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->frontier_queues.d_values[i], graph_slices[gpu]->predecessor_elements[i] * sizeof(MiscType)),
                    "CsrProblem cudaMalloc frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
              }
            }

            //
            // Allocate duplicate filter mask if necessary (for multi-gpu)
            //

            if ((frontier_type == MULTI_GPU_FRONTIERS) && (!graph_slices[gpu]->d_filter_mask))
            {

//              printf("GPU %d_filter_mask flags: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) graph_slices[gpu]->frontier_elements[1],
//                  (unsigned long long) graph_slices[gpu]->frontier_elements[1] * sizeof(ValidFlag));

              if (retval = util::B40CPerror(cudaMalloc((void**) &graph_slices[gpu]->d_filter_mask, graph_slices[gpu]->frontier_elements[1] * sizeof(ValidFlag)),
                  "CsrProblem cudaMalloc d_filter_mask failed", __FILE__, __LINE__)) return retval;
            }

            //
            // Initialize labels and visited mask
            //
//            printf("num_srcs=%d, srcs[0]=%d\n", graph_slices[gpu]->num_srcs, graph_slices[gpu]->srcs[0]);

//            if (_Program::srcVertex() == SINGLE)
//            {
//              graph_slices[gpu]->num_src = 1;
//              graph_slices[gpu]->srcs = new int[1];
//              graph_slices[gpu]->srcs[0] = src;
//            }
//            else if (_Program::srcVertex() == ALL)
//            {
//              graph_slices[gpu]->num_srcs = graph_slices[gpu]->nodes;
//            }
//            else
//            {
//              printf("Invalid srcVertex!\n");
//              exit(0);
//            }

            //only 1 sourc is allowed now
            int srcs[1] = { src };
            _Program::Initialize(graph_slices[gpu]->nodes, graph_slices[gpu]->edges, 1,
                srcs, graph_slices[gpu]->vertex_list,
                graph_slices[gpu]->frontier_queues.d_keys,
                graph_slices[gpu]->frontier_queues.d_values);

            int memset_block_size = 256;
            int memset_grid_size_max = 32 * 1024;	// 32K CTAs
            int memset_grid_size;
//
////            graph_slices[gpu]->vertex_list.reset();
////            graph_slices[gpu]->gather_list.reset();
//
//            // Initialize d_labels elements to -1
//            memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
//            util::MemsetKernel<VertexId><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
//                graph_slices[gpu]->d_labels,
//                -1,
//                graph_slices[gpu]->nodes);
//
//            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;
//
//            // Initialize d_dists elements to 1000000
//            memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
//            util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
//                graph_slices[gpu]->d_dists,
//                100000000,
//                graph_slices[gpu]->nodes);
//
//            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;
//
//            // Initialize d_changed elements to 0
//            memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
//            util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
//                graph_slices[gpu]->d_changed,
//                0,
//                graph_slices[gpu]->nodes);
//
//            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;

            // Initialize d_visited_mask elements to 0
            memset_grid_size = B40C_MIN(memset_grid_size_max, (visited_mask_elements + memset_block_size - 1) / memset_block_size);

            util::MemsetKernel<VisitedMask><<<memset_grid_size, memset_block_size, 0,
                graph_slices[gpu]->stream>>>(graph_slices[gpu]->d_visited_mask, 0, visited_mask_elements);

            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;

//            // Initialize d_preds elements to -1
//            memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
//            util::MemsetKernel<VertexId><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
//                graph_slices[gpu]->d_preds,
//                -1,
//                graph_slices[gpu]->nodes);
//
//            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;
//
//            // Initialize d_sigmas elements to 0
//            memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
//            util::MemsetKernel<EValue><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
//                graph_slices[gpu]->d_sigmas,
//                0.0f,
//                graph_slices[gpu]->nodes);
//
//            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;
//
//            // Initialize d_deltas elements to 0
//            memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
//            util::MemsetKernel<EValue><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
//                graph_slices[gpu]->d_deltas,
//                0.0f,
//                graph_slices[gpu]->nodes);
//
//            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;
//
//            // Initialize d_visit_flags elements to 0
//            memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
//            util::MemsetKernel<SizeT><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
//                graph_slices[gpu]->d_visit_flags,
//                0,
//                graph_slices[gpu]->nodes);
//
//            if (retval = util::B40CPerror(cudaThreadSynchronize(), "MemsetKernel failed", __FILE__, __LINE__)) return retval;

          }

          return retval;
        }
      };

    } // namespace bc
  } // namespace graph
} // namespace b40c
