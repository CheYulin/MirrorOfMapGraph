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
 * BC enactor
 ******************************************************************************/

#pragma once

#include <stdlib.h>

#include <config.h>

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/global_barrier.cuh>

#include <b40c/graph/GASengine/problem_type.cuh>
#include <b40c/graph/GASengine/csr_problem.cuh>
#include <b40c/graph/GASengine/enactor_base.cuh>

//#include <b40c/graph/GASengine/vertex_centric/filter_atomic/kernel.cuh>
//#include <b40c/graph/GASengine/vertex_centric/filter_atomic/kernel_policy.cuh>
#include <b40c/graph/GASengine/vertex_centric/gather/kernel.cuh>
#include <b40c/graph/GASengine/vertex_centric/gather/kernel_policy.cuh>
#include <b40c/graph/GASengine/vertex_centric/expand_atomic/kernel.cuh>
#include <b40c/graph/GASengine/vertex_centric/expand_atomic/kernel_policy.cuh>
#include <b40c/graph/GASengine/vertex_centric/contract_atomic/kernel.cuh>
#include <b40c/graph/GASengine/vertex_centric/contract_atomic/kernel_policy.cuh>
#include <b40c/graph/GASengine/vertex_centric/backward_contract_atomic/kernel.cuh>
#include <b40c/graph/GASengine/vertex_centric/backward_contract_atomic/kernel_policy.cuh>
#include <b40c/graph/GASengine/vertex_centric/backward_sum_atomic/kernel.cuh>
#include <b40c/graph/GASengine/vertex_centric/backward_sum_atomic/kernel_policy.cuh>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

using namespace std;

namespace b40c
{
  namespace graph
  {
    namespace GASengine
    {

      /**
       * Vertex-centric BC enactor
       *
       * For each BFS iteration, visited/duplicate vertices are culled from
       * the incoming edge-frontier in global memory.  The remaining vertices are
       * compacted to a vertex-frontier in global memory.  Then these
       * vertices are read back in and expanded to construct the outgoing
       * edge-frontier in global memory.
       */
      template<bool INSTRUMENT> // Whether or not to collect per-CTA clock-count statistics
      class EnactorVertexCentric: public EnactorBase
      {
        //---------------------------------------------------------------------
        // Members
        //---------------------------------------------------------------------

      protected:

        /**
         * CTA duty kernel stats
         */
        util::KernelRuntimeStatsLifetime expand_kernel_stats;
        util::KernelRuntimeStatsLifetime filter_kernel_stats;
        util::KernelRuntimeStatsLifetime contract_kernel_stats;
        util::KernelRuntimeStatsLifetime backward_sum_kernel_stats;
        util::KernelRuntimeStatsLifetime backward_contract_kernel_stats;

        unsigned long long total_runtimes; // Total time "worked" by each cta
        unsigned long long total_lifetimes; // Total time elapsed by each cta
        unsigned long long total_queued;

        /**
         * Throttle state.  We want the host to have an additional BFS iteration
         * of kernel launches queued up for for pipeline efficiency (particularly on
         * Windows), so we keep a pinned, mapped word that the traversal kernels will
         * signal when done.
         */
        volatile int *done;
        int *d_done;
        cudaEvent_t throttle_event;
        Config cfg;

        /**
         * Mechanism for implementing software global barriers from within
         * a single grid invocation
         */
        util::GlobalBarrierLifetime global_barrier;

        /**
         * Current iteration (mapped into GPU space so that it can
         * be modified by multi-iteration kernel launches)
         */
        volatile long long *iteration;
        long long *d_iteration;

        //---------------------------------------------------------------------
        // Methods
        //---------------------------------------------------------------------

      protected:

        /**
         * Prepare enactor for search.  Must be called prior to each search.
         */
        template<typename CsrProblem>
        cudaError_t Setup(CsrProblem &csr_problem, int expand_grid_size,
            int contract_grid_size, int iter)
        {
          typedef typename CsrProblem::SizeT SizeT;
          typedef typename CsrProblem::VertexId VertexId;
          typedef typename CsrProblem::VisitedMask VisitedMask;

          cudaError_t retval = cudaSuccess;

          do
          {

            // Make sure host-mapped "done" is initialized
            if (!done)
            {
              int flags = cudaHostAllocMapped;

              // Allocate pinned memory for done
              if (retval = util::B40CPerror(
                  cudaHostAlloc((void **) &done, sizeof(int) * 1, flags),
                  "EnactorContractExpand cudaHostAlloc done failed",
                  __FILE__, __LINE__))
                break;

              // Map done into GPU space
              if (retval =
                  util::B40CPerror(
                      cudaHostGetDevicePointer((void **) &d_done,
                          (void *) done, 0),
                      "EnactorContractExpand cudaHostGetDevicePointer done failed",
                      __FILE__, __LINE__))
                break;

              // Create throttle event
              if (retval =
                  util::B40CPerror(
                      cudaEventCreateWithFlags(&throttle_event,
                          cudaEventDisableTiming),
                      "EnactorContractExpand cudaEventCreateWithFlags throttle_event failed",
                      __FILE__, __LINE__))
                break;
            }

            // Make sure host-mapped "iteration" is initialized
            if (!iteration)
            {

              int flags = cudaHostAllocMapped;

              // Allocate pinned memory
              if (retval = util::B40CPerror(
                  cudaHostAlloc((void **) &iteration,
                      sizeof(long long) * 1, flags),
                  "EnactorContractExpand cudaHostAlloc iteration failed",
                  __FILE__, __LINE__))
                break;

              // Map into GPU space
              if (retval =
                  util::B40CPerror(
                      cudaHostGetDevicePointer((void **) &d_iteration,
                          (void *) iteration, 0),
                      "EnactorContractExpand cudaHostGetDevicePointer iteration failed",
                      __FILE__, __LINE__))
                break;
            }

            // Make sure software global barriers are initialized
            if (retval = global_barrier.Setup(expand_grid_size))
              break;

            // Make sure our runtime stats are initialized
            if (retval = expand_kernel_stats.Setup(expand_grid_size))
              break;
            if (retval = contract_kernel_stats.Setup(contract_grid_size))
              break;
//            if (retval = filter_kernel_stats.Setup(filter_grid_size)) break;
            if (retval = backward_sum_kernel_stats.Setup(expand_grid_size))
              break;
            if (retval = backward_contract_kernel_stats.Setup(
                contract_grid_size))
              break;

            // Reset statistics
            iteration[0] = iter;
            total_runtimes = 0;
            total_lifetimes = 0;
            total_queued = 0;
            done[0] = -1;

            // Single-gpu graph slice
            typename CsrProblem::GraphSlice *graph_slice =
                csr_problem.graph_slices[0];

            // Bind bitmask texture
            int bytes = (graph_slice->nodes + 8 - 1) / 8;
            cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
            if (retval =
                util::B40CPerror(
                    cudaBindTexture(0,
                        vertex_centric::contract_atomic::BitmaskTex<
                            VisitedMask>::ref,
                        graph_slice->d_visited_mask, bitmask_desc,
                        bytes),
                    "EnactorVertexCentric cudaBindTexture bitmask_tex_ref failed",
                    __FILE__, __LINE__))
              break;

//            if (retval = util::B40CPerror(cudaBindTexture(0, vertex_centric::filter_atomic::BitmaskTex<VisitedMask>::ref, graph_slice->d_visited_mask, bitmask_desc, bytes),
//                                          "EnactorVertexCentric cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

            // Bind row-offsets texture
            cudaChannelFormatDesc row_offsets_desc =
                cudaCreateChannelDesc<SizeT>();
            if (retval =
                util::B40CPerror(
                    cudaBindTexture(0,
                        vertex_centric::expand_atomic::RowOffsetTex<
                            SizeT>::ref,
                        graph_slice->d_row_offsets,
                        row_offsets_desc,
                        (graph_slice->nodes + 1) * sizeof(SizeT)),
                    "EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed",
                    __FILE__, __LINE__))
              break;

            if (retval =
                util::B40CPerror(
                    cudaBindTexture(0,
                        vertex_centric::backward_sum_atomic::RowOffsetTex<
                            SizeT>::ref,
                        graph_slice->d_row_offsets,
                        row_offsets_desc,
                        (graph_slice->nodes + 1) * sizeof(SizeT)),
                    "EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed",
                    __FILE__, __LINE__))
              break;

            // Bind column-offsets texture
            //            if (retval = util::B40CPerror(
            //                cudaBindTexture(0, vertex_centric::gather::RowOffsetTex<SizeT>::ref, graph_slice->d_column_offsets, row_offsets_desc, (graph_slice->nodes + 1) * sizeof(SizeT)),
            //                "EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

          }
          while (0);

          return retval;
        }

      public:

        /**
         * Constructor
         */
        EnactorVertexCentric(Config cfg, bool DEBUG = false) :
            cfg(cfg), EnactorBase(EDGE_FRONTIERS, DEBUG), iteration(NULL), d_iteration(
                NULL), total_queued(0), done(NULL), d_done(NULL)
        {
        }

        /**
         * Destructor
         */
        virtual ~EnactorVertexCentric()
        {
          if (iteration)
          {
            util::B40CPerror(cudaFreeHost((void *) iteration),
                "EnactorVertexCentric cudaFreeHost iteration failed",
                __FILE__, __LINE__);
          }
          if (done)
          {
            util::B40CPerror(cudaFreeHost((void *) done),
                "EnactorVertexCentric cudaFreeHost done failed", __FILE__,
                __LINE__);

            util::B40CPerror(cudaEventDestroy(throttle_event),
                "EnactorVertexCentric cudaEventDestroy throttle_event failed",
                __FILE__, __LINE__);
          }
        }

        /**
         * Obtain statistics about the last BFS search enacted
         */
        template<typename VertexId>
        void GetStatistics(long long &total_queued, VertexId &search_depth,
            double &avg_duty)
        {
          cudaThreadSynchronize();

          total_queued = this->total_queued;
          search_depth = this->iteration[0] - 1;

          avg_duty =
              (total_lifetimes > 0) ?
                                      double(total_runtimes) / total_lifetimes :
                                      0.0;
        }

        /**
         * Enacts a breadth-first-search on the specified graph problem. Invokes
         * new expansion and contraction grid kernels for each BFS iteration.
         *
         * @return cudaSuccess on success, error enumeration otherwise
         */
        template<typename ExpandPolicy, typename GatherPolicy,
            typename ContractPolicy, typename Program, typename CsrProblem>
        cudaError_t EnactIterativeSearch(CsrProblem &csr_problem, char* source_file_name,
            typename CsrProblem::SizeT* h_row_offsets)
        {
          typedef typename CsrProblem::SizeT SizeT;
          typedef typename CsrProblem::VertexId VertexId;
          typedef typename CsrProblem::EValue EValue;
          typedef typename CsrProblem::VisitedMask VisitedMask;

//          int src_node = cfg.getParameter<int>("src");
//          char* source_file_name = cfg.getParameter<char*>("source_file_name");
//          double max_queue_sizing = cfg.getParameter<double>("max_queue_sizing");

          DEBUG = cfg.getParameter<int>("verbose");
          cudaError_t retval = cudaSuccess;

          // Determine grid size(s)
          int expand_occupancy = ExpandPolicy::CTA_OCCUPANCY;
          int expand_grid_size = MaxGridSize(expand_occupancy);

//          int filter_occupancy = FilterPolicy::CTA_OCCUPANCY;
//          int filter_grid_size = MaxGridSize(filter_occupancy, max_grid_size);

          int gather_occupancy = GatherPolicy::CTA_OCCUPANCY;
          int gather_grid_size = MaxGridSize(gather_occupancy);

          int contract_occupancy = ContractPolicy::CTA_OCCUPANCY;
          int contract_grid_size = MaxGridSize(contract_occupancy);

          // Single-gpu graph slice
          typename CsrProblem::GraphSlice *graph_slice =
              csr_problem.graph_slices[0];

//          int num_src = 1;
//          int* srcs;
          double max_queue_sizing = cfg.getParameter<double>("max_queue_sizing");
//          int init_num_elements;
//          if (Program::srcVertex() == SINGLE)
//          {
//            init_num_elements = 1;
//            int src_node = cfg.getParameter<int>("src");
//            int origin = cfg.getParameter<int>("origin");
//            printf("origin: %d\n", origin);
//            const int max_src_num = 100;
//
//            if (strcmp(source_file_name, ""))
//            {
//              //TODO random sources
//              if (strcmp(source_file_name, "RANDOM"))
//              {
//                num_src = cfg.getParameter<int>("num_src");
//                srcs = new int[num_src];
//                printf("Using %d random starting vertices!\n", num_src);
//                srand (time(NULL));
//                int count = 0;
//                while (count < num_src)
//                {
//                  int tmp_src = rand() % graph_slice->nodes;
//                  if (graph_slice->row_offsets[tmp_src + 1] - graph_slice->row_offsets[tmp_src] > 0)
//                  {
//
//                  }
//                }
//
//              }
//              else
//              {
//                printf("Using source file!\n");
//                FILE* src_file;
//                if ((src_file = fopen(source_file_name, "r")) == NULL)
//                {
//                  printf("Source file open error!\n");
//                  exit(0);
//                }
//
//                srcs = new int[max_src_num];
//                for (num_src = 0; num_src < max_src_num; num_src++)
//                {
//                  if ((fscanf(src_file, "%d\n", &srcs[num_src]) != EOF))
//                  {
//                    if (origin == 1)
//                      srcs[num_src]--; //0-based index
//                  }
//                  else
//                    break;
//                }
//              }
//
//            }
//            else
//            {
//              printf("Single source vertex!\n");
//              num_src = 1;
//              srcs = new int[1];
//              srcs[0] = src_node;
//              if (origin == 1)
//                srcs[0]--;
//            }
//          }
//
//          else if (Program::srcVertex() == ALL)
//          {
//            srcs = new int[1]; //dummy not used
//            srcs[0] = -1; // dummy not used
//            num_src = 1;
//            init_num_elements = graph_slice->nodes;
//          }
//          else
//          {
//            printf("Invalid src vertex!\n");
//            exit(0);
//          }

          for (int outer_iter = 0; outer_iter < graph_slice->outer_iter_num; outer_iter++)
          {
            for (int iter = 0; iter < graph_slice->num_src; ++iter)
            {
              // Reset data
              if (retval = csr_problem.Reset(GetFrontierType(), max_queue_sizing,
                  graph_slice->srcs[iter]))
                break;

              if (retval = Setup(csr_problem, expand_grid_size,
                  contract_grid_size, 0))
                break;

              SizeT queue_length;
              VertexId queue_index = 0; // Work stealing/queue index
              int selector = 0;

              cudaEvent_t start, stop;
              cudaEventCreate(&start);
              cudaEventCreate(&stop);

              double startcontract, endcontract;
              double startexpand, endexpand;

              double elapsedcontract = 0.0;
              double elapsedexpand = 0.0;

              cudaEventRecord(start);
              double startTime = omp_get_wtime();

              // Forward phase BC iterations
              while (done[0] < 0)
              //            for (int i = 0; i < 1; i++)
              {
                if (DEBUG)
                  printf("Iteration: %lld\n", (long long) iteration[0]);
                if (DEBUG)
                  startcontract = omp_get_wtime();
                //
                // Contraction
                //
                vertex_centric::contract_atomic::Kernel<ContractPolicy, Program><<<
                    contract_grid_size, ContractPolicy::THREADS>>>(
                    graph_slice->srcs[iter],
                    iteration[0],
                    graph_slice->init_num_elements,                  // initial num_elements, for BFS it is 1
                    queue_index,                  // queue counter index
                    queue_index,                  // steal counter index
                    1,                  // number of GPUs
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector ^ 1],                  // filtered edge frontier in
                    graph_slice->frontier_queues.d_keys[selector],                  // vertex frontier out
                    graph_slice->frontier_queues.d_values[selector ^ 1],                  // predecessor in
                    graph_slice->vertex_list,
                    graph_slice->edge_list,
                    graph_slice->d_labels,                  // source distance out
                    graph_slice->d_preds,                  // prtedecessor out
                    graph_slice->d_sigmas, graph_slice->d_dists,
                    graph_slice->d_changed, graph_slice->d_visited_mask,
                    this->work_progress,
                    graph_slice->frontier_elements[selector ^ 1],                  // max filtered edge frontier vertices
                    graph_slice->frontier_elements[selector],                  // max vertex frontier vertices
                    this->contract_kernel_stats);

                if (DEBUG
                    && (retval = util::B40CPerror(cudaThreadSynchronize(),
                        "contract_atomic::Kernel failed ", __FILE__,
                        __LINE__)))
                  break;
                cudaEventQuery(throttle_event);                  // give host memory mapped visibility to GPU updates

                if (DEBUG)
                {
                  cudaDeviceSynchronize();
                  endcontract = omp_get_wtime();
                  elapsedcontract += endcontract - startcontract;
                }

                queue_index++;
                selector ^= 1;

                if (DEBUG)
                {
                  if (retval = work_progress.GetQueueLength(queue_index,
                      queue_length))
                    break;
                  printf("queue_length after contraction: %lld\n",
                      (long long) queue_length);

//                VertexId* test_vid = new VertexId[graph_slice->nodes];
//                cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                printf("Frontier after contraction: ");
//                for (int i = 0; i < queue_length; ++i)
//                {
//                  printf("%d, ", test_vid[i]);
//                }
//                printf("\n");
//                delete[] test_vid;

//                EValue *test_vid2 = new EValue[graph_slice->nodes];
//                cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                printf("d_dists after contract: ");
//                for (int i = 0; i < graph_slice->nodes; ++i)
//                {
//                  printf("%d, ", test_vid2[i]);
//                }
//                printf("\n");
//                delete[] test_vid2;

//                test_vid2 = new EValue[graph_slice->nodes];
//                cudaMemcpy(test_vid2, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                printf("d_min_dists after contract: ");
//                for (int i = 0; i < graph_slice->nodes; ++i)
//                {
//                  printf("%d, ", test_vid2[i]);
//                }
//                printf("\n");
//                delete[] test_vid2;
                }

                // Throttle
                if (iteration[0] & 1)
                {
                  if (retval =
                      util::B40CPerror(cudaEventRecord(throttle_event),
                          "EnactorVertexCentric cudaEventRecord throttle_event failed",
                          __FILE__, __LINE__))
                    break;
                }
                else
                {
                  if (retval =
                      util::B40CPerror(
                          cudaEventSynchronize(throttle_event),
                          "EnactorVertexCentric cudaEventSynchronize throttle_event failed",
                          __FILE__, __LINE__))
                    break;
                };

                // Check if done
                if (done[0] == 0)
                  break;

                if (Program::gatherOverEdges() == GATHER_IN_EDGES)
                {

//                if (retval = util::B40CPerror(cudaMemcpy(graph_slice->d_gather_results, graph_slice->d_dists, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToDevice),
//                    "CsrProblem cudaMemcpy d_gather_results failed", __FILE__, __LINE__)) break;

                  //
                  //gather only: gather stage of the GASengine, does not modify frontier
                  //
                  vertex_centric::gather::Kernel<GatherPolicy, Program><<<
                      gather_grid_size, GatherPolicy::THREADS>>>(
                      queue_index,                    // queue counter index
                      queue_index,                    // steal counter index
                      1,                    // number of GPUs
                      d_done,
                      graph_slice->frontier_queues.d_keys[selector ^ 1],                    // vertex frontier in
                      graph_slice->frontier_queues.d_keys[selector],                    // edge frontier out
//                  graph_slice->frontier_queues.d_values[selector],// predecessor out
                      graph_slice->d_row_indices,                    //pass in the CSC graph to gather for destination vertices
                      graph_slice->d_column_offsets,                    //pass in the CSC graph to gather for destination vertices
                      graph_slice->vertex_list, //
                      graph_slice->edge_list, //
                      //                  graph_slice->d_visit_flags,
                      this->work_progress,
                      graph_slice->frontier_elements[selector ^ 1],                    // max vertex frontier vertices
                      graph_slice->frontier_elements[selector],                    // max edge frontier vertices
                      this->expand_kernel_stats);

                  if (DEBUG
                      && (retval = util::B40CPerror(
                          cudaThreadSynchronize(),
                          "gather::Kernel failed ", __FILE__,
                          __LINE__)))
                    break;
                  cudaEventQuery(throttle_event);                    // give host memory mapped visibility to GPU updates

                  //                queue_index++;
                  //                selector ^= 1;

                  if (DEBUG)
                  {
                    if (work_progress.GetQueueLength(queue_index,
                        queue_length))
                      break;
                    total_queued += queue_length;

//                  if (DEBUG) printf("queue_length after gather: %lld\n", (long long) queue_length);

//                  EValue *test_vid2 = new EValue[graph_slice->nodes];
//                  cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                  printf("d_dists after gather: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%f, ", test_vid2[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid2;
//
//                  test_vid2 = new EValue[graph_slice->nodes];
//                  cudaMemcpy(test_vid2, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                  printf("d_gather_results after gather: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%f, ", test_vid2[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid2;

//                  VertexId* test_vid = new VertexId[graph_slice->nodes];
//                  cudaMemcpy(test_vid, graph_slice->d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                  printf("changed after gather: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%d, ", test_vid[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid;
                  }
                }

                if (DEBUG)
                  startexpand = omp_get_wtime();

                if (Program::applyOverEdges() == APPLY_FRONTIER)
                {
                  //
                  //apply stage
                  //
                  vertex_centric::gather::apply<GatherPolicy, Program><<<
                      gather_grid_size, GatherPolicy::THREADS>>>(
                      iteration[0], queue_index, this->work_progress,
                      graph_slice->frontier_queues.d_keys[selector ^ 1],
                      graph_slice->vertex_list, graph_slice->edge_list);

//                if (DEBUG)
//                {
//                  int num_changed;
//                  thrust::device_ptr<int> changed_ptr = thrust::device_pointer_cast(graph_slice->vertex_list.d_changed);
//                  num_changed = thrust::reduce(changed_ptr, changed_ptr + graph_slice->nodes);
//                  printf("num_changed=%d\n", num_changed);
//                }
                }

                if (Program::postApplyOverEdges() == POST_APPLY_FRONTIER)
                {

                  //
                  //                //reset dists and gather_results
                  //                //
                  vertex_centric::gather::reset_gather_result<GatherPolicy,
                      Program><<<gather_grid_size, GatherPolicy::THREADS>>>(
                      iteration[0], queue_index, this->work_progress,
                      graph_slice->frontier_queues.d_keys[selector ^ 1],
                      graph_slice->vertex_list,
                      graph_slice->edge_list,
                      graph_slice->d_visited_mask);

                  if (DEBUG
                      && (retval = util::B40CPerror(
                          cudaThreadSynchronize(),
                          "gather::reset_changed Kernel failed ",
                          __FILE__, __LINE__)))
                    break;

                  if (DEBUG)
                  {
//                  EValue *test_vid2 = new EValue[graph_slice->nodes];
//                  cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                  printf("d_dists after apply: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%f, ", test_vid2[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid2;
//
//                  VertexId *test_vid = new VertexId[graph_slice->nodes];
//                  cudaMemcpy(test_vid, graph_slice->vertex_list.d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                  printf("changed after apply: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%d, ", test_vid[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid;
                  }
                }
                else if (Program::postApplyOverEdges() == POST_APPLY_ALL)
                {
                  vertex_centric::gather::reset_gather_result<GatherPolicy,
                      Program><<<gather_grid_size, GatherPolicy::THREADS>>>(
                      iteration[0], graph_slice->nodes,
                      graph_slice->vertex_list, graph_slice->edge_list,
                      graph_slice->d_visited_mask);

                  if (DEBUG)
                  {
//                  EValue *test_vid2 = new EValue[graph_slice->nodes];
//                  cudaMemcpy(test_vid2, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                  printf("d_dists after apply: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%f, ", test_vid2[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid2;
//
//                  VertexId *test_vid = new VertexId[graph_slice->nodes];
//                  cudaMemcpy(test_vid, graph_slice->vertex_list.d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                  printf("changed after apply: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%d, ", test_vid[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid;
                  }

                }

                if (Program::expandOverEdges() == EXPAND_OUT_EDGES)
                {
                  //
                  // Expansion
                  //
                  vertex_centric::expand_atomic::Kernel<ExpandPolicy, Program><<<
                      expand_grid_size, ExpandPolicy::THREADS>>>(
                      iteration[0],
                      queue_index,                    // queue counter index
                      queue_index,                    // steal counter index
                      1,                    // number of GPUs
                      d_done,
                      graph_slice->frontier_queues.d_keys[selector ^ 1],                    // vertex frontier in
                      graph_slice->frontier_queues.d_keys[selector],                    // edge frontier out
                      graph_slice->frontier_queues.d_values[selector],                    // predecessor out
                      graph_slice->vertex_list,                    //
                      graph_slice->edge_list,
                      graph_slice->d_column_indices,
                      graph_slice->d_row_offsets, this->work_progress,
                      graph_slice->frontier_elements[selector ^ 1],                    // max vertex frontier vertices
                      graph_slice->frontier_elements[selector],                    // max edge frontier vertices
                      this->expand_kernel_stats);

                  if (DEBUG
                      && (retval = util::B40CPerror(
                          cudaThreadSynchronize(),
                          "expand_atomic::Kernel failed ", __FILE__,
                          __LINE__)))
                    break;
                  cudaEventQuery(throttle_event);                    // give host memory mapped visibility to GPU updates

                  if (DEBUG)
                  {
                    cudaDeviceSynchronize();
                    endexpand = omp_get_wtime();
                    elapsedexpand += endexpand - startexpand;
                  }
                }

                queue_index++;
                selector ^= 1;
                iteration[0]++;

                //              if (work_progress.GetQueueLength(queue_index, queue_length)) break;
                //              if (queue_length > max_queue_sizing * graph_slice->edges)
                //              {
                //                printf("Error: queue size not large enough!\n");
                //                retval = (cudaError_t) 1;
                //                break;
                //              }

                if (INSTRUMENT && DEBUG)
                //              if(1)
                {
                  if (work_progress.GetQueueLength(queue_index, queue_length))
                    break;

                  total_queued += queue_length;
                  printf("queue_length after expansion: %lld\n",
                      (long long) queue_length);

//                VertexId* test_vid = new VertexId[queue_length];
//                cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                printf("Frontier after expansion: ");
//                for (int i = 0; i < queue_length; ++i)
//                {
//                  printf("%d, ", test_vid[i]);
//                }
//                printf("\n");
//                delete[] test_vid;

//                test_vid = new VertexId[graph_slice->nodes];
//                cudaMemcpy(test_vid, graph_slice->vertex_list.d_dists, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                printf("d_dists after expansion: ");
//                for (int i = 0; i < graph_slice->nodes; ++i)
//                {
//                  printf("%d, ", test_vid[i]);
//                }
//                printf("\n");
//                delete[] test_vid;

//                test_vid = new VertexId[queue_length];
//                cudaMemcpy(test_vid, graph_slice->frontier_queues.d_values[selector ^ 1], queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                printf("d_predecesor after expansion: ");
//                for (int i = 0; i < queue_length; ++i)
//                {
//                  printf("%d, ", test_vid[i]);
//                }
//                printf("\n");
//                delete[] test_vid;
                }

                //              if (DEBUG) printf("\n%lld", (long long) iteration[0]);

                // Check if done
                if (done[0] == 0)
                  break;

              }

              cudaEventRecord(stop);
              cudaEventSynchronize(stop);
              cudaDeviceSynchronize();
              double endTime = omp_get_wtime();
              double elapsed_wall = (omp_get_wtime() - startTime) * 1000;

              float elapsed;
              cudaEventElapsedTime(&elapsed, start, stop);
              std::cout << "Kernel time took: " << elapsed << " ms" << std::endl;
              std::cout << "Wall time took: " << elapsed_wall << " ms"
                  << std::endl;
              std::cout << "Contract time took: " << elapsedcontract * 1000
                  << " ms" << std::endl;
              std::cout << "Expand time took: " << elapsedexpand * 1000 << " ms"
                  << std::endl;
              // Compute nodes and edges visited
              SizeT edges_visited = 0;
              SizeT nodes_visited = 0;
              VertexId* h_dists = new VertexId[graph_slice->nodes];
              cudaMemcpy(h_dists, graph_slice->d_labels,
                  graph_slice->nodes * sizeof(VertexId),
                  cudaMemcpyDeviceToHost);
              for (VertexId i = 0; i < graph_slice->nodes; i++)
              {
                if (h_dists[i] > -1)
                {
                  nodes_visited++;
                  edges_visited += h_row_offsets[i + 1] - h_row_offsets[i];
                }
                //              else
                //                printf("h_dists[%d]=%d\n", i, h_dists[i]);
              }
              delete[] h_dists;
              std::cout << "visited edges: " << edges_visited << std::endl;
              std::cout << "M-Edges / sec: "
                  << (double) edges_visited / (elapsed_wall * 1000.f)
                  << std::endl;
              printf("Total iteration: %lld\n", (long long) iteration[0]);

              // Check if any of the frontiers overflowed due to redundant expansion
              bool overflowed = false;
              retval = work_progress.CheckOverflow<SizeT>(overflowed);
              if (overflowed)
              {
                retval =
                    util::B40CPerror(cudaErrorInvalidConfiguration,
                        "Frontier queue overflow.  Please increase queue-sizing factor. ",
                        __FILE__, __LINE__);
                retval = (cudaError_t) 1;
              }
              std::cout << "retval: " << retval << std::endl;

            }
          }

//          delete[] srcs;

          return retval;
        }

        /**
         * Enacts a breadth-first-search on the specified graph problem. Invokes
         * new expansion and contraction grid kernels for each BFS iteration.
         *
         * @return cudaSuccess on success, error enumeration otherwise
         */
        template<typename CsrProblem, typename Program>
        cudaError_t EnactIterativeSearch(CsrProblem &csr_problem,
            char* source_file_name,
            typename CsrProblem::SizeT* h_row_offsets)
        {
          typedef typename CsrProblem::VertexId VertexId;
          typedef typename CsrProblem::SizeT SizeT;

          // GF100
          if (this->cuda_props.device_sm_version >= 200)
          {

            // Expansion kernel config
            typedef vertex_centric::expand_atomic::KernelPolicy<Program,
                typename CsrProblem::ProblemType, 200, // CUDA_ARCH
                INSTRUMENT, // INSTRUMENT
                1, // CTA_OCCUPANCY
                7, // LOG_THREADS
                0, // LOG_LOAD_VEC_SIZE
                0, // LOG_LOADS_PER_TILE
                5, // LOG_RAKING_THREADS
                util::io::ld::cg, // QUEUE_READ_MODIFIER,
                util::io::ld::NONE, // COLUMN_READ_MODIFIER,
                util::io::ld::NONE, // EDGE_VALUES_READ_MODIFIER,
                util::io::ld::cg, // ROW_OFFSET_ALIGNED_READ_MODIFIER,
                util::io::ld::NONE, // ROW_OFFSET_UNALIGNED_READ_MODIFIER,
                util::io::st::cg, // QUEUE_WRITE_MODIFIER,
                false, // WORK_STEALING
                32, // WARP_GATHER_THRESHOLD
                128 * 4, // CTA_GATHER_THRESHOLD,
                7> // LOG_SCHEDULE_GRANULARITY
            ExpandPolicy;

            // Gather kernel config
            typedef vertex_centric::gather::KernelPolicy<Program,
                typename CsrProblem::ProblemType, 200, // CUDA_ARCH
                INSTRUMENT, // INSTRUMENT
                1, // CTA_OCCUPANCY
                7, // LOG_THREADS
                0, // LOG_LOAD_VEC_SIZE
                0, // LOG_LOADS_PER_TILE
                5, // LOG_RAKING_THREADS
                util::io::ld::cg, // QUEUE_READ_MODIFIER,
                util::io::ld::NONE, // COLUMN_READ_MODIFIER,
                util::io::ld::NONE, // EDGE_VALUES_READ_MODIFIER,
                util::io::ld::cg, // ROW_OFFSET_ALIGNED_READ_MODIFIER,
                util::io::ld::NONE, // ROW_OFFSET_UNALIGNED_READ_MODIFIER,
                util::io::st::cg, // QUEUE_WRITE_MODIFIER,
                false, // NO WORK_STEALING : changed from true
                32, // WARP_GATHER_THRESHOLD
                128 * 4, // CTA_GATHER_THRESHOLD,
                7> // LOG_SCHEDULE_GRANULARITY
            GatherPolicy;

//            // Filter kernel config
//            typedef vertex_centric::filter_atomic::KernelPolicy<typename CsrProblem::ProblemType, 200, // CUDA_ARCH
//                    INSTRUMENT, // INSTRUMENT
//                    0, // SATURATION_QUIT
//                    8, // CTA_OCCUPANCY
//                    7, // LOG_THREADS
//                    1, // LOG_LOAD_VEC_SIZE
//                    1, // LOG_LOADS_PER_TILE
//                    5, // LOG_RAKING_THREADS
//                    util::io::ld::NONE, // QUEUE_READ_MODIFIER,
//                    util::io::st::NONE, // QUEUE_WRITE_MODIFIER,
//                    false, // WORK_STEALING
//                    9 > // LOG_SCHEDULE_GRANULARITY
//                    FilterPolicy;

            // Contraction kernel config
            typedef vertex_centric::contract_atomic::KernelPolicy<Program,
                typename CsrProblem::ProblemType, 200, // CUDA_ARCH
                INSTRUMENT, // INSTRUMENT
                0, // SATURATION_QUIT
                true, // DEQUEUE_PROBLEM_SIZE
                8, // CTA_OCCUPANCY
                7, // LOG_THREADS
                1, // LOG_LOAD_VEC_SIZE
                0, // LOG_LOADS_PER_TILE
                5, // LOG_RAKING_THREADS
                util::io::ld::NONE, // QUEUE_READ_MODIFIER,
                util::io::st::NONE, // QUEUE_WRITE_MODIFIER,
                false, // WORK_STEALING
                -1, // END_BITMASK_CULL 0 to never perform bitmask filtering, -1 to always perform bitmask filtering
                8> // LOG_SCHEDULE_GRANULARITY
            ContractPolicy;

            return EnactIterativeSearch<ExpandPolicy, GatherPolicy,
                ContractPolicy, Program>(csr_problem, source_file_name, h_row_offsets);
          }

          printf("Not yet tuned for this architecture\n");
          return cudaErrorInvalidDeviceFunction;
        }
      };

    } // namespace bc
  } // namespace graph
} // namespace b40c
