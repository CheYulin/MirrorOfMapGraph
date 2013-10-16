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

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/GASengine/problem_type.cuh>
#include <b40c/graph/GASengine/csr_problem.cuh>
#include <b40c/graph/GASengine/enactor_base.cuh>

#include <b40c/graph/GASengine/vertex_centric/filter_atomic/kernel.cuh>
#include <b40c/graph/GASengine/vertex_centric/filter_atomic/kernel_policy.cuh>
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
#include <omp.h>

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
      template<bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
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

        unsigned long long total_runtimes;			// Total time "worked" by each cta
        unsigned long long total_lifetimes;		// Total time elapsed by each cta
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
        cudaError_t Setup(CsrProblem &csr_problem, int expand_grid_size, int contract_grid_size, int filter_grid_size, int iter)
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
              if (retval = util::B40CPerror(cudaHostAlloc((void **) &done, sizeof(int) * 1, flags), "EnactorContractExpand cudaHostAlloc done failed", __FILE__, __LINE__)) break;

              // Map done into GPU space
              if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &d_done, (void *) done, 0), "EnactorContractExpand cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

              // Create throttle event
              if (retval = util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming), "EnactorContractExpand cudaEventCreateWithFlags throttle_event failed", __FILE__,
                  __LINE__)) break;
            }

            // Make sure host-mapped "iteration" is initialized
            if (!iteration)
            {

              int flags = cudaHostAllocMapped;

              // Allocate pinned memory
              if (retval = util::B40CPerror(cudaHostAlloc((void **) &iteration, sizeof(long long) * 1, flags), "EnactorContractExpand cudaHostAlloc iteration failed", __FILE__, __LINE__)) break;

              // Map into GPU space
              if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **) &d_iteration, (void *) iteration, 0), "EnactorContractExpand cudaHostGetDevicePointer iteration failed", __FILE__,
                  __LINE__)) break;
            }

            // Make sure software global barriers are initialized
            if (retval = global_barrier.Setup(expand_grid_size)) break;

            // Make sure our runtime stats are initialized
            if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
            if (retval = contract_kernel_stats.Setup(contract_grid_size)) break;
            if (retval = filter_kernel_stats.Setup(filter_grid_size)) break;
            if (retval = backward_sum_kernel_stats.Setup(expand_grid_size)) break;
            if (retval = backward_contract_kernel_stats.Setup(contract_grid_size)) break;

            // Reset statistics
            iteration[0] = iter;
            total_runtimes = 0;
            total_lifetimes = 0;
            total_queued = 0;
            done[0] = -1;

            // Single-gpu graph slice
            typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

            // Bind bitmask texture
            int bytes = (graph_slice->nodes + 8 - 1) / 8;
            cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
            if (retval = util::B40CPerror(cudaBindTexture(0, vertex_centric::contract_atomic::BitmaskTex<VisitedMask>::ref, graph_slice->d_visited_mask, bitmask_desc, bytes),
                "EnactorVertexCentric cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

            if (retval = util::B40CPerror(cudaBindTexture(0, vertex_centric::filter_atomic::BitmaskTex<VisitedMask>::ref, graph_slice->d_visited_mask, bitmask_desc, bytes),
                "EnactorVertexCentric cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

            // Bind row-offsets texture
            cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::B40CPerror(
                cudaBindTexture(0, vertex_centric::expand_atomic::RowOffsetTex<SizeT>::ref, graph_slice->d_row_offsets, row_offsets_desc, (graph_slice->nodes + 1) * sizeof(SizeT)),
                "EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            if (retval = util::B40CPerror(
                cudaBindTexture(0, vertex_centric::backward_sum_atomic::RowOffsetTex<SizeT>::ref, graph_slice->d_row_offsets, row_offsets_desc, (graph_slice->nodes + 1) * sizeof(SizeT)),
                "EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            // Bind column-offsets texture
            if (retval = util::B40CPerror(
                cudaBindTexture(0, vertex_centric::gather::RowOffsetTex<SizeT>::ref, graph_slice->d_column_offsets, row_offsets_desc, (graph_slice->nodes + 1) * sizeof(SizeT)),
                "EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

          }
          while (0);

          return retval;
        }

      public:

        /**
         * Constructor
         */
        EnactorVertexCentric(bool DEBUG = false) :
            EnactorBase(EDGE_FRONTIERS, DEBUG), iteration(NULL), d_iteration(NULL), total_queued(0), done(NULL), d_done(NULL)
        {
        }

        /**
         * Destructor
         */
        virtual ~EnactorVertexCentric()
        {
          if (iteration)
          {
            util::B40CPerror(cudaFreeHost((void *) iteration), "EnactorVertexCentric cudaFreeHost iteration failed", __FILE__, __LINE__);
          }
          if (done)
          {
            util::B40CPerror(cudaFreeHost((void *) done), "EnactorVertexCentric cudaFreeHost done failed", __FILE__, __LINE__);

            util::B40CPerror(cudaEventDestroy(throttle_event), "EnactorVertexCentric cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
          }
        }

        /**
         * Obtain statistics about the last BFS search enacted
         */
        template<typename VertexId>
        void GetStatistics(long long &total_queued, VertexId &search_depth, double &avg_duty)
        {
          cudaThreadSynchronize();

          total_queued = this->total_queued;
          search_depth = this->iteration[0] - 1;

          avg_duty = (total_lifetimes > 0) ? double(total_runtimes) / total_lifetimes : 0.0;
        }

        /**
         * Enacts a breadth-first-search on the specified graph problem. Invokes
         * new expansion and contraction grid kernels for each BFS iteration.
         *
         * @return cudaSuccess on success, error enumeration otherwise
         */
        template<typename ExpandPolicy, typename GatherPolicy, typename FilterPolicy, typename ContractPolicy, typename BackwardContractPolicy, typename BackwardSumPolicy, typename CsrProblem>
        cudaError_t EnactIterativeSearch(CsrProblem &csr_problem, typename CsrProblem::VertexId src_node, char* source_file_name, typename CsrProblem::SizeT* h_row_offsets, int max_grid_size = 0, double max_queue_sizing = 3.0)
        {
          typedef typename CsrProblem::SizeT SizeT;
          typedef typename CsrProblem::EValue EValue;
          typedef typename CsrProblem::VertexId VertexId;
          typedef typename CsrProblem::VisitedMask VisitedMask;

          DEBUG = false;
          cudaError_t retval = cudaSuccess;

          // Determine grid size(s)
          int expand_occupancy = ExpandPolicy::CTA_OCCUPANCY;
          int expand_grid_size = MaxGridSize(expand_occupancy, max_grid_size);

          int filter_occupancy = FilterPolicy::CTA_OCCUPANCY;
          int filter_grid_size = MaxGridSize(filter_occupancy, max_grid_size);

          int gather_occupancy = GatherPolicy::CTA_OCCUPANCY;
          int gather_grid_size = MaxGridSize(gather_occupancy, max_grid_size);

          int contract_occupancy = ContractPolicy::CTA_OCCUPANCY;
          int contract_grid_size = MaxGridSize(contract_occupancy, max_grid_size);

          int backward_contract_occupancy = BackwardContractPolicy::CTA_OCCUPANCY;
          int backward_contract_grid_size = MaxGridSize(backward_contract_occupancy, max_grid_size);

          int backward_sum_occupancy = BackwardSumPolicy::CTA_OCCUPANCY;
          int backward_sum_grid_size = MaxGridSize(backward_sum_occupancy, max_grid_size);

          if (DEBUG)
          {
            printf("BFS expand occupancy %d, level-grid size %d\n", expand_occupancy, expand_grid_size);
            printf("BFS contract occupancy %d, level-grid size %d\n", contract_occupancy, contract_grid_size);
            printf("BFS backward contract occupancy %d, level-grid size %d\n", backward_contract_occupancy, backward_contract_grid_size);
            printf("BFS backward sum occupancy %d, level-grid size %d\n", backward_sum_occupancy, backward_sum_grid_size);
          }
          
//          FILE* src_file;
//          if ((src_file = fopen(source_file_name, "r")) == NULL)
//          {
//            printf("Source file open error!\n");
//            exit(0);
//          }
//          int num_src;
//          const int max_src_num = 100;
//          int* srcs = new int[max_src_num];
//          for (num_src = 0; num_src < max_src_num; num_src++)
//          {
//            if ((fscanf(src_file, "%d\n", &srcs[num_src]) != EOF))
//            {
//              srcs[num_src]--; //0-based index
//            }
//            else
//              break;
//          }

          for (int iter = 0; iter < 1; ++iter)
          {
            // Reset data for single pass BC algorithm
            if (retval = csr_problem.Reset(GetFrontierType(), max_queue_sizing)) break;

            // Lazy initialization
            if (retval = Setup(csr_problem, expand_grid_size, contract_grid_size, filter_grid_size, 0)) break;

            // Single-gpu graph slice
            typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

            SizeT queue_length;
            VertexId queue_index = 0;					// Work stealing/queue index
            int selector = 0;
            int NUM_iter = 0;

//            if (retval = util::B40CPerror(cudaMemcpy(graph_slice->frontier_queues.d_keys[selector ^ 1], graph_slice->d_vertex_ids, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToDevice),
//                "CsrProblem cudaMemcpy d_vertex_ids failed", __FILE__, __LINE__)) return retval;

            printf("Starting at vertex %d\n", src_node);
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            double startcontract;
            double startexpand;
            double startapply;
            
            double elapsedcontract = 0.0;      
            double elapsedexpand = 0.0;
            double elapsedapply = 0.0;
            double elapsedinit = 0.0;

            cudaEventRecord(start);
            double startTime = omp_get_wtime();

            // Forward phase BC iterations
            while (done[0] < 0)
//            for (int i = 0; i < 4; i++)
            {
              if (DEBUG) printf("Iteration: %lld\n", (long long) NUM_iter);

//              startinit = omp_get_wtime();
//              // Initialize d_gather_results elements to LARGE
//              SizeT memset_block_size = 256;
//              SizeT memset_grid_size = B40C_MIN(65535, (graph_slice->nodes + memset_block_size - 1) / memset_block_size);
//              util::MemsetKernel<EValue><<<memset_grid_size, memset_block_size, 0, graph_slice->stream>>>(
//                  graph_slice->d_gather_results,
//                  100000000,
//                  graph_slice->nodes);
//              
////              cudaEventRecord(endinit);
////              cudaEventSynchronize(endinit);
////              float tmpelapsed;
////              cudaEventElapsedTime(&tmpelapsed, startinit, endinit);
//              cudaDeviceSynchronize();
//              endinit = omp_get_wtime();
//              elapsedinit += endinit - startinit;
              
//              cudaEventRecord(startcontract);
              if (DEBUG) startcontract = omp_get_wtime()* 1000;
              //
              // Contraction
              //

              vertex_centric::contract_atomic::Kernel<ContractPolicy><<<contract_grid_size, ContractPolicy::THREADS>>>(
                  src_node,
//                  iteration[0],
                  NUM_iter,
                  0,//unused: we obtain this from device-side counters instead
                  queue_index,// queue counter index
                  queue_index,// steal counter index
                  1,// number of GPUs
                  d_done,
                  graph_slice->frontier_queues.d_keys[selector ^ 1],// filtered edge frontier in
                  graph_slice->frontier_queues.d_keys[selector],// vertex frontier out
                  graph_slice->frontier_queues.d_values[selector^1],// predecessor in
//                  graph_slice->d_labels,// source distance out
//                  graph_slice->d_preds,// prtedecessor out
//                  graph_slice->d_sigmas,
                  graph_slice->d_dists,//
                  graph_slice->d_dists_out,//
                  graph_slice->d_gather_results,//
                  graph_slice->d_changed,//
                  graph_slice->d_visited_mask,//
                  this->work_progress,//
                  graph_slice->frontier_elements[selector ^ 1],// max filtered edge frontier vertices
                  graph_slice->frontier_elements[selector],// max vertex frontier vertices
                  this->contract_kernel_stats);

              if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;
              cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates
              
              
              
//              cudaEventRecord(endcontract);
//              cudaEventSynchronize(endcontract);
//              cudaEventElapsedTime(&tmpelapsed, startcontract, endcontract);
              if (DEBUG) 
              {
                cudaDeviceSynchronize();
                elapsedcontract += omp_get_wtime()*1000 - startcontract;
              }
                

              queue_index++;
              selector ^= 1;

//              if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
//              if(queue_length == 0)
//                break;

              if (DEBUG)
              {
                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                printf("queue_length after contraction: %lld\n", (long long) queue_length);

//                VertexId* test_vid = new VertexId[queue_length];
//                cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                printf("Frontier after contraction: ");
//                for (int i = 0; i < queue_length; ++i)
//                {
//                  printf("%d, ", test_vid[i]);
//                }
//                printf("\n");
//                delete[] test_vid;
//
//                EValue* test_vid2 = new EValue[graph_slice->nodes];
//                cudaMemcpy(test_vid2, graph_slice->d_gather_results, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                printf("d_gather_results after contraction: ");
//                for (int i = 0; i < graph_slice->nodes; ++i)
//                {
//                  printf("%d, ", test_vid2[i]);
//                }
//                printf("\n");
//                delete[] test_vid2;

              }
//              if (INSTRUMENT)
//              {
//                if (retval = contract_kernel_stats.Accumulate(contract_grid_size, total_runtimes, total_lifetimes)) break;
//              }

//              cudaDeviceSynchronize();
              // Throttle
//              if (iteration[0] & 1)
              if (NUM_iter & 1)
              {
                if (retval = util::B40CPerror(cudaEventRecord(throttle_event), "EnactorVertexCentric cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
              }
              else
              {
                if (retval = util::B40CPerror(cudaEventSynchronize(throttle_event), "EnactorVertexCentric cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
              };

              // Check if done
              if (done[0] == 0) break;

//              if (iteration[0] != 0)
//              if (NUM_iter != 0)
//              {
//
////                if (retval = util::B40CPerror(cudaMemcpy(graph_slice->d_gather_results, graph_slice->d_dists, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToDevice),
////                    "CsrProblem cudaMemcpy d_gather_results failed", __FILE__, __LINE__)) break;
//
//                //
//                //gather only: gather stage of the GASengine, does not modify frontier
//                //
//                vertex_centric::gather::Kernel<GatherPolicy><<<gather_grid_size, GatherPolicy::THREADS>>>(
//                    queue_index,                // queue counter index
//                    queue_index,// steal counter index
//                    1,// number of GPUs
//                    d_done,
//                    graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
//                    graph_slice->frontier_queues.d_keys[selector],// edge frontier out
////                  graph_slice->frontier_queues.d_values[selector],// predecessor out
//                    graph_slice->d_row_indices,//pass in the CSC graph to gather for destination vertices
//                    graph_slice->d_column_offsets,//pass in the CSC graph to gather for destination vertices
//                    graph_slice->d_num_out_edges,// pass number of out edges
//                    graph_slice->d_dists,//the page rank
//                    graph_slice->d_gather_results,//gather results
//                    graph_slice->d_changed,// changed flag
////                  graph_slice->d_visit_flags,
//                    this->work_progress,
//                    graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
//                    graph_slice->frontier_elements[selector],// max edge frontier vertices
//                    this->expand_kernel_stats);
//
//                if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "gather::Kernel failed ", __FILE__, __LINE__))) break;
//                cudaEventQuery(throttle_event);// give host memory mapped visibility to GPU updates
//
//                //                queue_index++;
//                //                selector ^= 1;
//
//                if (INSTRUMENT && DEBUG)
//                {
//                  if (work_progress.GetQueueLength(queue_index, queue_length)) break;
//                  total_queued += queue_length;
//
////                  if (DEBUG) printf("queue_length after gather: %lld\n", (long long) queue_length);
//
////                  EValue *test_vid2 = new EValue[graph_slice->nodes];
////                  cudaMemcpy(test_vid2, graph_slice->d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
////                  printf("d_dists after gather: ");
////                  for (int i = 0; i < graph_slice->nodes; ++i)
////                  {
////                    printf("%d, ", test_vid2[i]);
////                  }
////                  printf("\n");
////                  delete[] test_vid2;
//
////                  EValue* test_vid2 = new EValue[graph_slice->nodes];
////                  cudaMemcpy(test_vid2, graph_slice->d_gather_results, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
////                  printf("d_gather_results after gather: ");
////                  for (int i = 0; i < graph_slice->nodes; ++i)
////                  {
////                    printf("%d, ", test_vid2[i]);
////                  }
////                  printf("\n");
////                  delete[] test_vid2;
//
////                  VertexId* test_vid = new VertexId[graph_slice->nodes];
////                  cudaMemcpy(test_vid, graph_slice->d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
////                  printf("changed after gather: ");
////                  for (int i = 0; i < graph_slice->nodes; ++i)
////                  {
////                    printf("%d, ", test_vid[i]);
////                  }
////                  printf("\n");
////                  delete[] test_vid;
//
//                  if (INSTRUMENT)
//                  {
//                    if (retval = expand_kernel_stats.Accumulate(gather_grid_size, total_runtimes, total_lifetimes)) break;
//                  }
//                }
//
////              if (DEBUG) printf("\n%lld", (long long) iteration[0]);
//
//                // Check if done
//                if (done[0] == 0) break;
//
//              cudaEventRecord(startapply);
              if (DEBUG) startapply = omp_get_wtime()* 1000;
                //
                //apply stage
                //
                vertex_centric::gather::apply<GatherPolicy><<<gather_grid_size, GatherPolicy::THREADS>>>(iteration[0], queue_index, this->work_progress, graph_slice->frontier_queues.d_keys[selector ^ 1],
                    graph_slice->d_changed, graph_slice->d_dists, graph_slice->d_dists_out, graph_slice->d_gather_results);
                
//                cudaEventRecord(endapply);
//              cudaEventSynchronize(endapply);
//              cudaEventElapsedTime(&tmpelapsed, startapply, endapply);
                if (DEBUG) 
                {
                  cudaDeviceSynchronize();
                  elapsedapply += omp_get_wtime()*1000 - startapply;
                }
              
//              //
//                //reset dists and gather_results
//                //
                vertex_centric::gather::reset_gather_result<GatherPolicy><<<gather_grid_size, GatherPolicy::THREADS>>>(iteration[0], queue_index, this->work_progress, graph_slice->frontier_queues.d_keys[selector ^ 1],
                    graph_slice->d_changed, graph_slice->d_dists, graph_slice->d_dists_out, graph_slice->d_gather_results);
              

                if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "gather::reset_changed Kernel failed ", __FILE__, __LINE__))) break;
//                cudaMemcpy(graph_slice->d_dists, graph_slice->d_dists_out, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToDevice);// copy dist_out to dists

                
                if (INSTRUMENT && DEBUG)
                {
//                  EValue *test_vid2 = new EValue[graph_slice->nodes];
//                  cudaMemcpy(test_vid2, graph_slice->d_dists, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToHost);
//                  printf("d_dists after apply: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%d, ", test_vid2[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid2;
//
//                  VertexId *test_vid = new VertexId[graph_slice->nodes];
//                  cudaMemcpy(test_vid, graph_slice->d_changed, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                  printf("changed after apply: ");
//                  for (int i = 0; i < graph_slice->nodes; ++i)
//                  {
//                    printf("%d, ", test_vid[i]);
//                  }
//                  printf("\n");
//                  delete[] test_vid;
                }
//              }

//                cudaEventRecord(startexpand);
                if(DEBUG)
                {
                  cudaDeviceSynchronize();
                  startexpand = omp_get_wtime() * 1000;
                }
               
//               printf("Starting expansion\n");
              //
              // Expansion
              //
              vertex_centric::expand_atomic::Kernel<ExpandPolicy><<<expand_grid_size, ExpandPolicy::THREADS>>>(
                  NUM_iter,                //
                  graph_slice->nodes,
                  graph_slice->edges,
                  queue_index,// queue counter index
                  queue_index,// steal counter index
                  1,// number of GPUs
                  d_done,
                  graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                  graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                  graph_slice->frontier_queues.d_values[selector],// predecessor out
                  graph_slice->d_column_indices,
                  graph_slice->d_row_offsets,
                  graph_slice->d_dists,
                  graph_slice->d_dists_out,
                  graph_slice->d_gather_results,//gather results
                  graph_slice->d_changed,
                  graph_slice->d_visited_flag,
                  this->work_progress,
                  graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                  graph_slice->frontier_elements[selector],// max edge frontier vertices
                  this->expand_kernel_stats);

              if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "expand_atomic::Kernel failed ", __FILE__, __LINE__))) break;
              cudaEventQuery(throttle_event);                // give host memory mapped visibility to GPU updates
              
//              cudaEventRecord(endexpand);
//              cudaEventSynchronize(endexpand);
//              cudaEventElapsedTime(&tmpelapsed, startexpand, endexpand);
              if(DEBUG)
              {
                cudaDeviceSynchronize();
                elapsedexpand += omp_get_wtime()*1000 - startexpand;
              }

//              cudaMemcpy(graph_slice->d_dists, graph_slice->d_dists_out, graph_slice->nodes * sizeof(EValue), cudaMemcpyDeviceToDevice);// copy dist_out to dists

              queue_index++;
              selector ^= 1;
//              iteration[0]++;
              NUM_iter++;
              //if (work_progress.GetQueueLength(queue_index, queue_length)) break;
              //if(queue_length > max_queue_sizing * graph_slice->edges)
              //{
              //  printf("Error: queuse size out of bound!\n");
              //  break;
              //}

              if (INSTRUMENT && DEBUG)
              {
                if (work_progress.GetQueueLength(queue_index, queue_length)) break;

                total_queued += queue_length;
                if (DEBUG) printf("queue_length after expansion: %lld\n", (long long) queue_length);

//                VertexId* test_vid = new VertexId[queue_length];
//                cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                printf("Frontier after expansion: ");
//                for (int i = 0; i < queue_length; ++i)
//                {
//                  printf("%d, ", test_vid[i]);
//                }
//                printf("\n");
//                delete[] test_vid;
//
//                test_vid = new VertexId[graph_slice->nodes];
//                cudaMemcpy(test_vid, graph_slice->d_dists, graph_slice->nodes * sizeof(VertexId), cudaMemcpyDeviceToHost);
//                printf("d_dists after expansion: ");
//                for (int i = 0; i < graph_slice->nodes; ++i)
//                {
//                  printf("%d, ", test_vid[i]);
//                }
//                printf("\n");
//                delete[] test_vid;
              }

//              if (DEBUG) printf("\n%lld", (long long) iteration[0]);

              // Check if done
              if (done[0] == 0) break;

              //reset d_visited_flag
//              cudaMemset(graph_slice->d_visited_flag, 0, graph_slice->nodes * sizeof(VertexId));
//              if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "cudaMemset d_visited_flag for failed ", __FILE__, __LINE__))) break;

//              // Check if any of the frontiers overflowed due to redundant expansion
//              bool overflowed = false;
//              if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
//              if (overflowed)
//              {
//                retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
//                break;
//              }

//              //
//              // Filter
//              //
//
//              vertex_centric::filter_atomic::Kernel<FilterPolicy><<<filter_grid_size, FilterPolicy::THREADS>>>(
//                  queue_index,                                            // queue counter index
//                  queue_index,// steal counter index
//                  d_done,
//                  graph_slice->frontier_queues.d_keys[selector ^ 1],// edge frontier in
//                  graph_slice->frontier_queues.d_keys[selector],// vertex frontier out
//                  graph_slice->frontier_queues.d_values[selector ^ 1],// predecessor in
//                  graph_slice->frontier_queues.d_values[selector],// predecessor out
//                  graph_slice->d_visited_mask,
//                  this->work_progress,
//                  graph_slice->frontier_elements[selector ^ 1],// max edge frontier vertices
//                  graph_slice->frontier_elements[selector],// max vertex frontier vertices
//                  this->filter_kernel_stats);
//
//              if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "filter_atomic::Kernel failed ", __FILE__, __LINE__))) break;
//
//              queue_index++;
//              selector ^= 1;
//
//              if (DEBUG)
//              {
//                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
//                printf("queue_length after filter: %lld\n", (long long) queue_length);
//              }
//              if (INSTRUMENT)
//              {
//                if (retval = filter_kernel_stats.Accumulate(filter_grid_size, total_runtimes, total_lifetimes)) break;
//              }

            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            double endTime = omp_get_wtime();
            double elapsed_wall = (omp_get_wtime() - startTime) * 1000;

            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            std::cout << "Kernel time took: " << elapsed << " ms" << std::endl;
            std::cout << "Wall time took: " << elapsed_wall << " ms" << std::endl;
            std::cout << "Contract time took: " << elapsedcontract << " ms" << std::endl;
            std::cout << "Expand time took: " << elapsedexpand << " ms" << std::endl;
            std::cout << "Apply time took: " << elapsedapply << " ms" << std::endl;
            std::cout << "Init time took: " << elapsedinit*1000 << " ms" << std::endl;
            
            // Compute nodes and edges visited
            SizeT edges_visited = 0;
            SizeT nodes_visited = 0;
            VertexId* h_dists = new VertexId[graph_slice->nodes];
            cudaMemcpy(h_dists, graph_slice->d_dists, graph_slice->nodes * sizeof (VertexId), cudaMemcpyDeviceToHost);
            for (VertexId i = 0; i < graph_slice->nodes; i++)
            {
              if (h_dists[i] > -1)
              {
                nodes_visited++;
                edges_visited += h_row_offsets[i + 1] - h_row_offsets[i];
              }
            }
            delete [] h_dists;
            std::cout << "M-Edges / sec: " << (double) edges_visited / (elapsed_wall * 1000.f) << std::endl;
            
            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            retval = work_progress.CheckOverflow<SizeT>(overflowed);         
            if (overflowed)
            {
              retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
              retval = (cudaError_t) 1;
            }
            std::cout << "Total iteration: " << NUM_iter << std::endl;
            std::cout << "retval: " << retval << std::endl;

//            if (retval) break;
//
//            if (DEBUG) printf("\n");

//            // Call Setup again to initialize kernel stats vars
//            int max_search_depth = --iteration[0];
//            if (retval = Setup(csr_problem, expand_grid_size, contract_grid_size, max_search_depth)) break;
//
//            // Check if any of the frontiers overflowed due to redundant expansion
//            bool overflowed = false;
//            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
//            if (overflowed)
//            {
//              retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
//              break;
//            }
//
//            // Backward phase here
//            // backward_contract
//            // backward_sum
//            //
//            queue_index = 0;					// Reset work stealing/queue index
//            selector = 0;                    // Reset selector for ping-pong
//
//            // Ignore the most outside layer
//            iteration[0]--;
//
//            while (iteration[0] > 0)
//            {
//              //
//              // Backward contraction
//              //
//
//              vertex_centric::backward_contract_atomic::Kernel<BackwardContractPolicy><<<backward_contract_grid_size, BackwardContractPolicy::THREADS>>>(
//                  max_search_depth,
//                  iteration[0],
//                  graph_slice->nodes,										// num_elements
//                  queue_index,// queue counter index
//                  queue_index,// steal counter index
//                  1,// number of GPUs
//                  d_done,
//                  graph_slice->d_vertex_ids,// filtered edge frontier in
//                  graph_slice->frontier_queues.d_keys[selector],// vertex frontier out
//                  graph_slice->d_preds,// predecessor in
//                  graph_slice->d_labels,
//                  graph_slice->d_visited_mask,
//                  this->work_progress,
//                  graph_slice->frontier_elements[selector ^ 1],// max filtered edge frontier vertices
//                  graph_slice->frontier_elements[selector],// max vertex frontier vertices
//                  this->backward_contract_kernel_stats);
//
//              if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "backward_contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;
//              cudaEventQuery(throttle_event);// give host memory mapped visibility to GPU updates
//
//              queue_index++;
//              selector ^= 1;
//
//              if (DEBUG)
//              {
//                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
//              }
//              if (INSTRUMENT)
//              {
//                if (retval = backward_contract_kernel_stats.Accumulate(
//                        contract_grid_size,
//                        total_runtimes,
//                        total_lifetimes)) break;
//              }
//
//              // Throttle
//              if ((max_search_depth - iteration[0]) & 1)
//              {
//                if (retval = util::B40CPerror(cudaEventRecord(throttle_event),
//                        "EnactorVertexCentric cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
//              }
//              else
//              {
//                if (retval = util::B40CPerror(cudaEventSynchronize(throttle_event),
//                        "EnactorVertexCentric cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
//              };
//
//              //
//              // Backward sum
//
//              vertex_centric::backward_sum_atomic::Kernel<BackwardSumPolicy>
//              <<<expand_grid_size, BackwardSumPolicy::THREADS>>>(
//                  iteration[0],
//                  queue_index,// queue counter index
//                  queue_index,// steal counter index
//                  1,// number of GPUs
//                  d_done,
//                  graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
//                  graph_slice->d_column_indices,
//                  graph_slice->d_row_offsets,
//                  graph_slice->d_node_values,
//                  graph_slice->d_labels,
//                  graph_slice->d_sigmas,
//                  graph_slice->d_deltas,
//                  graph_slice->d_visit_flags,
//                  this->work_progress,
//                  graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
//                  graph_slice->frontier_elements[selector],// max edge frontier vertices
//                  this->backward_sum_kernel_stats);
//
//              if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "backward_sum_atomic::Kernel failed ", __FILE__, __LINE__))) break;
//
//              cudaEventQuery(throttle_event);// give host memory mapped visibility to GPU updates
//
//              queue_index++;
//              selector ^= 1;
//              iteration[0]--;
//
//              if (INSTRUMENT || DEBUG)
//              {
//                if (work_progress.GetQueueLength(queue_index, queue_length)) break;
//                total_queued += queue_length;
//                if (INSTRUMENT)
//                {
//                  if (retval = backward_sum_kernel_stats.Accumulate(
//                          expand_grid_size,
//                          total_runtimes,
//                          total_lifetimes)) break;
//                }
//              }
//            }

//            printf("Total iteration: %lld\n", (long long) iteration[0]);

          }

//          delete [] srcs;
          return retval;
        }

        /**
         * Enacts a breadth-first-search on the specified graph problem. Invokes
         * new expansion and contraction grid kernels for each BFS iteration.
         *
         * @return cudaSuccess on success, error enumeration otherwise
         */
        template<typename CsrProblem>
        cudaError_t EnactIterativeSearch(CsrProblem &csr_problem, typename CsrProblem::VertexId src_node, char* source_file_name, typename CsrProblem::SizeT* h_row_offsets, int max_grid_size = 0, double max_queue_sizing = 5.0)
        {
          typedef typename CsrProblem::VertexId VertexId;
          typedef typename CsrProblem::SizeT SizeT;

          // GF100
          if (this->cuda_props.device_sm_version >= 200)
          {

            // Expansion kernel config
            typedef vertex_centric::expand_atomic::KernelPolicy<typename CsrProblem::ProblemType, 200,					// CUDA_ARCH
                INSTRUMENT, 			// INSTRUMENT
                8,						// CTA_OCCUPANCY
                7,						// LOG_THREADS
                0,						// LOG_LOAD_VEC_SIZE
                0,						// LOG_LOADS_PER_TILE
                5,						// LOG_RAKING_THREADS
                util::io::ld::cg,		// QUEUE_READ_MODIFIER,
                util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
                util::io::ld::NONE,     // EDGE_VALUES_READ_MODIFIER,
                util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
                util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
                util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
                false,					// WORK_STEALING
                32,						// WARP_GATHER_THRESHOLD
                128 * 4, 				// CTA_GATHER_THRESHOLD,
                7>						// LOG_SCHEDULE_GRANULARITY
            ExpandPolicy;

            // Gather kernel config
            typedef vertex_centric::gather::KernelPolicy<typename CsrProblem::ProblemType, 200,                  // CUDA_ARCH
                INSTRUMENT,             // INSTRUMENT
                8,                      // CTA_OCCUPANCY
                7,                      // LOG_THREADS
                0,                      // LOG_LOAD_VEC_SIZE
                0,                      // LOG_LOADS_PER_TILE
                5,                      // LOG_RAKING_THREADS
                util::io::ld::cg,       // QUEUE_READ_MODIFIER,
                util::io::ld::NONE,     // COLUMN_READ_MODIFIER,
                util::io::ld::NONE,     // EDGE_VALUES_READ_MODIFIER,
                util::io::ld::cg,       // ROW_OFFSET_ALIGNED_READ_MODIFIER,
                util::io::ld::NONE,     // ROW_OFFSET_UNALIGNED_READ_MODIFIER,
                util::io::st::cg,       // QUEUE_WRITE_MODIFIER,
                false,                   // NO WORK_STEALING : changed from true
                32,                     // WARP_GATHER_THRESHOLD
                128 * 4,                // CTA_GATHER_THRESHOLD,
                7>                      // LOG_SCHEDULE_GRANULARITY
            GatherPolicy;

            // Filter kernel config
            typedef vertex_centric::filter_atomic::KernelPolicy<typename CsrProblem::ProblemType, 200,                    // CUDA_ARCH
                INSTRUMENT,             // INSTRUMENT
                0,                      // SATURATION_QUIT
                8,                      // CTA_OCCUPANCY
                7,                      // LOG_THREADS
                1,                      // LOG_LOAD_VEC_SIZE
                1,                      // LOG_LOADS_PER_TILE
                5,                      // LOG_RAKING_THREADS
                util::io::ld::NONE,     // QUEUE_READ_MODIFIER,
                util::io::st::NONE,     // QUEUE_WRITE_MODIFIER,
                false,                  // WORK_STEALING
                9>                      // LOG_SCHEDULE_GRANULARITY
            FilterPolicy;

            // Contraction kernel config
            typedef vertex_centric::contract_atomic::KernelPolicy<typename CsrProblem::ProblemType, 200,					// CUDA_ARCH
                INSTRUMENT, 			// INSTRUMENT
                0, 						// SATURATION_QUIT
                true, 					// DEQUEUE_PROBLEM_SIZE
                8,						// CTA_OCCUPANCY
                7,						// LOG_THREADS
                1,						// LOG_LOAD_VEC_SIZE
                0,						// LOG_LOADS_PER_TILE
                5,						// LOG_RAKING_THREADS
                util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
                util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
                false,					// WORK_STEALING
                0,						// END_BITMASK_CULL (never cull b/c filter does the bitmask culling)
                8> 						// LOG_SCHEDULE_GRANULARITY
            ContractPolicy;

            // Contraction kernel config
            typedef vertex_centric::backward_contract_atomic::KernelPolicy<typename CsrProblem::ProblemType, 200,					// CUDA_ARCH
                INSTRUMENT, 			// INSTRUMENT
                0, 						// SATURATION_QUIT
                true, 					// DEQUEUE_PROBLEM_SIZE
                8,						// CTA_OCCUPANCY
                7,						// LOG_THREADS
                1,						// LOG_LOAD_VEC_SIZE
                0,						// LOG_LOADS_PER_TILE
                5,						// LOG_RAKING_THREADS
                util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
                util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
                false,					// WORK_STEALING
                0,						// END_BITMASK_CULL (never cull b/c filter does the bitmask culling)
                8> 						// LOG_SCHEDULE_GRANULARITY
            BackwardContractPolicy;

            // Backward sum kernel config
            typedef vertex_centric::backward_sum_atomic::KernelPolicy<typename CsrProblem::ProblemType, 200,					// CUDA_ARCH
                INSTRUMENT, 			// INSTRUMENT
                8,						// CTA_OCCUPANCY
                7,						// LOG_THREADS
                0,						// LOG_LOAD_VEC_SIZE
                0,						// LOG_LOADS_PER_TILE
                5,						// LOG_RAKING_THREADS
                util::io::ld::cg,		// QUEUE_READ_MODIFIER,
                util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
                util::io::ld::NONE,     // EDGE_VALUES_READ_MODIFIER,
                util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
                util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
                util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
                true,					// WORK_STEALING
                32,						// WARP_GATHER_THRESHOLD
                128 * 4, 				// CTA_GATHER_THRESHOLD,
                7>						// LOG_SCHEDULE_GRANULARITY
            BackwardSumPolicy;

            return EnactIterativeSearch<ExpandPolicy, GatherPolicy, FilterPolicy, ContractPolicy, BackwardContractPolicy, BackwardSumPolicy>(csr_problem, src_node, source_file_name, h_row_offsets, max_grid_size, max_queue_sizing);
          }

          printf("Not yet tuned for this architecture\n");
          return cudaErrorInvalidDeviceFunction;
        }
      };

    } // namespace bc
  } // namespace graph
} // namespace b40c
