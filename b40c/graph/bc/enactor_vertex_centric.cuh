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

#include <b40c/graph/bc/problem_type.cuh>
#include <b40c/graph/bc/csr_problem.cuh>
#include <b40c/graph/bc/enactor_base.cuh>

#include <b40c/graph/bc/vertex_centric/expand_atomic/kernel.cuh>
#include <b40c/graph/bc/vertex_centric/expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bc/vertex_centric/contract_atomic/kernel.cuh>
#include <b40c/graph/bc/vertex_centric/contract_atomic/kernel_policy.cuh>
#include <b40c/graph/bc/vertex_centric/backward_contract_atomic/kernel.cuh>
#include <b40c/graph/bc/vertex_centric/backward_contract_atomic/kernel_policy.cuh>
#include <b40c/graph/bc/vertex_centric/backward_sum_atomic/kernel.cuh>
#include <b40c/graph/bc/vertex_centric/backward_sum_atomic/kernel_policy.cuh>


namespace b40c {
namespace graph {
namespace bc {



/**
 * Vertex-centric BC enactor
 *  
 * For each BFS iteration, visited/duplicate vertices are culled from
 * the incoming edge-frontier in global memory.  The remaining vertices are
 * compacted to a vertex-frontier in global memory.  Then these
 * vertices are read back in and expanded to construct the outgoing
 * edge-frontier in global memory.
 */
template <bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
class EnactorVertexCentric : public EnactorBase
{

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

protected:

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime expand_kernel_stats;
	util::KernelRuntimeStatsLifetime contract_kernel_stats;
    util::KernelRuntimeStatsLifetime backward_sum_kernel_stats;
	util::KernelRuntimeStatsLifetime backward_contract_kernel_stats;

	unsigned long long 		total_runtimes;			// Total time "worked" by each cta
	unsigned long long 		total_lifetimes;		// Total time elapsed by each cta
	unsigned long long 		total_queued;

	/**
	 * Throttle state.  We want the host to have an additional BFS iteration
	 * of kernel launches queued up for for pipeline efficiency (particularly on
	 * Windows), so we keep a pinned, mapped word that the traversal kernels will
	 * signal when done.
	 */
	volatile int 	*done;
	int 			*d_done;
	cudaEvent_t		throttle_event;

	/**
	 * Mechanism for implementing software global barriers from within
	 * a single grid invocation
	 */
	util::GlobalBarrierLifetime 		global_barrier;

	/**
	 * Current iteration (mapped into GPU space so that it can
	 * be modified by multi-iteration kernel launches)
	 */
	volatile long long 					*iteration;
	long long 							*d_iteration;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

protected:

	/**
	 * Prepare enactor for search.  Must be called prior to each search.
	 */
	template <typename CsrProblem>
	cudaError_t Setup(
		CsrProblem &csr_problem,
		int expand_grid_size,
		int contract_grid_size,
		int iter)
	{
		typedef typename CsrProblem::SizeT 			    SizeT;
		typedef typename CsrProblem::VertexId 			VertexId;
		typedef typename CsrProblem::VisitedMask 		VisitedMask;

		cudaError_t retval = cudaSuccess;

		do {

			// Make sure host-mapped "done" is initialized
			if (!done) {
				int flags = cudaHostAllocMapped;

				// Allocate pinned memory for done
				if (retval = util::B40CPerror(cudaHostAlloc((void **)&done, sizeof(int) * 1, flags),
					"EnactorContractExpand cudaHostAlloc done failed", __FILE__, __LINE__)) break;

				// Map done into GPU space
				if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_done, (void *) done, 0),
					"EnactorContractExpand cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

				// Create throttle event
				if (retval = util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
					"EnactorContractExpand cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
			}

			// Make sure host-mapped "iteration" is initialized
			if (!iteration) {

				int flags = cudaHostAllocMapped;

				// Allocate pinned memory
				if (retval = util::B40CPerror(cudaHostAlloc((void **)&iteration, sizeof(long long) * 1, flags),
					"EnactorContractExpand cudaHostAlloc iteration failed", __FILE__, __LINE__)) break;

				// Map into GPU space
				if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_iteration, (void *) iteration, 0),
					"EnactorContractExpand cudaHostGetDevicePointer iteration failed", __FILE__, __LINE__)) break;
			}

			// Make sure software global barriers are initialized
			if (retval = global_barrier.Setup(expand_grid_size)) break;

			// Make sure our runtime stats are initialized
			if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
			if (retval = contract_kernel_stats.Setup(contract_grid_size)) break;
            if (retval = backward_sum_kernel_stats.Setup(expand_grid_size)) break;
			if (retval = backward_contract_kernel_stats.Setup(contract_grid_size)) break;

			// Reset statistics
			iteration[0] 		= iter;
			total_runtimes 		= 0;
			total_lifetimes 	= 0;
			total_queued 		= 0;
			done[0] 			= -1;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Bind bitmask texture
			int bytes = (graph_slice->nodes + 8 - 1) / 8;
			cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					vertex_centric::contract_atomic::BitmaskTex<VisitedMask>::ref,
					graph_slice->d_visited_mask,
					bitmask_desc,
					bytes),
				"EnactorVertexCentric cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind row-offsets texture
			cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					vertex_centric::expand_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;
            
            if (retval = util::B40CPerror(cudaBindTexture(
					0,
					vertex_centric::backward_sum_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorVertexCentric cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;


		} while (0);

		return retval;
	}


public: 	
	
	/**
	 * Constructor
	 */
	EnactorVertexCentric(bool DEBUG = false) :
		EnactorBase(EDGE_FRONTIERS, DEBUG),
		iteration(NULL),
		d_iteration(NULL),
		total_queued(0),
		done(NULL),
		d_done(NULL)
	{}


	/**
	 * Destructor
	 */
	virtual ~EnactorVertexCentric()
	{
		if (iteration) {
			util::B40CPerror(cudaFreeHost((void *) iteration), "EnactorVertexCentric cudaFreeHost iteration failed", __FILE__, __LINE__);
		}
		if (done) {
			util::B40CPerror(cudaFreeHost((void *) done),
					"EnactorVertexCentric cudaFreeHost done failed", __FILE__, __LINE__);

			util::B40CPerror(cudaEventDestroy(throttle_event),
				"EnactorVertexCentric cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
		}
	}


    /**
     * Obtain statistics about the last BFS search enacted 
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_queued,
    	VertexId &search_depth,
    	double &avg_duty)
    {
		cudaThreadSynchronize();

		total_queued = this->total_queued;
    	search_depth = this->iteration[0] - 1;

    	avg_duty = (total_lifetimes > 0) ?
    		double(total_runtimes) / total_lifetimes :
    		0.0;
    }

    /**
	 * Enacts a breadth-first-search on the specified graph problem. Invokes
	 * new expansion and contraction grid kernels for each BFS iteration.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
		typename ExpandPolicy,
		typename ContractPolicy,
		typename BackwardContractPolicy,
		typename BackwardSumPolicy,
		typename CsrProblem>
	cudaError_t EnactIterativeSearch(
		CsrProblem 						&csr_problem,
        typename CsrProblem::VertexId 	src_node,
		int 							max_grid_size = 0,
        int                             max_queue_sizing = 1)
	{
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;

		do {
			// Determine grid size(s)
			int expand_occupancy 			= ExpandPolicy::CTA_OCCUPANCY;
			int expand_grid_size 			= MaxGridSize(expand_occupancy, max_grid_size);

			int contract_occupancy			= ContractPolicy::CTA_OCCUPANCY;
			int contract_grid_size 			= MaxGridSize(contract_occupancy, max_grid_size);

			int backward_contract_occupancy	= BackwardContractPolicy::CTA_OCCUPANCY;
			int backward_contract_grid_size = MaxGridSize(backward_contract_occupancy, max_grid_size);

			int backward_sum_occupancy		= BackwardSumPolicy::CTA_OCCUPANCY;
			int backward_sum_grid_size 		= MaxGridSize(backward_sum_occupancy, max_grid_size);

			if (DEBUG) {
				printf("BFS expand occupancy %d, level-grid size %d\n",
					expand_occupancy, expand_grid_size);
				printf("BFS contract occupancy %d, level-grid size %d\n",
					contract_occupancy, contract_grid_size);
                printf("BFS backward contract occupancy %d, level-grid size %d\n",
					backward_contract_occupancy, backward_contract_grid_size);
				printf("BFS backward sum occupancy %d, level-grid size %d\n",
					backward_sum_occupancy, backward_sum_grid_size);

				printf("Iteration, Contraction queue, Expansion queue\n");
				printf("0, 0");
			}

            int start = 0;
            int end = csr_problem.nodes;
            if (src_node != -1)
            {
            	start = src_node;
            	end = src_node+1;
            }
           
            for (int src = start; src < end; ++src)
            {
                // Reset data for single pass BC algorithm
                if (retval = csr_problem.Reset(GetFrontierType(), max_queue_sizing)) break;

                // Lazy initialization
                if (retval = Setup(csr_problem, expand_grid_size, contract_grid_size, 0)) break;

                // Single-gpu graph slice
                typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];


                SizeT queue_length;
                VertexId queue_index 		= 0;					// Work stealing/queue index
                int selector 				= 0;

                // Forward phase BC iterations
                while (done[0] < 0) {
                    //
                    // Contraction
                    //
                    vertex_centric::contract_atomic::Kernel<ContractPolicy>
                        <<<contract_grid_size, ContractPolicy::THREADS>>>(
                                src,
                                iteration[0],
                                0,														// num_elements (unused: we obtain this from device-side counters instead)
                                queue_index,											// queue counter index
                                queue_index,											// steal counter index
                                1,														// number of GPUs
                                d_done,
                                graph_slice->frontier_queues.d_keys[selector ^ 1],		// filtered edge frontier in
                                graph_slice->frontier_queues.d_keys[selector],			// vertex frontier out
                                graph_slice->frontier_queues.d_values[selector^1],	    // predecessor in
                                graph_slice->d_labels,                                  // source distance out
                                graph_slice->d_preds,                                   // prtedecessor out
                                graph_slice->d_sigmas,
                                graph_slice->d_visited_mask,
                                this->work_progress,
                                graph_slice->frontier_elements[selector ^ 1],			// max filtered edge frontier vertices
                                graph_slice->frontier_elements[selector],				// max vertex frontier vertices
                                this->contract_kernel_stats);

                    if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;
                    cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                    /*VertexId *test_vid = new VertexId[graph_slice->nodes];
                    cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector^1], graph_slice->nodes*sizeof(VertexId), cudaMemcpyDeviceToHost);
                    printf("\n");
                    for (int i = 0; i < graph_slice->nodes; ++i)
                    {
                        printf("test data:%d\n", test_vid[i]);
                    }
                    delete[] test_vid;*/


                    queue_index++;
                    selector ^= 1;

                    if (DEBUG) {
                        if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                        printf(", %lld", (long long) queue_length);
                    }
                    if (INSTRUMENT) {
                        if (retval = contract_kernel_stats.Accumulate(
                                    contract_grid_size,
                                    total_runtimes,
                                    total_lifetimes)) break;
                    }

                    // Throttle
                    if (iteration[0] & 1) {
                        if (retval = util::B40CPerror(cudaEventRecord(throttle_event),
                                    "EnactorVertexCentric cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                    } else {
                        if (retval = util::B40CPerror(cudaEventSynchronize(throttle_event),
                                    "EnactorVertexCentric cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                    };

                    // Check if done
                    if (done[0] == 0) break;

                    //
                    // Expansion
                    //

                    vertex_centric::expand_atomic::Kernel<ExpandPolicy>
                        <<<expand_grid_size, ExpandPolicy::THREADS>>>(
                                queue_index,											// queue counter index
                                queue_index,											// steal counter index
                                1,														// number of GPUs
                                d_done,
                                graph_slice->frontier_queues.d_keys[selector ^ 1],		// vertex frontier in
                                graph_slice->frontier_queues.d_keys[selector],			// edge frontier out
                                graph_slice->frontier_queues.d_values[selector],		// predecessor out
                                graph_slice->d_column_indices,
                                graph_slice->d_row_offsets,
                                this->work_progress,
                                graph_slice->frontier_elements[selector ^ 1],			// max vertex frontier vertices
                                graph_slice->frontier_elements[selector],				// max edge frontier vertices
                                this->expand_kernel_stats);

                    if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "expand_atomic::Kernel failed ", __FILE__, __LINE__))) break;
                    cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                    queue_index++;
                    selector ^= 1;
                    iteration[0]++;

                    if (INSTRUMENT || DEBUG) {
                        if (work_progress.GetQueueLength(queue_index, queue_length)) break;
                        total_queued += queue_length;
                        if (DEBUG) printf(", %lld", (long long) queue_length);
                        if (INSTRUMENT) {
                            if (retval = expand_kernel_stats.Accumulate(
                                        expand_grid_size,
                                        total_runtimes,
                                        total_lifetimes)) break;
                        }
                    }

                    if (DEBUG) printf("\n%lld", (long long) iteration[0]);

                    // Check if done
                    if (done[0] == 0) break;

                }
                if (retval) break;

                if (DEBUG) printf("\n");

                // Call Setup again to initialize kernel stats vars
                int max_search_depth = --iteration[0];
                if (retval = Setup(csr_problem, expand_grid_size, contract_grid_size, max_search_depth)) break;

                // Check if any of the frontiers overflowed due to redundant expansion
                bool overflowed = false;
                if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
                if (overflowed) {
                    retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
                    break;
                }

                // Backward phase here
                // backward_contract
                // backward_sum
                //
                queue_index 		= 0;					// Reset work stealing/queue index
                selector 			= 0;                    // Reset selector for ping-pong

                // Ignore the most outside layer
                iteration[0]--;

                while (iteration[0] > 0) {
                    //
                    // Backward contraction
                    //

                    vertex_centric::backward_contract_atomic::Kernel<BackwardContractPolicy>
                        <<<backward_contract_grid_size, BackwardContractPolicy::THREADS>>>(
                                max_search_depth,
                                iteration[0],
                                graph_slice->nodes,										// num_elements
                                queue_index,											// queue counter index
                                queue_index,											// steal counter index
                                1,														// number of GPUs
                                d_done,
                                graph_slice->d_vertex_ids,		                        // filtered edge frontier in
                                graph_slice->frontier_queues.d_keys[selector],			// vertex frontier out
                                graph_slice->d_preds,	                                // predecessor in
                                graph_slice->d_labels,
                                graph_slice->d_visited_mask,
                                this->work_progress,
                                graph_slice->frontier_elements[selector ^ 1],			// max filtered edge frontier vertices
                                graph_slice->frontier_elements[selector],				// max vertex frontier vertices
                                this->backward_contract_kernel_stats);

                    if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "backward_contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;
                    cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                    queue_index++;
                    selector ^= 1;

                    if (DEBUG) {
                        if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    }
                    if (INSTRUMENT) {
                        if (retval = backward_contract_kernel_stats.Accumulate(
                                    contract_grid_size,
                                    total_runtimes,
                                    total_lifetimes)) break;
                    }

                    // Throttle
                    if ((max_search_depth - iteration[0]) & 1) {
                        if (retval = util::B40CPerror(cudaEventRecord(throttle_event),
                                    "EnactorVertexCentric cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                    } else {
                        if (retval = util::B40CPerror(cudaEventSynchronize(throttle_event),
                                    "EnactorVertexCentric cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                    };

                    //
                    // Backward sum
                    
                    vertex_centric::backward_sum_atomic::Kernel<BackwardSumPolicy>
                        <<<expand_grid_size, BackwardSumPolicy::THREADS>>>(
                                iteration[0],
                                queue_index,											// queue counter index
                                queue_index,											// steal counter index
                                1,														// number of GPUs
                                d_done,
                                graph_slice->frontier_queues.d_keys[selector ^ 1],		// vertex frontier in
                                graph_slice->d_column_indices,
                                graph_slice->d_row_offsets,
                                graph_slice->d_node_values,
                                graph_slice->d_labels,
                                graph_slice->d_sigmas,
                                graph_slice->d_deltas,
                                graph_slice->d_visit_flags,
                                this->work_progress,
                                graph_slice->frontier_elements[selector ^ 1],			// max vertex frontier vertices
                                graph_slice->frontier_elements[selector],				// max edge frontier vertices
                                this->backward_sum_kernel_stats);

                    if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "backward_sum_atomic::Kernel failed ", __FILE__, __LINE__))) break;

                    cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                    queue_index++;
                    selector ^= 1;
                    iteration[0]--;

                    if (INSTRUMENT || DEBUG) {
                        if (work_progress.GetQueueLength(queue_index, queue_length)) break;
                        total_queued += queue_length;
                        if (INSTRUMENT) {
                            if (retval = backward_sum_kernel_stats.Accumulate(
                                        expand_grid_size,
                                        total_runtimes,
                                        total_lifetimes)) break;
                        }
                    }
                }
            }

		} while(0);


		return retval;
	}


    /**
	 * Enacts a breadth-first-search on the specified graph problem. Invokes
	 * new expansion and contraction grid kernels for each BFS iteration.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <typename CsrProblem>
	cudaError_t EnactIterativeSearch(
		CsrProblem 						&csr_problem,
        typename CsrProblem::VertexId 	src_node,
		int 							max_grid_size = 0,
        int                             max_queue_sizing = 1)
	{
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::SizeT 			SizeT;

		// GF100
		if (this->cuda_props.device_sm_version >= 200) {

			// Expansion kernel config
			typedef vertex_centric::expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
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
					ExpandPolicy;


			// Contraction kernel config
			typedef vertex_centric::contract_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
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
			typedef vertex_centric::backward_contract_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
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
			typedef vertex_centric::backward_sum_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
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

			return EnactIterativeSearch<ExpandPolicy, ContractPolicy, BackwardContractPolicy, BackwardSumPolicy>(
				csr_problem, src_node, max_grid_size, max_queue_sizing);
		}

		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidDeviceFunction;
	}
};



} // namespace bc
} // namespace graph
} // namespace b40c
