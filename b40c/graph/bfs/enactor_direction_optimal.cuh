/******************************************************************************
 * Copyright 2012-2013 Yangzihao Wang
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
 * This part of the code is based on Duane's b40c library, project site:
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 ******************************************************************************/

/******************************************************************************
 * Direction Optimal BFS enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/problem_type.cuh>
#include <b40c/graph/bfs/enactor_base.cuh>

#include <b40c/graph/bfs/direction/kernel.cuh>
#include <b40c/graph/bfs/direction/kernel_policy.cuh>

#include <b40c/graph/bfs/direction/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/direction/expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/direction/contract_atomic/kernel.cuh>
#include <b40c/graph/bfs/direction/contract_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/direction/bottom_up_expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/direction/bottom_up_expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/direction/bottom_up_contract_atomic/kernel.cuh>
#include <b40c/graph/bfs/direction/bottom_up_contract_atomic/kernel_policy.cuh>




namespace b40c {
namespace graph {
namespace bfs {



/**
 * Direction-optimal BFS enactor
 * 
 */
template <bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
class EnactorDirection : public EnactorBase
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
	template <typename CsrCscProblem>
	cudaError_t Setup(
		CsrCscProblem &csr_problem,
		int grid_size)
	{
		typedef typename CsrCscProblem::SizeT 			    SizeT;
		typedef typename CsrCscProblem::VertexId 			VertexId;
		typedef typename CsrCscProblem::VisitedMask 		VisitedMask;

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
			if (retval = global_barrier.Setup(grid_size)) break;

			// Make sure our runtime stats are initialized
			if (retval = expand_kernel_stats.Setup(grid_size)) break;
			if (retval = contract_kernel_stats.Setup(grid_size)) break;

			// Reset statistics
			iteration[0] 		= 0;
			total_runtimes 		= 0;
			total_lifetimes 	= 0;
			total_queued 		= 0;
			done[0] 			= -1;

			// Single-gpu graph slice
			typename CsrCscProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Bind bitmask texture
			int bytes = (graph_slice->nodes + 8 - 1) / 8;
			cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
				    direction::contract_atomic::BitmaskTex<VisitedMask>::ref,
					graph_slice->d_visited_mask,
					bitmask_desc,
					bytes),
				"EnactorDirection cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind row-offsets texture
			cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					direction::expand_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorDirection cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}


public: 	
	
	/**
	 * Constructor
	 */
	EnactorDirection(bool DEBUG = false) :
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
	virtual ~EnactorDirection()
	{
		if (iteration) {
			util::B40CPerror(cudaFreeHost((void *) iteration), "EnactorDirection cudaFreeHost iteration failed", __FILE__, __LINE__);
		}
		if (done) {
			util::B40CPerror(cudaFreeHost((void *) done),
					"EnactorDirection cudaFreeHost done failed", __FILE__, __LINE__);

			util::B40CPerror(cudaEventDestroy(throttle_event),
				"EnactorDirection cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
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
	 * Enacts a breadth-first-search on the specified graph problem.  Invokes
	 * a single grid kernel that itself steps over BFS iterations.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
		typename KernelPolicy,
		typename CsrCscProblem>
	cudaError_t EnactSearch(
		CsrCscProblem 						&csr_problem,
		typename CsrCscProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		typedef typename CsrCscProblem::SizeT 			SizeT;
		typedef typename CsrCscProblem::VertexId 		VertexId;
		typedef typename CsrCscProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;

		do {

			// Determine grid size
			int occupancy = KernelPolicy::CTA_OCCUPANCY;
			int grid_size = MaxGridSize(occupancy, max_grid_size);

			if (DEBUG) {
				printf("DEBUG: BFS occupancy %d, grid size %d\n", occupancy, grid_size);
				fflush(stdout);
			}

			// Single-gpu graph slice
			typename CsrCscProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Lazy initialization
			if (retval = Setup(csr_problem, grid_size)) break;

			// Initiate single-grid kernel
		    direction::Kernel<KernelPolicy>
					<<<grid_size, KernelPolicy::THREADS>>>(
				0,												// iteration
				0,												// queue_index
				0,												// steal_index
				src,
				graph_slice->frontier_queues.d_keys[1],			// edge frontier
				graph_slice->frontier_queues.d_keys[0],			// vertex frontier
				graph_slice->frontier_queues.d_values[1],		// predecessor edge frontier
				graph_slice->d_column_indices,
				graph_slice->d_row_offsets,
				graph_slice->d_row_indices,
				graph_slice->d_column_offsets,
				graph_slice->d_labels,
				graph_slice->d_visited_mask,
				graph_slice->d_frontier_bitmap_in,
				graph_slice->d_frontier_bitmap_out,
				this->work_progress,
				graph_slice->frontier_elements[1],				// max edge frontier vertices
				graph_slice->frontier_elements[0],				// max vertex frontier vertices
				this->global_barrier,
				this->expand_kernel_stats,
				(VertexId *) d_iteration);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor Direction Optimal Kernel failed ", __FILE__, __LINE__))) break;

			if (INSTRUMENT) {
				// Get stats
				if (retval = expand_kernel_stats.Accumulate(
					grid_size,
					total_runtimes,
					total_lifetimes,
					total_queued)) break;
			}

			// Check if any of the frontiers overflowed due to redundant expansion
			bool overflowed = false;
			if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
			if (overflowed) {
				retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
				break;
			}

		} while (0);

		return retval;
	}


    /**
 	 * Enacts a breadth-first-search on the specified graph problem.  Invokes
	 * a single grid kernel that itself steps over BFS iterations.
 	 *
 	 * @return cudaSuccess on success, error enumeration otherwise
 	 */
     template <typename CsrCscProblem>
 	cudaError_t EnactSearch(
 		CsrCscProblem 						&csr_problem,
 		typename CsrCscProblem::VertexId 	src,
 		int 							max_grid_size = 0)
 	{
 		if (this->cuda_props.device_sm_version >= 200) {

 			// Fused-iteration direction optimal tuning configuration
 			typedef direction::KernelPolicy<
 				typename CsrCscProblem::ProblemType,
 				200,					// CUDA_ARCH
 				INSTRUMENT, 			// INSTRUMENT
 				0, 						// SATURATION_QUIT

 				// Tunable parameters (generic)
 				8,						// MIN_CTA_OCCUPANCY
 				7,						// LOG_THREADS
 				util::io::ld::cg,		// QUEUE_READ_MODIFIER,
 				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
 				util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
 				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
 				util::io::st::cg,		// QUEUE_WRITE_MODIFIER,

 				// Tunable parameters (contract)
 				0,						// CONTRACT_LOG_LOAD_VEC_SIZE
 				2,						// CONTRACT_LOG_LOADS_PER_TILE
 				5,						// CONTRACT_LOG_RAKING_THREADS
 				false,					// CONTRACT_WORK_STEALING
				3,						// CONTRACT_END_BITMASK_CULL
				6, 						// CONTRACT_LOG_SCHEDULE_GRANULARITY

 				0,						// EXPAND_LOG_LOAD_VEC_SIZE
 				0,						// EXPAND_LOG_LOADS_PER_TILE
 				5,						// EXPAND_LOG_RAKING_THREADS
 				true,					// EXPAND_WORK_STEALING
 				32,						// EXPAND_WARP_GATHER_THRESHOLD
 				128 * 4, 				// EXPAND_CTA_GATHER_THRESHOLD,
 				6> 						// EXPAND_LOG_SCHEDULE_GRANULARITY
					KernelPolicy;

 			return EnactSearch<KernelPolicy, CsrCscProblem>(
 				csr_problem, src, max_grid_size);

 		}

		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidConfiguration;
 	}
    
};


} // namespace bfs
} // namespace graph
} // namespace b40c
