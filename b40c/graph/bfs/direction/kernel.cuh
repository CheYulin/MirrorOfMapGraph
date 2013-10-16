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
 * BFS direction optimal kernel (fused BFS iterations).
 *
 * Both contraction and expansion phases are fused within the same kernel,
 * separated by software global barriers.  The kernel itself also steps through
 * BFS iterations (without iterative kernel invocations) using software global
 * barriers.
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/direction/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/direction/contract_atomic/kernel.cuh>
#include <b40c/graph/bfs/direction/bottom_up_expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/direction/bottom_up_contract_atomic/kernel.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace direction {


/******************************************************************************
 * Kernel entrypoint
 ******************************************************************************/

/**
 * Contract-expand kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId 		iteration,					// Current BFS iteration
	typename KernelPolicy::VertexId			queue_index,				// Current frontier queue counter index
	typename KernelPolicy::VertexId			steal_index,				// Current workstealing counter index
	typename KernelPolicy::VertexId 		src,						// Source vertex (may be -1 if iteration != 0)
	typename KernelPolicy::VertexId 		*d_edge_frontier,			// Edge frontier
	typename KernelPolicy::VertexId 		*d_vertex_frontier,			// Vertex frontier
	typename KernelPolicy::VertexId 		*d_predecessor,				// Predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
	typename KernelPolicy::VertexId			*d_column_indices,			// CSR column-indices array
	typename KernelPolicy::SizeT			*d_row_offsets,				// CSR row-offsets array
	typename KernelPolicy::VertexId         *d_row_indices,             // CSC row-indices array
	typename KernelPolicy::SizeT            *d_column_offsets,          // CSC column-offsets array
	typename KernelPolicy::VertexId			*d_labels,					// BFS labels to set
	typename KernelPolicy::VisitedMask 		*d_visited_mask,			// Mask for detecting visited status
	bool                                    *d_frontier_bitmap_in,      // BFS frontier bitmap
	bool                                    *d_frontier_bitmap_out,     // BFS frontier bitmap
	util::CtaWorkProgress 					work_progress,				// Atomic workstealing and queueing counters
	typename KernelPolicy::SizeT			max_edge_frontier, 			// Maximum number of elements we can place into the outgoing edge frontier
	typename KernelPolicy::SizeT			max_vertex_frontier, 		// Maximum number of elements we can place into the outgoing vertex frontier
	util::GlobalBarrier						global_barrier,				// Software global barrier
	util::KernelRuntimeStats				kernel_stats,				// Kernel timing statistics (used when KernelPolicy::INSTRUMENT)
	typename KernelPolicy::VertexId			*d_iteration)				// Place to write final BFS iteration count
{
	typedef typename KernelPolicy::ContractKernelPolicy 	        ContractKernelPolicy;
	typedef typename KernelPolicy::ExpandKernelPolicy 		        ExpandKernelPolicy;
	typedef typename KernelPolicy::BottomUpContractKernelPolicy     BottomUpContractKernelPolicy;
	typedef typename KernelPolicy::BottomUpExpandKernelPolicy       BottomUpExpandKernelPolicy;
	typedef typename KernelPolicy::VertexId 				VertexId;
	typedef typename KernelPolicy::SizeT 					SizeT;

	int num_gpus = 1;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStart();
	}

	if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {

		// Reset all counters
		work_progress.template Reset<SizeT>();

		// Determine work decomposition for first iteration
		if (threadIdx.x == 0) {

			// We'll be the only block with active work this iteration.
			// Enqueue the source for us to subsequently process.
			util::io::ModifiedStore<BottomUpExpandKernelPolicy::QUEUE_WRITE_MODIFIER>::St(src, d_edge_frontier);

			if (BottomUpExpandKernelPolicy::MARK_PREDECESSORS) {
				// Enqueue predecessor of source
				VertexId predecessor = -2;
				util::io::ModifiedStore<BottomUpExpandKernelPolicy::QUEUE_WRITE_MODIFIER>::St(predecessor, d_predecessor);
			}

			// Initialize work decomposition in smem
			SizeT num_elements = 1;
			smem_storage.bottomupcontract.state.work_decomposition.template Init<BottomUpContractKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);
		}
	}


	// Barrier to protect work decomposition
	__syncthreads();

	// Don't do workstealing this iteration because without a
	// global barrier after queue-reset, the queue may be inconsistent
	// across CTAs
	bottom_up_contract_atomic::SweepPass<BottomUpContractKernelPolicy, false>::Invoke(
		iteration,
		queue_index,
		steal_index,
		num_gpus,
		d_edge_frontier,
		d_vertex_frontier,
		d_predecessor,
		d_labels,
		work_progress,
		smem_storage.bottomupcontract.state.work_decomposition,
		max_vertex_frontier,
		smem_storage.bottomupcontract);

	queue_index++;
	steal_index++;

	global_barrier.Sync();


		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (BottomUpExpandKernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.bottomupexpand.state.work_decomposition.template Init<BottomUpExpandKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);

		}

		// Barrier to protect work decomposition
		__syncthreads();

		bottom_up_expand_atomic::SweepPass<BottomUpExpandKernelPolicy, BottomUpExpandKernelPolicy::WORK_STEALING>::Invoke(
			queue_index,
			steal_index,
			num_gpus,
			d_vertex_frontier,
			d_edge_frontier,
			d_frontier_bitmap_in,
			d_frontier_bitmap_out,
			d_predecessor,
			d_row_indices,
			d_column_offsets,
			work_progress,
			smem_storage.bottomupexpand.state.work_decomposition,
			max_edge_frontier,
			smem_storage.bottomupexpand);

		iteration++;
		queue_index++;
		steal_index++;

		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Flop
		//---------------------------------------------------------------------

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (BottomUpExpandKernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.bottomupcontract.state.work_decomposition.template Init<BottomUpContractKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		bottom_up_contract_atomic::SweepPass<BottomUpContractKernelPolicy, BottomUpContractKernelPolicy::WORK_STEALING>::Invoke(
			iteration,
			queue_index,
			steal_index,
			num_gpus,
			d_edge_frontier,
			d_vertex_frontier,
			d_predecessor,
			d_labels,
			work_progress,
			smem_storage.bottomupcontract.state.work_decomposition,
			max_vertex_frontier,
			smem_storage.bottomupcontract);

		queue_index++;
		steal_index++;

		global_barrier.Sync();
}

} // namespace direction
} // namespace bfs
} // namespace graph
} // namespace b40c

