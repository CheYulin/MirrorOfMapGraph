/******************************************************************************
 * 
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
 ******************************************************************************/

/******************************************************************************
 * BFS atomic expansion kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/graph/bc/vertex_centric/expand_atomic/cta.cuh>

namespace b40c {
namespace graph {
namespace bc {
namespace vertex_centric {
namespace expand_atomic {



/**
 * Expansion pass (non-workstealing)
 */
template <typename KernelPolicy, bool WORK_STEALING>
struct SweepPass
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::VertexId 		&queue_index,
		typename KernelPolicy::VertexId 		&steal_index,
		int								 		&num_gpus,
		typename KernelPolicy::VertexId 		*&d_vertex_frontier,
		typename KernelPolicy::VertexId 		*&d_edge_frontier,
		typename KernelPolicy::VertexId 		*&d_predecessor,
		typename KernelPolicy::VertexId			*&d_column_indices,
		typename KernelPolicy::SizeT			*&d_row_offsets,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
		typename KernelPolicy::SizeT			&max_edge_frontier,
		SmemStorage								&smem_storage)
	{
		typedef Cta<KernelPolicy>			 			Cta;
		typedef typename KernelPolicy::SizeT 			SizeT;

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelPolicy::LOG_TILE_ELEMENTS,
			KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

		// Return if we have no work to do
		if (!work_limits.elements) {
			return;
		}

		// CTA processing abstraction
		Cta cta(
			queue_index,
			num_gpus,
			smem_storage,
			d_vertex_frontier,
			d_edge_frontier,
			d_predecessor,
			d_column_indices,
			d_row_offsets,
			work_progress,
			max_edge_frontier);

		// Process full tiles
		while (work_limits.offset < work_limits.guarded_offset) {

			cta.ProcessTile(work_limits.offset);
			work_limits.offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-i/o
		if (work_limits.guarded_elements) {
			cta.ProcessTile(
				work_limits.offset,
				work_limits.guarded_elements);
		}
	}
};


/**
 * Atomically steal work from a global work progress construct
 */
template <typename SizeT, typename StealIndex>
__device__ __forceinline__ SizeT StealWork(
	util::CtaWorkProgress &work_progress,
	int count,
	StealIndex steal_index)
{
	__shared__ SizeT s_offset;		// The offset at which this CTA performs tile processing, shared by all

	// Thread zero atomically steals work from the progress counter
	if (threadIdx.x == 0) {
		s_offset = work_progress.Steal<SizeT>(count, steal_index);
	}

	__syncthreads();		// Protect offset

	return s_offset;
}


/**
 * Expansion pass (workstealing)
 */
template <typename KernelPolicy>
struct SweepPass <KernelPolicy, true>
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::VertexId 		&queue_index,
		typename KernelPolicy::VertexId 		&steal_index,
		int 									&num_gpus,
		typename KernelPolicy::VertexId 		*&d_vertex_frontier,
		typename KernelPolicy::VertexId 		*&d_edge_frontier,
		typename KernelPolicy::VertexId 		*&d_predecessor,
		typename KernelPolicy::VertexId			*&d_column_indices,
		typename KernelPolicy::SizeT			*&d_row_offsets,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
		typename KernelPolicy::SizeT			&max_edge_frontier,
		SmemStorage								&smem_storage)
	{
		typedef Cta<KernelPolicy> 						Cta;
		typedef typename KernelPolicy::SizeT 			SizeT;

		// CTA processing abstraction
		Cta cta(
			queue_index,
			num_gpus,
			smem_storage,
			d_vertex_frontier,
			d_edge_frontier,
			d_predecessor,
			d_column_indices,
			d_row_offsets,
			work_progress,
			max_edge_frontier);

		// Total number of elements in full tiles
		SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelPolicy::TILE_ELEMENTS - 1));

		// Worksteal full tiles, if any
		SizeT offset;
		while ((offset = StealWork<SizeT>(work_progress, KernelPolicy::TILE_ELEMENTS, steal_index)) < unguarded_elements) {
			cta.ProcessTile(offset);
		}

		// Last CTA does any extra, guarded work (first tile seen)
		if (blockIdx.x == gridDim.x - 1) {
			SizeT guarded_elements = work_decomposition.num_elements - unguarded_elements;
			cta.ProcessTile(unguarded_elements, guarded_elements);
		}
	}
};


/******************************************************************************
 * Arch dispatch
 ******************************************************************************/

/**
 * Not valid for this arch (default)
 */
template <
    typename    KernelPolicy,
    bool        VALID = (__B40C_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
	typedef typename KernelPolicy::VertexId VertexId;
	typedef typename KernelPolicy::SizeT SizeT;
    typedef typename KernelPolicy::EValue EValue;
	typedef typename KernelPolicy::VisitedMask VisitedMask;

	static __device__ __forceinline__ void Kernel(
		VertexId 					&queue_index,
		VertexId 					&steal_index,
		int							&num_gpus,
		volatile int 				*&d_done,
		VertexId 					*&d_vertex_frontier,
		VertexId 					*&d_edge_frontier,
		VertexId 					*&d_predecessor,
		VertexId					*&d_column_indices,
		SizeT						*&d_row_offsets,
		util::CtaWorkProgress 		&work_progress,
		SizeT						&max_vertex_frontier,
		SizeT						&max_edge_frontier,
		util::KernelRuntimeStats	&kernel_stats)
	{
		// empty
	}
};


/**
 * Valid for this arch (policy matches compiler-inserted macro)
 */
template <typename KernelPolicy>
struct Dispatch<KernelPolicy, true>
{
	typedef typename KernelPolicy::VertexId VertexId;
	typedef typename KernelPolicy::SizeT SizeT;
    typedef typename KernelPolicy::EValue EValue;
	typedef typename KernelPolicy::VisitedMask VisitedMask;

	static __device__ __forceinline__ void Kernel(
		VertexId 					&queue_index,
		VertexId 					&steal_index,
		int							&num_gpus,
		volatile int 				*&d_done,
		VertexId 					*&d_vertex_frontier,
		VertexId 					*&d_edge_frontier,
		VertexId 					*&d_predecessor,
		VertexId					*&d_column_indices,
		SizeT						*&d_row_offsets,
		util::CtaWorkProgress 		&work_progress,
		SizeT						&max_vertex_frontier,
		SizeT						&max_edge_frontier,
		util::KernelRuntimeStats	&kernel_stats)
	{

		// Shared storage for the kernel
		__shared__ typename KernelPolicy::SmemStorage smem_storage;

		if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStart();
		}

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);

			// Check if we previously overflowed
			if (num_elements >= max_vertex_frontier) {
				num_elements = 0;
			}

			// Signal to host that we're done
			if (num_elements == 0) {
				if (d_done) d_done[0] = num_elements;
			}

			// Initialize work decomposition in smem
			smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);

		}

		// Barrier to protect work decomposition
		__syncthreads();

		SweepPass<KernelPolicy, KernelPolicy::WORK_STEALING>::Invoke(
			queue_index,
			steal_index,
			num_gpus,
			d_vertex_frontier,
			d_edge_frontier,
			d_predecessor,
			d_column_indices,
			d_row_offsets,
			work_progress,
			smem_storage.state.work_decomposition,
			max_edge_frontier,
			smem_storage);

		if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStop();
			kernel_stats.Flush();
		}
	}
};


/******************************************************************************
 * Expansion Kernel Entrypoint
 ******************************************************************************/

/**
 * Expansion kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId 		queue_index,				// Current frontier queue counter index
	typename KernelPolicy::VertexId 		steal_index,				// Current workstealing counter index
	int										num_gpus,					// Number of GPUs
	volatile int 							*d_done,					// Flag to set when we detect incoming edge frontier is empty
	typename KernelPolicy::VertexId 		*d_vertex_frontier,			// Incoming vertex frontier
	typename KernelPolicy::VertexId 		*d_edge_frontier,			// Outgoing edge frontier
	typename KernelPolicy::VertexId 		*d_predecessor,				// Outgoing predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
	typename KernelPolicy::VertexId			*d_column_indices,			// CSR column-indices array
	typename KernelPolicy::SizeT			*d_row_offsets,				// CSR row-offsets array
	util::CtaWorkProgress 					work_progress,				// Atomic workstealing and queueing counters
	typename KernelPolicy::SizeT			max_vertex_frontier, 		// Maximum number of elements we can place into the outgoing vertex frontier
	typename KernelPolicy::SizeT			max_edge_frontier, 			// Maximum number of elements we can place into the outgoing edge frontier
	util::KernelRuntimeStats				kernel_stats)				// Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
{
	Dispatch<KernelPolicy>::Kernel(
		queue_index,
		steal_index,
		num_gpus,
		d_done,
		d_vertex_frontier,
		d_edge_frontier,
		d_predecessor,
		d_column_indices,
		d_row_offsets,
		work_progress,
		max_vertex_frontier,
		max_edge_frontier,
		kernel_stats);
}

} // namespace expand_atomic
} // namespace vertex_centric
} // namespace bc
} // namespace graph
} // namespace b40c

