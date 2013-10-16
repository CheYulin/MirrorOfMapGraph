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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * CTA tile-processing abstraction for BFS frontier contraction
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/util/operators.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>

namespace b40c {
namespace graph {
namespace bc {
namespace vertex_centric {
namespace backward_contract_atomic {


/**
 * Templated texture reference for visited mask
 */
template <typename VisitedMask>
struct BitmaskTex
{
	static texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename VisitedMask>
texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> BitmaskTex<VisitedMask>::ref;


/**
 * CTA tile-processing abstraction for BFS frontier contraction
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::ValidFlag		ValidFlag;
	typedef typename KernelPolicy::VisitedMask 		VisitedMask;
	typedef typename KernelPolicy::SizeT 			SizeT;
    typedef typename KernelPolicy::EValue           EValue;

	typedef typename KernelPolicy::RakingDetails 	RakingDetails;
	typedef typename KernelPolicy::SmemStorage		SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	VertexId 				*d_in;						// Incoming edge frontier
	VertexId 				*d_out;						// Outgoing vertex frontier
	VertexId 				*d_predecessor_in;			// Incoming predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
	VertexId				*d_labels;					// BFS labels to set
	VisitedMask 			*d_visited_mask;			// Mask for detecting visited status

	// Work progress
	VertexId 				iteration;					// Current BFS iteration
	VertexId 				queue_index;				// Current frontier queue counter index
	util::CtaWorkProgress	&work_progress;				// Atomic workstealing and queueing counters
	SizeT					max_vertex_frontier;		// Maximum size (in elements) of outgoing vertex frontier
	int 					num_gpus;					// Number of GPUs

	// Operational details for raking scan grid
	RakingDetails 			raking_details;

	// Shared memory for the CTA
	SmemStorage				&smem_storage;

	// Whether or not to perform bitmask culling (incurs extra latency on small frontiers)
	bool 					bitmask_cull;



	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile of incoming edge frontier to process
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
			LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
		};


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
		VertexId 	predecessor_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Whether or not the corresponding vertex_id is valid for exploring
		ValidFlag 	flags[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Global scatter offsets
		SizeT 		ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate over vertex id
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * InitFlags
			 */
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				// Initially valid if vertex-id is valid
				tile->flags[LOAD][VEC] = (tile->vertex_id[LOAD][VEC] == -1) ? 0 : 1;
				tile->ranks[LOAD][VEC] = tile->flags[LOAD][VEC];

				// Next
				Iterate<LOAD, VEC + 1>::InitFlags(tile);
			}

			/**
			 * VertexCull
			 */
			static __device__ __forceinline__ void VertexCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Row index on our GPU (vertex ids are striped across GPUs)
					VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

					// Load label of node
					VertexId label;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						    label,
						    cta->d_labels + row_id);


					if (label != cta->iteration) {
						tile->vertex_id[LOAD][VEC] = -1;
					}
                }

				// Next
				Iterate<LOAD, VEC + 1>::VertexCull(cta, tile);
			}

		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::InitFlags(tile);
			}

			// VertexCull
			static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::VertexCull(cta, tile);
			}
		};



		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile) {}

			// VertexCull
			static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Initializer
		 */
		__device__ __forceinline__ void InitFlags()
		{
			Iterate<0, 0>::InitFlags(this);
		}

		/**
		 * Culls vertices
		 */
		__device__ __forceinline__ void VertexCull(Cta *cta)
		{
			Iterate<0, 0>::VertexCull(cta, this);
		}
		
	};




	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId 				iteration,
		VertexId 				queue_index,
		int						num_gpus,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_predecessor_in,
		VertexId 				*d_labels,
		VisitedMask 			*d_visited_mask,
		util::CtaWorkProgress	&work_progress,
		SizeT					max_vertex_frontier) :

			iteration(iteration),
			queue_index(queue_index),
			num_gpus(num_gpus),
			raking_details(
				smem_storage.state.raking_elements,
				smem_storage.state.warpscan,
				0),
			smem_storage(smem_storage),
			d_in(d_in),
			d_out(d_out),
			d_predecessor_in(d_predecessor_in),
			d_labels(d_labels),
			d_visited_mask(d_visited_mask),
			work_progress(work_progress),
			max_vertex_frontier(max_vertex_frontier),
			bitmask_cull(
				(KernelPolicy::END_BITMASK_CULL < 0) ?
					true : 														// always bitmask cull
					(KernelPolicy::END_BITMASK_CULL == 0) ?
						false : 												// never bitmask cull
						(iteration < KernelPolicy::END_BITMASK_CULL))
	{
		// Initialize history duplicate-filter
		for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelPolicy::THREADS) {
			smem_storage.history[offset] = -1;
		}
	}


	/**
	 * Process a single, full tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::QUEUE_READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_id,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		tile.VertexCull(this);
        		
		// Init valid flags and ranks
		tile.InitFlags();

		// Protect repurposable storage that backs both raking lanes and local cull scratch
		__syncthreads();

		// Scan tile of ranks, using an atomic add to reserve
		// space in the contracted queue, seeding ranks
		util::Sum<SizeT> scan_op;
		SizeT new_queue_offset = util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTileWithEnqueue(
			raking_details,
			tile.ranks,
			work_progress.GetQueueCounter<SizeT>(queue_index + 1),
			scan_op);

		// Check updated queue offset for overflow due to redundant expansion
		if (new_queue_offset >= max_vertex_frontier) {
			work_progress.SetOverflow<SizeT>();
			util::ThreadExit();
		}

		// Scatter directly (without first contracting in smem scratch), predicated
		// on flags
		util::io::ScatterTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::QUEUE_WRITE_MODIFIER>::Scatter(
				d_out,
				tile.vertex_id,
				tile.flags,
				tile.ranks);
	}
};


} // namespace backward_contract_atomic
} // namespace vertex_centric
} // namespace bc
} // namespace graph
} // namespace b40c

