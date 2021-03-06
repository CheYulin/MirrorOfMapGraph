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
 * Configuration policy for partitioning upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace partition {
namespace upsweep {


/**
 * A detailed partitioning upsweep kernel configuration policy type that specializes kernel
 * code for a specific pass. It encapsulates tuning configuration
 * policy details derived from TuningPolicy
 */
template <typename TuningPolicy>
struct KernelPolicy : TuningPolicy
{
	enum {		// N.B.: We use an enum type here b/c of a NVCC-win compiler bug involving ternary expressions in static-const fields

		BINS 								= 1 << TuningPolicy::LOG_BINS,
		THREADS								= 1 << TuningPolicy::LOG_THREADS,

		LOG_WARPS							= TuningPolicy::LOG_THREADS - B40C_LOG_WARP_THREADS_BFS(TuningPolicy::CUDA_ARCH),
		WARPS								= 1 << LOG_WARPS,

		LOAD_VEC_SIZE						= 1 << TuningPolicy::LOG_LOAD_VEC_SIZE,
		LOADS_PER_TILE						= 1 << TuningPolicy::LOG_LOADS_PER_TILE,

		LOG_TILE_ELEMENTS_PER_THREAD		= TuningPolicy::LOG_LOAD_VEC_SIZE + TuningPolicy::LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD			= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 					= LOG_TILE_ELEMENTS_PER_THREAD + TuningPolicy::LOG_THREADS,
		TILE_ELEMENTS						= 1 << LOG_TILE_ELEMENTS,

		// A shared-memory composite counter lane is a row of 32-bit words, one word per thread, each word a
		// composite of four 8-bit bin counters.  I.e., we need one lane for every four distribution bins.

		LOG_COMPOSITE_LANES 				= (TuningPolicy::LOG_BINS >= 2) ?
												TuningPolicy::LOG_BINS - 2 :
												0,	// Always at least one lane
		COMPOSITE_LANES 					= 1 << LOG_COMPOSITE_LANES,
	
		LOG_COMPOSITES_PER_LANE				= TuningPolicy::LOG_THREADS,				// Every thread contributes one partial for each lane
		COMPOSITES_PER_LANE 				= 1 << LOG_COMPOSITES_PER_LANE,
	
		// To prevent bin-counter overflow, we must partially-aggregate the
		// 8-bit composite counters back into SizeT-bit registers periodically.  Each lane
		// is assigned to a warp for aggregation.  Each lane is therefore equivalent to
		// four rows of SizeT-bit bin-counts, each the width of a warp.
	
		LOG_LANES_PER_WARP					= B40C_MAX(0, LOG_COMPOSITE_LANES - LOG_WARPS),
		LANES_PER_WARP 						= 1 << LOG_LANES_PER_WARP,
	
		LOG_COMPOSITES_PER_LANE_PER_THREAD 	= LOG_COMPOSITES_PER_LANE - B40C_LOG_WARP_THREADS_BFS(TuningPolicy::CUDA_ARCH),					// Number of partials per thread to aggregate
		COMPOSITES_PER_LANE_PER_THREAD 		= 1 << LOG_COMPOSITES_PER_LANE_PER_THREAD,
	
		AGGREGATED_ROWS						= BINS,
		AGGREGATED_PARTIALS_PER_ROW 		= B40C_WARP_THREADS_BFS(TuningPolicy::CUDA_ARCH),
		PADDED_AGGREGATED_PARTIALS_PER_ROW 	= AGGREGATED_PARTIALS_PER_ROW + 1,

		// Unroll tiles in batches of X elements per thread (X = log(255) is maximum without risking overflow)
		LOG_UNROLL_COUNT 					= 6 - LOG_TILE_ELEMENTS_PER_THREAD,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		union {
			// Composite counter storage
			union {
				char counters[COMPOSITE_LANES][THREADS][4];
				int words[COMPOSITE_LANES][THREADS];
			} composite_counters;

			// Final bin reduction storage
			typename TuningPolicy::SizeT aggregate[AGGREGATED_ROWS][PADDED_AGGREGATED_PARTIALS_PER_ROW];
		};
	};
	
	enum {
		THREAD_OCCUPANCY					= B40C_SM_THREADS(TuningPolicy::CUDA_ARCH) >> TuningPolicy::LOG_THREADS,
		SMEM_OCCUPANCY						= B40C_SMEM_BYTES(TuningPolicy::CUDA_ARCH) / sizeof(SmemStorage),
		MAX_CTA_OCCUPANCY					= B40C_MIN(B40C_SM_CTAS(TuningPolicy::CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),

		VALID								= (MAX_CTA_OCCUPANCY > 0),
	};
};



} // namespace upsweep
} // namespace partition
} // namespace b40c

