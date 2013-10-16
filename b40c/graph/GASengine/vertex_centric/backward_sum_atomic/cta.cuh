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
 * CTA tile-processing abstraction for BFS backward sum kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace graph {
namespace GASengine {
namespace vertex_centric {
namespace backward_sum_atomic {



    /**
     * CTA tile-processing abstraction for BFS frontier expansion
     */
    template <typename SizeT>
        struct RowOffsetTex
        {
            static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
        };
    template <typename SizeT>
        texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;



    /**
     * Derivation of KernelPolicy that encapsulates tile-processing routines
     */
    template <typename KernelPolicy>
        struct Cta
        {

            /**
             * Helper device functions
             */

            //CTA reduction
            template<typename T>
                static __device__ __forceinline__ T CTAReduce(T* partial)
                {
                    for (size_t s=KernelPolicy::THREADS/2; s>0; s>>=1)
                    {
                        if (threadIdx.x < s)
                            partial[threadIdx.x] += partial[threadIdx.x+s];
                        __syncthreads();
                    }
                    return partial[0];
                }

            //Warp reduction
            template<typename T>
                static __device__ __forceinline__ T WarpReduce(T* partial, size_t warp_id)
                {
                    for (size_t s=B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)/2; s>0; s>>=1)
                    {
                        if (threadIdx.x < warp_id*B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)+s)
                            partial[threadIdx.x] += partial[threadIdx.x+s];
                        __syncthreads();
                    }
                    return partial[warp_id*B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)];
                }

            //---------------------------------------------------------------------
            // Typedefs
            //---------------------------------------------------------------------

            typedef typename KernelPolicy::VertexId 			VertexId;
            typedef typename KernelPolicy::SizeT 				SizeT;
            typedef typename KernelPolicy::EValue               EValue;

            typedef typename KernelPolicy::SmemStorage			SmemStorage;

            typedef typename KernelPolicy::SoaScanOp			SoaScanOp;
            typedef typename KernelPolicy::RakingSoaDetails 	RakingSoaDetails;
            typedef typename KernelPolicy::TileTuple 			TileTuple;

            typedef util::Tuple<
                SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
                      SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> 		RankSoa;

            //---------------------------------------------------------------------
            // Members
            //---------------------------------------------------------------------

            // Input and output device pointers
            int                     iteration;
            VertexId 				*d_in;						// Incoming vertex frontier
            VertexId				*d_column_indices;			// CSR column-indices array
            SizeT					*d_row_offsets;				// CSR row-offsets array
            EValue                  *d_node_values;             // Node values array
            VertexId                *d_labels;                  // Search depth in
            EValue                  *d_sigmas;
            EValue                  *d_deltas;
            SizeT                   *d_visit_flags;             // Global vertex visit flag, preventing value on one vertex being multiple updated

            // Work progress
            VertexId 				queue_index;				// Current frontier queue counter index
            util::CtaWorkProgress	&work_progress;				// Atomic workstealing and queueing counters
            SizeT					max_edge_frontier;			// Maximum size (in elements) of outgoing edge frontier
            int 					num_gpus;					// Number of GPUs

            // Operational details for raking grid
            RakingSoaDetails 		raking_soa_details;

            // Shared memory for the CTA
            SmemStorage				&smem_storage;



            //---------------------------------------------------------------------
            // Helper Structures
            //---------------------------------------------------------------------

            /**
             * Tile of incoming vertex frontier to process
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

                            typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;


                            //---------------------------------------------------------------------
                            // Members
                            //---------------------------------------------------------------------

                            // Dequeued vertex ids
                            VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

                            // Edge list details
                            SizeT		row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
                            SizeT		row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

                            // Global scatter offsets.  Coarse for CTA/warp-based scatters, fine for scan-based scatters
                            SizeT 		fine_count;
                            SizeT		coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
                            SizeT		fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

                            // Progress for expanding scan-based gather offsets
                            SizeT		row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
                            SizeT		progress;

                            //---------------------------------------------------------------------
                            // Helper Structures
                            //---------------------------------------------------------------------

                            /**
                             * Iterate next vector element
                             */
                            template <int LOAD, int VEC, int dummy = 0>
                                struct Iterate
                                {
                                    /**
                                     * Init
                                     */
                                    template <typename Tile>
                                        static __device__ __forceinline__ void Init(Tile *tile)
                                        {
                                            tile->row_length[LOAD][VEC] = 0;
                                            tile->row_progress[LOAD][VEC] = 0;

                                            Iterate<LOAD, VEC + 1>::Init(tile);
                                        }

                                    /**
                                     * Inspect
                                     */
                                    template <typename Cta, typename Tile>
                                        static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
                                        {
                                            if (tile->vertex_id[LOAD][VEC] != -1) {

                                                // Translate vertex-id into local gpu row-id (currently stride of num_gpu)
                                                VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

                                                // Load neighbor row range from d_row_offsets
                                                Vec2SizeT row_range;
                                                row_range.x = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id);
                                                row_range.y = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1);

                                                // Node is previously unvisited: compute row offset and length
                                                tile->row_offset[LOAD][VEC] = row_range.x;
                                                tile->row_length[LOAD][VEC] = row_range.y - row_range.x;
                                            }

                                            tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
                                                tile->row_length[LOAD][VEC] : 0;

                                            tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
                                                0 : tile->row_length[LOAD][VEC];

                                            Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
                                        }


                                    /**
                                     * Expand by CTA
                                     */
                                    template <typename Cta, typename Tile>
                                        static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
                                        {
                                            // CTA-based expansion/loading
                                            while (true) {

                                                // Vie
                                                if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD) {
                                                    cta->smem_storage.state.cta_comm = threadIdx.x;
                                                }

                                                __syncthreads();

                                                // Check
                                                int owner = cta->smem_storage.state.cta_comm;
                                                if (owner == KernelPolicy::THREADS) {
                                                    // No contenders
                                                    break;
                                                }

                                                if (owner == threadIdx.x) {

                                                    // Got control of the CTA: command it
                                                    cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];										// start
                                                    cta->smem_storage.state.warp_comm[0][1] = tile->coarse_row_rank[LOAD][VEC];									// queue rank
                                                    cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob 

                                                    // Unset row length
                                                    tile->row_length[LOAD][VEC] = 0;

                                                    // Unset my command
                                                    cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;	// invalid
                                                }

                                                __syncthreads();

                                                // Read commands
                                                SizeT coop_offset 	= cta->smem_storage.state.warp_comm[0][0];
                                                SizeT coop_rank	 	= cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
                                                SizeT coop_oob 		= cta->smem_storage.state.warp_comm[0][2];

                                                VertexId neighbor_id;

                                                EValue delta_value;
                                                EValue from_sigma_value;
                                                EValue to_sigma_value;
                                                EValue result;
                                                util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                from_sigma_value, cta->d_sigmas + tile->vertex_id[LOAD][VEC]);
                                                while (coop_offset + KernelPolicy::THREADS < coop_oob) {

                                                    // Gather
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);
                                                                                                          
                                                    // Gather sigma value and compute delta value
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            to_sigma_value, cta->d_sigmas + neighbor_id);
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            delta_value, cta->d_deltas + neighbor_id);
                                                    result = 1.0f * from_sigma_value / to_sigma_value * (1.0f + delta_value);

                                                    cta->smem_storage.gather_delta_values[threadIdx.x] += result;
                                                         
                                                    coop_offset += KernelPolicy::THREADS;
                                                    coop_rank += KernelPolicy::THREADS;
                                                }

                                                if (coop_offset + threadIdx.x < coop_oob) {
                                                    // Gather
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);
                        
                                                    // Gather sigma value and compute delta value
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            to_sigma_value, cta->d_sigmas + neighbor_id);
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            delta_value, cta->d_deltas + neighbor_id);
                                                    result = 1.0f * from_sigma_value / to_sigma_value * (1.0f + delta_value);

                                                    cta->smem_storage.gather_delta_values[threadIdx.x] += result;
                                                }
                                            }

                                            EValue final_delta_value = CTAReduce<EValue>(cta->smem_storage.gather_delta_values);
                                            // vie for write into delta_value
                                            if (threadIdx.x == 0)
                                            {
                                            	if (atomicCAS(cta->d_visit_flags+tile->vertex_id[LOAD][VEC], 0, 1) == 0)
                                            	{
                                            		util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                            		final_delta_value, cta->d_deltas+tile->vertex_id[LOAD][VEC]);

                                                    EValue current_delta;
                                            		//accumulate to node values
                                            		util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                            		   current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            		current_delta += final_delta_value;
                                            		util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                            		current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            	}
                                            }

                                            // Next vector element
                                            Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
                                        }

                                    /**
                                     * Expand by warp
                                     */
                                    template <typename Cta, typename Tile>
                                        static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
                                        {
                                            if (KernelPolicy::WARP_GATHER_THRESHOLD < KernelPolicy::CTA_GATHER_THRESHOLD) {

                                                // Warp-based expansion/loading
                                                int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                                                int lane_id = util::LaneId();

                                                while (__any(tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)) {

                                                    if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD) {
                                                        // Vie for control of the warp
                                                        cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
                                                    }

                                                    if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0]) {

                                                        // Got control of the warp
                                                        cta->smem_storage.state.warp_comm[warp_id][0] = tile->row_offset[LOAD][VEC];									// start
                                                        cta->smem_storage.state.warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC];								// queue rank
                                                        cta->smem_storage.state.warp_comm[warp_id][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob

                                                        // Unset row length
                                                        tile->row_length[LOAD][VEC] = 0;
                                                    }

                                                    SizeT coop_offset 	= cta->smem_storage.state.warp_comm[warp_id][0];
                                                    SizeT coop_rank 	= cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
                                                    SizeT coop_oob 		= cta->smem_storage.state.warp_comm[warp_id][2];

                                                    VertexId neighbor_id;
                                                    EValue delta_value;
                                                    EValue from_sigma_value;
                                                    EValue to_sigma_value;
                                                    EValue result;
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                    from_sigma_value, cta->d_sigmas + tile->vertex_id[LOAD][VEC]);

                                                    while (coop_offset  + B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) < coop_oob) {

                                                        // Gather
                                                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                neighbor_id, cta->d_column_indices + coop_offset + lane_id);

                                                        // Gather sigma value and compute delta value
                                                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                to_sigma_value, cta->d_sigmas + neighbor_id);
                                                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                delta_value, cta->d_deltas + neighbor_id);
                                                        result = 1.0f * from_sigma_value / to_sigma_value * (1.0f + delta_value);

                                                        cta->smem_storage.gather_delta_values[warp_id*B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)+lane_id] += result;
                                                        

                                                        coop_offset += B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                                                        coop_rank += B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                                                    }

                                                    if (coop_offset + lane_id < coop_oob) {
                                                        // Gather
                                                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                neighbor_id, cta->d_column_indices + coop_offset + lane_id);


                                                        // Gather sigma value and compute delta value
                                                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                to_sigma_value, cta->d_sigmas + neighbor_id);
                                                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                                delta_value, cta->d_deltas + neighbor_id);
                                                        result = 1.0f * from_sigma_value / to_sigma_value * (1.0f + delta_value);

                                                        cta->smem_storage.gather_delta_values[warp_id*B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)+lane_id] += result;

                                                        EValue final_delta_value = WarpReduce(cta->smem_storage.gather_delta_values, warp_id);
                                                        // vie for write into delta_values
                                                        if (threadIdx.x == 0)
                                                        {
                                                        	//try to set the d_visit_flag in global memory
                                                        	if (atomicCAS(cta->d_visit_flags+tile->vertex_id[LOAD][VEC], 0, 1) == 0)
                                                        	{
                                                        		util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                                        		final_delta_value, cta->d_deltas+tile->vertex_id[LOAD][VEC]);

                                                                EValue current_delta;
                                            		            //accumulate to node values
                                            		            util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                            		               current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            		            current_delta += final_delta_value;
                                            		            util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                            		            current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            		        }
                                                        }
                                                        

                                                    }

                                                }

                                                // Next vector element
                                                Iterate<LOAD, VEC + 1>::ExpandByWarp(cta, tile);
                                            }
                                        }


                                    /**
                                     * Expand by scan
                                     */
                                    template <typename Cta, typename Tile>
                                        static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
                                        {
                                            // Attempt to make further progress on this dequeued item's neighbor
                                            // list if its current offset into local scratch is in range
                                            SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;

                                            EValue edge_value;
                                            EValue delta_value;
                                            EValue from_sigma_value;
                                            EValue to_sigma_value;
                                            EValue result = 0;
                                            VertexId neighbor_id;
                                            VertexId dst_depth;
                                            util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                    from_sigma_value, cta->d_sigmas + tile->vertex_id[LOAD][VEC]);


                                            while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
                                                    (scratch_offset < SmemStorage::GATHER_ELEMENTS))
                                            {
                                                // Put gather offset into scratch space
                                                cta->smem_storage.gather_offsets[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];

                                                util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                        neighbor_id, cta->d_column_indices + cta->smem_storage.gather_offsets[scratch_offset]);
                                                util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                        dst_depth, cta->d_labels + neighbor_id);
                                                if (dst_depth == cta->iteration + 1)
                                                {
                                                    // Gather sigma value and compute delta value
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            to_sigma_value, cta->d_sigmas + neighbor_id);
                                                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                            delta_value, cta->d_deltas + neighbor_id);
                                                    result += 1.0f * from_sigma_value / to_sigma_value * (1.0f + delta_value);
                                                }
                                                tile->row_progress[LOAD][VEC]++;
                                                scratch_offset++;
                                            }

                                            
                                            SizeT visit_flag;
                                            util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                    visit_flag,
                                                    cta->d_visit_flags + tile->vertex_id[LOAD][VEC]);
                                            
                                            if (visit_flag == 0)
                                            {
                                                util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                                            result, cta->d_deltas+tile->vertex_id[LOAD][VEC]);
                                                if (tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC])
                                                {
                                                    //still has content to write, set visit_flag to 1
                                                    atomicCAS(cta->d_visit_flags+tile->vertex_id[LOAD][VEC], 0, 1);
                                                    EValue current_delta;
                                            		util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                            		   current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            		current_delta += result;
                                            		util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                            		    current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                                }
                                                else
                                                {
                                                    //finished write vertex value, set visit_flag to 2
                                                    atomicCAS(cta->d_visit_flags+tile->vertex_id[LOAD][VEC], 0, 2);
                                                    //Accumulate into node_values
                                                    EValue current_delta;
                                            		util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                            		   current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            		current_delta += result;
                                            		util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                            		    current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                                }
                                            }
                                            else if (visit_flag == 1)
                                            {
                                                util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                                        edge_value,
                                                        cta->d_deltas+tile->vertex_id[LOAD][VEC]);
                                                result += edge_value;
                                                util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                                        result, cta->d_deltas+tile->vertex_id[LOAD][VEC]);

                                                atomicCAS(cta->d_visit_flags+tile->vertex_id[LOAD][VEC], 1, 2);
                                                //Accumulate into node_values
                                                EValue current_delta;
                                            	util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
                                            	   current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            	current_delta += result;
                                            	util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
                                            	current_delta, cta->d_node_values + tile->vertex_id[LOAD][VEC]);
                                            }
                                            //otherwise it's a repeating visit to old vertex, do nothing


                                            // Next vector element
                                            Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
                                            }
                                        };


                                    /**
                                     * Iterate next load
                                     */
                                    template <int LOAD, int dummy>
                                        struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
                                        {
                                            /**
                                             * Init
                                             */
                                            template <typename Tile>
                                                static __device__ __forceinline__ void Init(Tile *tile)
                                                {
                                                    Iterate<LOAD + 1, 0>::Init(tile);
                                                }

                                            /**
                                             * Inspect
                                             */
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
                                                {
                                                    Iterate<LOAD + 1, 0>::Inspect(cta, tile);
                                                }

                                            /**
                                             * Expand by CTA
                                             */
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
                                                {
                                                    Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
                                                }

                                            /**
                                             * Expand by warp
                                             */
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
                                                {
                                                    Iterate<LOAD + 1, 0>::ExpandByWarp(cta, tile);
                                                }

                                            /**
                                             * Expand by scan
                                             */
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
                                                {
                                                    Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
                                                }
                                        };

                                    /**
                                     * Terminate
                                     */
                                    template <int dummy>
                                        struct Iterate<LOADS_PER_TILE, 0, dummy>
                                        {
                                            // Init
                                            template <typename Tile>
                                                static __device__ __forceinline__ void Init(Tile *tile) {}

                                            // Inspect
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

                                            // ExpandByCta
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile) {}

                                            // ExpandByWarp
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile) {}

                                            // ExpandByScan
                                            template <typename Cta, typename Tile>
                                                static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile) {}
                                        };


                                    //---------------------------------------------------------------------
                                    // Interface
                                    //---------------------------------------------------------------------

                                    /**
                                     * Constructor
                                     */
                                    __device__ __forceinline__ Tile()
                                    {
                                        Iterate<0, 0>::Init(this);
                                    }

                                    /**
                                     * Inspect dequeued vertices, updating label if necessary and
                                     * obtaining edge-list details
                                     */
                                    template <typename Cta>
                                        __device__ __forceinline__ void Inspect(Cta *cta)
                                        {
                                            Iterate<0, 0>::Inspect(cta, this);
                                        }

                                    /**
                                     * Expands neighbor lists for valid vertices at CTA-expansion granularity
                                     */
                                    template <typename Cta>
                                        __device__ __forceinline__ void ExpandByCta(Cta *cta)
                                        {
                                            Iterate<0, 0>::ExpandByCta(cta, this);
                                        }

                                    /**
                                     * Expands neighbor lists for valid vertices a warp-expansion granularity
                                     */
                                    template <typename Cta>
                                        __device__ __forceinline__ void ExpandByWarp(Cta *cta)
                                        {
                                            Iterate<0, 0>::ExpandByWarp(cta, this);
                                        }

                                    /**
                                     * Expands neighbor lists by local scan rank
                                     */
                                    template <typename Cta>
                                        __device__ __forceinline__ void ExpandByScan(Cta *cta)
                                        {
                                            Iterate<0, 0>::ExpandByScan(cta, this);
                                        }
                                };


                            //---------------------------------------------------------------------
                            // Methods
                            //---------------------------------------------------------------------

                            /**
                             * Constructor
                             */
                            __device__ __forceinline__ Cta(
                                    int                     iteration,
                                    VertexId 				queue_index,
                                    int						num_gpus,
                                    SmemStorage 			&smem_storage,
                                    VertexId 				*d_in,
                                    VertexId 				*d_column_indices,
                                    SizeT 					*d_row_offsets,
                                    EValue                  *d_node_values,
                                    VertexId                *d_labels,
                                    EValue                  *d_sigmas,
                                    EValue                  *d_deltas,
                                    VertexId                *d_visit_flags,
                                    util::CtaWorkProgress	&work_progress,
                                    SizeT					max_edge_frontier) :

                                iteration(iteration),
                                queue_index(queue_index),
                                num_gpus(num_gpus),
                                smem_storage(smem_storage),
                                raking_soa_details(
                                        typename RakingSoaDetails::GridStorageSoa(
                                            smem_storage.coarse_raking_elements,
                                            smem_storage.fine_raking_elements),
                                        typename RakingSoaDetails::WarpscanSoa(
                                            smem_storage.state.coarse_warpscan,
                                            smem_storage.state.fine_warpscan),
                                        TileTuple(0, 0)),
                                d_in(d_in),
                                d_column_indices(d_column_indices),
                                d_row_offsets(d_row_offsets),
                                d_node_values(d_node_values),
                                d_labels(d_labels),
                                d_sigmas(d_sigmas),
                                d_deltas(d_deltas),
                                d_visit_flags(d_visit_flags),
                                work_progress(work_progress),
                                max_edge_frontier(max_edge_frontier)
                                {
                                    if (threadIdx.x == 0) {
                                        smem_storage.state.cta_comm = KernelPolicy::THREADS;		// invalid
                                        smem_storage.state.overflowed = false;						// valid
                                    }
                                }


                            /**
                             * Process a single tile
                             */
                            __device__ __forceinline__ void ProcessTile(
                                    SizeT cta_offset,
                                    SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
                            {
                                Tile<
                                    KernelPolicy::LOG_LOADS_PER_TILE,
                                KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

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

                                // Inspect dequeued vertices, updating label and obtaining
                                // edge-list details
                                tile.Inspect(this);

                                // Scan tile with carry update in raking threads
                                SoaScanOp scan_op;
                                TileTuple totals;	
                                util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
                                        totals,
                                        raking_soa_details,
                                        RankSoa(tile.coarse_row_rank, tile.fine_row_rank),
                                        scan_op);

                                SizeT coarse_count = totals.t0;
                                tile.fine_count = totals.t1;

                                // Use a single atomic add to reserve room in the queue
                                if (threadIdx.x == 0) {

                                    SizeT enqueue_amt = coarse_count + tile.fine_count;
                                    SizeT enqueue_offset = work_progress.Enqueue(enqueue_amt, queue_index + 1);

                                    smem_storage.state.coarse_enqueue_offset = enqueue_offset;
                                    smem_storage.state.fine_enqueue_offset = enqueue_offset + coarse_count;

                                    // Check for queue overflow due to redundant expansion
                                    if (enqueue_offset + enqueue_amt >= max_edge_frontier) {
                                        smem_storage.state.overflowed = true;
                                        work_progress.SetOverflow<SizeT>();
                                    }
                                }

                                // Protect overflowed flag
                                __syncthreads();

                                // Quit if overflow
                                if (smem_storage.state.overflowed) {
                                    util::ThreadExit();
                                }

                                if (coarse_count > 0)
                                {
                                    // Enqueue valid edge lists into outgoing queue
                                    tile.ExpandByCta(this);

                                    // Enqueue valid edge lists into outgoing queue
                                    tile.ExpandByWarp(this);
                                }

                                //
                                // Enqueue the adjacency lists of unvisited node-IDs by repeatedly
                                // gathering edges into the scratch space, and then
                                // having the entire CTA copy the scratch pool into the outgoing
                                // frontier queue.
                                //

                                tile.progress = 0;
                                while (tile.progress < tile.fine_count) {
                                    // Fill the scratch space with gather-offsets for neighbor-lists.
                                    tile.ExpandByScan(this);
                                    tile.progress += SmemStorage::GATHER_ELEMENTS;
                                    __syncthreads();
                                }
                            }
                        };



} // namespace backward_sum_atomic
} // namespace vertex_centric
} // namespace bc
} // namespace graph
} // namespace b40c

