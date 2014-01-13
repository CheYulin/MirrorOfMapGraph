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
 * CTA tile-processing abstraction for BFS frontier expansion
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

namespace b40c
{
  namespace graph
  {
    namespace GASengine
    {
      namespace vertex_centric
      {
        namespace gather
        {

          /**
           * CTA tile-processing abstraction for BFS frontier expansion
           */
          template<typename SizeT>
          struct RowOffsetTex
          {
            static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
          };
          template<typename SizeT>
          texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;

          /**
           * Derivation of KernelPolicy that encapsulates tile-processing routines
           */
          template<typename KernelPolicy, typename Program>
          struct Cta
          {

            /**
             * Helper device functions
             */

            //CTA reduction
            template<typename T>
            static __device__                                                        __forceinline__ T CTAReduce(T* partial)
            {
              for (size_t s = KernelPolicy::THREADS / 2; s > 0; s >>= 1)
              {
            	typename Program::gather_sum gather_sum_functor;
                if (threadIdx.x < s) partial[threadIdx.x] = gather_sum_functor(partial[threadIdx.x], partial[threadIdx.x + s]);
                __syncthreads();
              }
              return partial[0];
            }

            template<typename T>
            static __device__                                                        __forceinline__ T CTAReduceMIN(T* partial)
            {
              for (size_t s = KernelPolicy::THREADS / 2; s > 0; s >>= 1)
              {
                if (threadIdx.x < s) partial[threadIdx.x] = min(partial[threadIdx.x], partial[threadIdx.x + s]);
                __syncthreads();
              }
              return partial[0];
            }

            //Warp reduction
            template<typename T>
            static __device__                                                        __forceinline__ T WarpReduce(T* partial, size_t warp_id)
            {
              for (size_t s = B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) / 2; s > 0; s >>= 1)
              {
            	typename Program::gather_sum gather_sum_functor;
                if (threadIdx.x < warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + s) partial[threadIdx.x] = gather_sum_functor(partial[threadIdx.x], partial[threadIdx.x + s]);
                __syncthreads();
              }
              return partial[warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)];
            }

//            //Warp reduction
//            template<typename T>
//            static __device__                                          __forceinline__ T WarpReduceMIN(T* partial, size_t warp_id)
//            {
//              for (size_t s = B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) / 2; s > 0; s >>= 1)
//              {
//                if (threadIdx.x < warp_id * B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + s) partial[threadIdx.x] = min(partial[threadIdx.x], partial[threadIdx.x + s]);
//                __syncthreads();
//              }
//              return partial[warp_id * B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)];
//            }
            //Warp reduction
            template<typename T>
            static __device__                                                          __forceinline__ T WarpReduceMIN(T* partial, size_t warp_id)
            {
              for (size_t s = B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) / 2; s > 0; s >>= 1)
              {
                if (threadIdx.x < warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + s) partial[threadIdx.x] = min(partial[threadIdx.x], partial[threadIdx.x + s]);
                __syncthreads();
              }
              return partial[warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH)];
            }

            //---------------------------------------------------------------------
            // Typedefs
            //---------------------------------------------------------------------

            typedef typename KernelPolicy::VertexId VertexId;
            typedef typename KernelPolicy::SizeT SizeT;
            typedef typename KernelPolicy::EValue EValue;

            typedef typename KernelPolicy::SmemStorage SmemStorage;

            typedef typename KernelPolicy::SoaScanOp SoaScanOp;
            typedef typename KernelPolicy::RakingSoaDetails RakingSoaDetails;
            typedef typename KernelPolicy::TileTuple TileTuple;
            typedef typename KernelPolicy::VertexType VertexType;

            typedef util::Tuple<SizeT (*)[KernelPolicy::LOAD_VEC_SIZE], SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> RankSoa;

            //---------------------------------------------------------------------
            // Members
            //---------------------------------------------------------------------

            // Input and output device pointers
            VertexId *d_in;						// Incoming vertex frontier
            VertexId *d_out;						// Outgoing edge frontier
            VertexId *d_labels;                 // BFS labels to set
//            EValue *d_dists; //
//            EValue* d_gather_results; //gather results
//            int *d_changed; //changed flag
//            SizeT *d_num_out_edges; // number of out edges
//            SizeT *d_visit_flags;             // Global vertex visit flag, preventing value on one vertex being multiple updated
//            VertexId *d_predecessor_out;						// Outgoing predecessor edge frontier
            VertexId *d_column_indices;			// CSR column-indices array
            SizeT *d_row_offsets;				// CSR row-offsets array
            VertexType vertex_list;

            // Work progress
            VertexId queue_index;			// Current frontier queue counter index
            util::CtaWorkProgress &work_progress;	// Atomic workstealing and queueing counters
            SizeT max_edge_frontier;	// Maximum size (in elements) of outgoing edge frontier
            int num_gpus;					// Number of GPUs

            // Operational details for raking grid
            RakingSoaDetails raking_soa_details;

            // Shared memory for the CTA
            SmemStorage &smem_storage;

            //---------------------------------------------------------------------
            // Helper Structures
            //---------------------------------------------------------------------

            /**
             * Tile of incoming vertex frontier to process
             */
            template<int LOG_LOADS_PER_TILE, int LOG_LOAD_VEC_SIZE>
            struct Tile
            {
              //---------------------------------------------------------------------
              // Typedefs and Constants
              //---------------------------------------------------------------------

              enum
              {
                LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE, LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE
              };

              typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;

              //---------------------------------------------------------------------
              // Members
              //---------------------------------------------------------------------

              // Dequeued vertex ids
              VertexId vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

              // Edge list details
              SizeT row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
              SizeT row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

              // Global scatter offsets.  Coarse for CTA/warp-based scatters, fine for scan-based scatters
              SizeT fine_count;
              SizeT coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
              SizeT fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

              // Progress for expanding scan-based gather offsets
              SizeT row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];
              SizeT progress;

              //---------------------------------------------------------------------
              // Helper Structures
              //---------------------------------------------------------------------

              /**
               * Iterate next vector element
               */
              template<int LOAD, int VEC, int dummy = 0>
              struct Iterate
              {
                /**
                 * Init
                 */
                template<typename Tile>
                static __device__ __forceinline__ void Init(Tile *tile)
                {
                  tile->row_length[LOAD][VEC] = 0;
                  tile->row_progress[LOAD][VEC] = 0;
                  Iterate<LOAD, VEC + 1>::Init(tile);
                }

                /**
                 * Inspect
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
                {
                  if (tile->vertex_id[LOAD][VEC] != -1)
                  {

                    // Translate vertex-id into local gpu row-id (currently stride of num_gpu)
                    VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

                    // Load neighbor row range from d_row_offsets
                    Vec2SizeT row_range;
                    row_range.x = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id);
                    row_range.y = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1);

                    // Node is previously unvisited: compute row offset and length
                    tile->row_offset[LOAD][VEC] = row_range.x;
                    tile->row_length[LOAD][VEC] = row_range.y - row_range.x;

//                    if (tile->vertex_id[LOAD][VEC] == 38) printf("Inspect: tile->vertex_id[LOAD][VEC]=%d, row_range.x=%d, row_range.y=%d\n", tile->vertex_id[LOAD][VEC], row_range.x, row_range.y);
                  }

                  tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ? tile->row_length[LOAD][VEC] : 0;

                  tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ? 0 : tile->row_length[LOAD][VEC];

//                  if (tile->vertex_id[LOAD][VEC] == 38)
//                    printf("Inspect: tile->vertex_id[LOAD][VEC]=%d, tile->fine_row_rank[LOAD][VEC]=%d, tile->coarse_row_rank[LOAD][VEC]=%d\n", tile->vertex_id[LOAD][VEC],
//                        tile->fine_row_rank[LOAD][VEC], tile->coarse_row_rank[LOAD][VEC]);

                  Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
                }

                /**
                 * Expand by CTA
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
                {
                  // CTA-based expansion/loading
                  while (true)
                  {
//                    int changed;
//                    util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(changed, cta->d_changed + tile->vertex_id[LOAD][VEC]);
                    cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;
                    __syncthreads();

                    // Vie
                    if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD) //qualify for cta and not visited
                    {
                      cta->smem_storage.state.cta_comm = threadIdx.x;
                    }

                    __syncthreads();

//                    printf("Gather_CTA280: bidx=%d, tidx=%d, tile->row_length[LOAD][VEC]=%d, cta->smem_storage.state.cta_comm=%d, CTA_GATHER_THRESHOLD=%d\n", blockIdx.x, threadIdx.x,
//                        tile->row_length[LOAD][VEC], cta->smem_storage.state.cta_comm, KernelPolicy::CTA_GATHER_THRESHOLD);

                    // Check
                    int owner = cta->smem_storage.state.cta_comm;
                    if (owner == KernelPolicy::THREADS)
                    {
                      // No contenders
                      break;
                    }

                    for (int i = threadIdx.x; i < KernelPolicy::SmemStorage::GATHER_ELEMENTS; i += blockDim.x * gridDim.x)
                    {
                      cta->smem_storage.gather_delta_values[i] = 100000000;
                    }

//                    __syncthreads();

                    if (owner == threadIdx.x)
                    {

                      // Got control of the CTA: command it
                      cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];			// start
                      cta->smem_storage.state.warp_comm[0][1] = tile->coarse_row_rank[LOAD][VEC];	// queue rank
                      cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];	// oob
                      cta->smem_storage.state.warp_comm[0][3] = tile->vertex_id[LOAD][VEC];	// predecessor

                      // Unset row length
                      tile->row_length[LOAD][VEC] = 0;

                      // Unset my command
                      cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;	// invalid
                    }

                    __syncthreads();

                    // Read commands
                    SizeT coop_offset = cta->smem_storage.state.warp_comm[0][0];
                    SizeT coop_rank = cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
                    SizeT coop_oob = cta->smem_storage.state.warp_comm[0][2];

//                    printf("Gather_CTA310: vid=%d, coop_offset=%d, coop_rank=%d, coop_oob=%d\n", tile->vertex_id[LOAD][VEC], coop_offset, coop_rank, coop_oob);

//                    VertexId predecessor_id = cta->smem_storage.state.warp_comm[0][3];
                    VertexId row_id = cta->smem_storage.state.warp_comm[0][3];

                    VertexId neighbor_id;
//                    EValue nb_dist;
                    EValue new_dist;
//                    SizeT num_out_edge;
                    int iter_count = 0;

                    while (coop_offset + threadIdx.x < coop_oob)
                    {

                      // Gather
                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);

                      typename Program::gather_edge gather_edge_functor;
				      gather_edge_functor(row_id, neighbor_id, cta->vertex_list, new_dist);

					  typename Program::gather_sum gather_sum_functor;
					  cta->smem_storage.gather_delta_values[threadIdx.x] = gather_sum_functor(cta->smem_storage.gather_delta_values[threadIdx.x], new_dist);
//
//                      // Scatter neighbor
////                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(neighbor_id, cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
//
//                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(nb_dist, cta->d_dists + neighbor_id);
////                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(num_out_edge, cta->d_num_out_edges + neighbor_id);
//                      new_dist = nb_dist + 1;
//
////                      if (iter_count == 0)
////                        cta->smem_storage.gather_delta_values[threadIdx.x] = new_dist;
////                      else
//                      cta->smem_storage.gather_delta_values[threadIdx.x] = min(new_dist, cta->smem_storage.gather_delta_values[threadIdx.x]);

//                      printf("Gather_CTA323: bidx=%d, tidx=%d, vid=%d, neighbor_id=%d, gather_delta_values=%d\n", blockIdx.x, threadIdx.x, tile->vertex_id[LOAD][VEC], neighbor_id,
//                          cta->smem_storage.gather_delta_values[threadIdx.x]);

//                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(predecessor_id, cta->d_predecessor_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
//                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(new_dist, cta->d_dists + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

                      coop_offset += KernelPolicy::THREADS;
                      coop_rank += KernelPolicy::THREADS;
                      iter_count++;
                    }

                    typename Program::GatherType final_delta_value = CTAReduce<EValue>(cta->smem_storage.gather_delta_values);
                    // vie for write into delta_value
                    if (threadIdx.x == 0)
                    {
                    	typename Program::gather_vertex gather_vertex_functor;
                    	gather_vertex_functor(row_id, final_delta_value, cta->vertex_list);
//                    if (atomicCAS(cta->d_visit_flags + tile->vertex_id[LOAD][VEC], 0, 1) == 0)
//                      if (atomicCAS(cta->d_visit_flags + row_id, 0, 1) == 0)
//                      {
//                      util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(final_delta_value, cta->d_gather_results + row_id);

//                        EValue current_dist, final_dist;
//                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(current_dist, cta->d_dists + row_id);
//                        final_dist = min(current_dist, final_delta_value);

//                        if (current_dist != final_dist)
//                        {
//                        util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(current_dist, cta->d_dists + tile->vertex_id[LOAD][VEC]);
//                        util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(1, cta->d_changed + tile->vertex_id[LOAD][VEC]);
//                          util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(final_dist, cta->d_dists + row_id);
//                          util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(3, cta->d_changed + row_id);
//                          atomicExch(cta->d_changed + row_id, 3);
//                        }
//                        else
//                        {
//                        util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(0, cta->d_changed + tile->vertex_id[LOAD][VEC]);
//                          util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(2, cta->d_changed + row_id);
//                          atomicExch(cta->d_changed + row_id, 2);
//                        }
//                    if(tile->vertex_id[LOAD][VEC] == 8191)
//                        if (row_id == 1049) printf("Gather_CTA376: bidx=%d, tidx=%d, vid=%d, current_dist=%d, final_dist=%d\n", blockIdx.x, threadIdx.x, row_id, current_dist, final_dist);

//                      }
                    }
                  }
                  // Next vector element
                  Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
                }

                /**
                 * Expand by warp
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
                {
//                  if(threadIdx.x==0)
//                  printf("Gather_WARP412: bidx=%d, tidx=%d, vid=%d, WARP_GATHER_THRESHOLD=%d, CTA_GATHER_THRESHOLD=%d\n", blockIdx.x, threadIdx.x, tile->vertex_id[LOAD][VEC],
//                      KernelPolicy::WARP_GATHER_THRESHOLD, KernelPolicy::CTA_GATHER_THRESHOLD);
                  if (KernelPolicy::WARP_GATHER_THRESHOLD < KernelPolicy::CTA_GATHER_THRESHOLD)
                  {
                    // Warp-based expansion/loading
                    int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                    int lane_id = util::LaneId();

                    while (__any(tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)) //qualify for warp
                    {

                      for (int i = threadIdx.x; i < KernelPolicy::SmemStorage::GATHER_ELEMENTS; i += blockDim.x * gridDim.x)
                      {
                        cta->smem_storage.gather_delta_values[i] = 100000000;
                      }

                      if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)
                      {
//                        printf("Gather_WARP422: vid=%d, tile->row_length[LOAD][VEC]=%d, lane_id=%d\n", tile->vertex_id[LOAD][VEC], tile->row_length[LOAD][VEC], lane_id);
                        // Vie for control of the warp
                        cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
                      }

                      if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0])
                      {
                        // Got control of the warp
                        cta->smem_storage.state.warp_comm[warp_id][0] = tile->row_offset[LOAD][VEC];	// start
                        cta->smem_storage.state.warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC];	// queue rank
                        cta->smem_storage.state.warp_comm[warp_id][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];	// oob
                        cta->smem_storage.state.warp_comm[warp_id][3] = tile->vertex_id[LOAD][VEC];	// predecessor
//                        printf("Gather_WARP348: vid=%d, row_offset=%d, coarse_row_rank=%d, row_length=%d\n", tile->vertex_id[LOAD][VEC], cta->smem_storage.state.warp_comm[warp_id][0],
//                            cta->smem_storage.state.warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC], cta->smem_storage.state.warp_comm[warp_id][2]);

                        // Unset row length
                        tile->row_length[LOAD][VEC] = 0;
                      }

                      SizeT coop_offset = cta->smem_storage.state.warp_comm[warp_id][0];
                      SizeT coop_rank = cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
                      SizeT coop_oob = cta->smem_storage.state.warp_comm[warp_id][2];
                      VertexId row_id = cta->smem_storage.state.warp_comm[warp_id][3];

                      // VertexId predecessor_id = cta->smem_storage.state.warp_comm[warp_id][3];

                      VertexId neighbor_id;
                      EValue new_dist;
//                      SizeT num_out_edge;
                      while (coop_offset + lane_id < coop_oob)
                      {

                    	  typename Program::gather_edge gather_edge_functor;
						  gather_edge_functor(row_id, neighbor_id, cta->vertex_list, new_dist);

						  typename Program::gather_sum gather_sum_functor;
						  cta->smem_storage.gather_delta_values[threadIdx.x] = gather_sum_functor(cta->smem_storage.gather_delta_values[threadIdx.x], new_dist);
//                        // Gather
//                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, cta->d_column_indices + coop_offset + lane_id);
//
//                        // Scatter neighbor
//                        //util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(neighbor_id, cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
//
//                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(nb_dist, cta->d_dists + neighbor_id);
////                        util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(num_out_edge, cta->d_num_out_edges + neighbor_id);
//                        new_dist = nb_dist + 1;
//
////                        printf("Gather_WARP465: bidx=%d, tidx=%d, vid=%d, neighbor_id=%d, nb_dist=%d, s_dist=%d, new_dist=%d\n", blockIdx.x, threadIdx.x, row_id, neighbor_id, nb_dist,
////                            cta->smem_storage.gather_delta_values[warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + lane_id], new_dist);
//
//                        cta->smem_storage.gather_delta_values[warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + lane_id] = min(new_dist,cta->smem_storage.gather_delta_values[warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + lane_id]);

                        // Scatter predecessor
                        //util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(predecessor_id, cta->d_predecessor_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);

                        coop_offset += B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                        coop_rank += B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH);
                      }

//                      if (row_id == 1049)
//                        printf("bidx=%d, tidx=%d, warp_id=%d, lane_id=%d, smem_storage.gather_delta_values[%d]=%d\n", blockIdx.x, threadIdx.x, warp_id, lane_id, lane_id,
//                            cta->smem_storage.gather_delta_values[warp_id * B40C_WARP_THREADS_BFS(KernelPolicy::CUDA_ARCH) + lane_id]);
                      typename Program::GatherType final_dist = WarpReduce(cta->smem_storage.gather_delta_values, warp_id);
//                      printf("Gather_WARP522: bidx=%d, tidx=%d, reduce=%d\n", blockIdx.x, threadIdx.x, final_dist);
                      // vie for write into delta_values
//                        if (threadIdx.x == 0) //correct? lane_id == 0??
                      if (lane_id == 0)
                      {
                    	  typename Program::gather_vertex gather_vertex_functor;
					      gather_vertex_functor(row_id, final_dist, cta->vertex_list);
//                        util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(final_dist, cta->d_gather_results + row_id);
//                        EValue current_dist, dist;
//                        //try to set the d_visit_flag in global memory
////                          if (atomicCAS(cta->d_visit_flags + tile->vertex_id[LOAD][VEC], 0, 1) == 0)
////                        if (atomicCAS(cta->d_visit_flags + row_id, 0, 1) == 0)
//                        {
//
//                          //accumulate to node values
////                            util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(current_dist, cta->d_dists + tile->vertex_id[LOAD][VEC]);
//                          util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(current_dist, cta->d_dists + row_id);
//
//                          dist = min(current_dist, final_dist);
//                          if (row_id == 1) printf("bidx=%d, tidx=%d, row_id=%d, current_dist=%d, final_dist=%d\n", blockIdx.x, threadIdx.x, row_id, current_dist, final_dist);
//                          if (current_dist != dist)
//                          {
////                              util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(dist, cta->d_dists + tile->vertex_id[LOAD][VEC]);
////                              util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(1, cta->d_changed + tile->vertex_id[LOAD][VEC]);
//                            util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(dist, cta->d_dists + row_id);
////                            util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(3, cta->d_changed + row_id);
//                            atomicExch(cta->d_changed + row_id, 3);
//                          }
//                          else
//                          {
////                            util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(0, cta->d_changed + tile->vertex_id[LOAD][VEC]);
////                            util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(2, cta->d_changed + row_id);
//                            atomicExch(cta->d_changed + row_id, 2);
//                          }
//                            if(tile->vertex_id[LOAD][VEC] == 8191)

                      }
//                        if (row_id == 1049)
//                          printf("Gather_WARP490: bidx=%d, tidx=%d, vid=%d, current_dist=%d, dist=%d, final_dist=%d\n", blockIdx.x, threadIdx.x, row_id, current_dist, dist, final_dist);
                    }

                  }

                  // Next vector element
                  Iterate<LOAD, VEC + 1>::ExpandByWarp(cta, tile);
                }

                /**
                 * Expand by scan
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
                {
                  // Attempt to make further progress on this dequeued item's neighbor
                  // list if its current offset into local scratch is in range
//                  SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;
                  VertexId neighbor_id;
                  EValue new_dist;

                  typename Program::GatherType dist = Program::INIT_VALUE;

//                  printf("Gather_SCAN586: bidx=%d, tidx=%d, vidx=%d, r_progress=%d, r_lenghth=%d\n", blockIdx.x, threadIdx.x, tile->vertex_id[LOAD][VEC], tile->row_progress[LOAD][VEC],  tile->row_length[LOAD][VEC]);
                  tile->row_progress[LOAD][VEC] = 0;

                  if (tile->row_length[LOAD][VEC] > 0)
                  {

                    while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) /*&& (scratch_offset < SmemStorage::GATHER_ELEMENTS)*/)
                    {

//                      // Put gather offset into scratch space
////                      cta->smem_storage.gather_offsets[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];
//
////                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, cta->d_column_indices + cta->smem_storage.gather_offsets[scratch_offset]);
                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, cta->d_column_indices + tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC]);

                      typename Program::gather_edge gather_edge_functor;
					  gather_edge_functor(tile->vertex_id[LOAD][VEC], neighbor_id, cta->vertex_list, new_dist);

					  typename Program::gather_sum gather_sum_functor;
					  dist = gather_sum_functor(dist, new_dist);

                      ////                      int num_out_edge;
////                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(num_out_edge, cta->d_num_out_edges + neighbor_id);
//                      // Put dequeued vertex as the predecessor into scratch space
//                      //cta->smem_storage.gather_predecessors[scratch_offset] = tile->vertex_id[LOAD][VEC];
//                      util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(nb_dist, cta->d_dists + neighbor_id);
//                      new_dist = nb_dist+1;
//                      dist = min(new_dist, dist);

//                      printf("Gather_SCAN603: bidx=%d, tidx=%d, vid=%d, neighbor_id=%d, nb_dist=%d, dist=%d\n", blockIdx.x, threadIdx.x, tile->vertex_id[LOAD][VEC], neighbor_id, nb_dist, dist);

                      tile->row_progress[LOAD][VEC]++;
//                      scratch_offset++;
                    }
                    typename Program::gather_vertex gather_vertex_functor;
				    gather_vertex_functor(tile->vertex_id[LOAD][VEC], dist, cta->vertex_list);

//                    util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(dist, cta->d_gather_results + tile->vertex_id[LOAD][VEC]);

                  }
                  //otherwise it's a repeating visit to old vertex, do nothing

                  // Next vector element
                  Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
                }
              };

              /**
               * Iterate next load
               */
              template<int LOAD, int dummy>
              struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
              {
                /**
                 * Init
                 */
                template<typename Tile>
                static __device__ __forceinline__ void Init(Tile *tile)
                {
                  Iterate<LOAD + 1, 0>::Init(tile);
                }

                /**
                 * Inspect
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
                {
                  Iterate<LOAD + 1, 0>::Inspect(cta, tile);
                }

                /**
                 * Expand by CTA
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
                {
                  Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
                }

                /**
                 * Expand by warp
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
                {
                  Iterate<LOAD + 1, 0>::ExpandByWarp(cta, tile);
                }

                /**
                 * Expand by scan
                 */
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
                {
                  Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
                }
              };

              /**
               * Terminate
               */
              template<int dummy>
              struct Iterate<LOADS_PER_TILE, 0, dummy>
              {
                // Init
                template<typename Tile>
                static __device__ __forceinline__ void Init(Tile *tile)
                {
                }

                // Inspect
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
                {
                }

                // ExpandByCta
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
                {
                }

                // ExpandByWarp
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
                {
                }

                // ExpandByScan
                template<typename Cta, typename Tile>
                static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
                {
                }
              };

              //---------------------------------------------------------------------
              // Interface
              //---------------------------------------------------------------------

              /**
               * Constructor
               */
              __device__ __forceinline__
              Tile()
              {
                Iterate<0, 0>::Init(this);
              }

              /**
               * Inspect dequeued vertices, updating label if necessary and
               * obtaining edge-list details
               */
              template<typename Cta>
              __device__ __forceinline__ void Inspect(Cta *cta)
              {
                Iterate<0, 0>::Inspect(cta, this);
              }

              /**
               * Expands neighbor lists for valid vertices at CTA-expansion granularity
               */
              template<typename Cta>
              __device__ __forceinline__ void ExpandByCta(Cta *cta)
              {
                Iterate<0, 0>::ExpandByCta(cta, this);
              }

              /**
               * Expands neighbor lists for valid vertices a warp-expansion granularity
               */
              template<typename Cta>
              __device__ __forceinline__ void ExpandByWarp(Cta *cta)
              {
                Iterate<0, 0>::ExpandByWarp(cta, this);
              }

              /**
               * Expands neighbor lists by local scan rank
               */
              template<typename Cta>
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
            __device__ __forceinline__
            Cta(VertexId queue_index, int num_gpus, SmemStorage &smem_storage, VertexId *d_in, VertexId *d_out, VertexId *d_column_indices, SizeT *d_row_offsets,
            		VertexType vertex_list, util::CtaWorkProgress &work_progress, SizeT max_edge_frontier) :

                queue_index(queue_index), num_gpus(num_gpus), smem_storage(smem_storage), raking_soa_details(
                    typename RakingSoaDetails::GridStorageSoa(smem_storage.coarse_raking_elements, smem_storage.fine_raking_elements),
                    typename RakingSoaDetails::WarpscanSoa(smem_storage.state.coarse_warpscan, smem_storage.state.fine_warpscan), TileTuple(0, 0)), d_in(d_in), d_out(d_out), d_column_indices(
                    d_column_indices), d_row_offsets(d_row_offsets),  vertex_list(vertex_list), work_progress(
                    work_progress), max_edge_frontier(max_edge_frontier)
            {
              if (threadIdx.x == 0)
              {
                smem_storage.state.cta_comm = KernelPolicy::THREADS; // invalid
                smem_storage.state.overflowed = false; // valid
              }
            }

            /**
             * Process a single tile
             */
            __device__ __forceinline__ void ProcessTile(SizeT cta_offset, SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
            {
              Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

              // Load tile
              util::io::LoadTile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE, KernelPolicy::THREADS, KernelPolicy::QUEUE_READ_MODIFIER, false>::LoadValid(tile.vertex_id, d_in,
                  cta_offset, guarded_elements, (VertexId) -1);

              // Inspect dequeued vertices, updating label and obtaining
              // edge-list details
              tile.Inspect(this);

              // Scan tile with carry update in raking threads
              SoaScanOp scan_op;
              TileTuple totals;
              util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(totals, raking_soa_details, RankSoa(tile.coarse_row_rank, tile.fine_row_rank), scan_op);

              SizeT coarse_count = totals.t0;
              tile.fine_count = totals.t1;
//              if (threadIdx.x == 0)printf("Gather_processtile884: bidx=%d, tidx=%d, coarse_count=%d, fine_count=%d\n", blockIdx.x, threadIdx.x, coarse_count, tile.fine_count);

              // Use a single atomic add to reserve room in the queue
              if (threadIdx.x == 0)
              {

//                SizeT enqueue_amt = coarse_count + tile.fine_count;
////                SizeT enqueue_offset = work_progress.Enqueue(enqueue_amt, queue_index + 1);
//                SizeT enqueue_offset = work_progress.Enqueue(enqueue_amt, 0);
//
//                smem_storage.state.coarse_enqueue_offset = enqueue_offset;
//                smem_storage.state.fine_enqueue_offset = enqueue_offset + coarse_count;
//
//                // Check for queue overflow due to redundant expansion
//                if (enqueue_offset + enqueue_amt >= max_edge_frontier)
//                {
//                  smem_storage.state.overflowed = true;
//                  work_progress.SetOverflow<SizeT>();
//                }
              }

              // Protect overflowed flag
              __syncthreads();

              // Quit if overflow
              if (smem_storage.state.overflowed)
              {
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

//              printf("ProcessTile875: bidx=%d, tidx=%d, vid[0][0]=%d, tile.row_progress[0][0]=%d\n", blockIdx.x, threadIdx.x, tile.vertex_id[0][0], tile.row_progress[0][0]);

              tile.progress = 0;
//              while (tile.progress < tile.fine_count)
              {
                // Fill the scratch space with gather-offsets for neighbor-lists.
                tile.ExpandByScan(this);

//                __syncthreads();
//
//                // Copy scratch space into queue
//                int scratch_remainder = B40C_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);
//
//                for (int scratch_offset = threadIdx.x; scratch_offset < scratch_remainder; scratch_offset += KernelPolicy::THREADS)
//                {
//                  // Gather a neighbor
//                  VertexId neighbor_id;
//                  util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(neighbor_id, d_column_indices + smem_storage.gather_offsets[scratch_offset]);
//
//                  // Scatter it into queue
//                  util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(neighbor_id, d_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
//
//                  // Scatter predecessor it into queue
//                  VertexId predecessor_id = smem_storage.gather_predecessors[scratch_offset];
//                  //VertexId predecessor_id = 1;
//                  util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(predecessor_id, d_predecessor_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
//                }

//                tile.progress += SmemStorage::GATHER_ELEMENTS;

//                __syncthreads();
              }
            }
          }
          ;

        } // namespace expand_atomic
      } // namespace vertex_centric
    } // namespace bc
  } // namespace graph
} // namespace b40c

