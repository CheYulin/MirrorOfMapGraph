/**
Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This work was (partially) funded by the DARPA XDATA program under
AFRL Contract #FA8750-13-C-0002.

This material is based upon work supported by the Defense Advanced
Research Projects Agency (DARPA) under Contract No. D14PC00029.
 */
/******************************************************************************

Portions of this file are:

Copyright 2013 Royal Caliber LLC. (http://www.royal-caliber.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************/
 
#pragma once

#include <vector>
#include <iterator>
#include <moderngpu.cuh>
//#include <util.h>
#include <util/mgpucontext.h>
#include <mgpuenums.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <time.h>
#include <device/ctaloadbalance.cuh>
#include <device/loadstore.cuh>
using namespace b40c;
using namespace graph;

namespace GASengine
{
  namespace vertex_centric
  {
    namespace mgpukernel
    {
      template<typename Tuning, int NT, int VT, typename Program>
      __global__ void kernel_scatter_mgpu(int frontier_selector, int move_count,
          int num_active, int* d_edge_frontier_size,
          typename Program::SizeT* offsets,
          typename Program::VertexId* active_vertices,
          typename Program::VertexId* edge_count_scan,
          typename Program::VertexId* indices,
          const int* mp_global,
          typename Program::VertexId* edge_frontier,
          typename Program::VertexType vertex_list,
          typename Program::EdgeType edge_list,
          typename Program::VertexId* d_edgeCSC_indices,
          typename Program::DataType* misc_values)
      {
        __shared__ int indices_shared[NT * (VT + 1)];
        int tid = threadIdx.x;
        int block = blockIdx.x;

        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
//          for(int i=0; i<num_active; i++)
//          {
//            printf("%d\n", vertex_list.d_dists[i]);
//          }
          d_edge_frontier_size[frontier_selector ^ 1] = 0;
//          d_edge_frontier_size[frontier_selector] += move_count;
        }

        // Load balance the move IDs (counting_iterator) over the scan of the
        // interval sizes.
        int4 range = mgpu::CTALoadBalance<NT, VT>(move_count, edge_count_scan,
            num_active, block, tid, mp_global, indices_shared, true);

        // The interval indices are in the left part of shared memory (moveCount).
        // The scan of interval counts are in the right part (intervalCount).
        move_count = range.y - range.x;
        num_active = range.w - range.z;
        int* move_shared = indices_shared;
        int* intervals_shared = indices_shared + move_count;
        int* intervals_shared2 = intervals_shared - range.z;

//        if (blockIdx.x == 0 && threadIdx.x == 0)
//        {
//          for (int i = 0; i < move_count + num_active; i++)
//          {
//            printf("indices_shared[%d]=%d\n", i, indices_shared[i]);
//
//          }
//        }

        // Read out the interval indices and scan offsets.
        int interval[VT], rank[VT];
#pragma unroll
        for (int i = 0; i < VT; ++i)
        {
          int index = NT * i + tid;
          int gid = range.x + index;
          interval[i] = range.z; //initialize interval to range.z
          if (index < move_count)
          {
            interval[i] = move_shared[index];
            rank[i] = gid - intervals_shared2[interval[i]];
          }
        }
        __syncthreads();

        // Load and distribute the gather and scatter indices.
        int gather[VT];

        // Load the gather pointers into intervals_shared.
//        mgpu::DeviceMemToMemLoop < NT > (num_active, offsets + active_vertices[range.z], tid,
//            intervals_shared);

        for (int i = tid; i < num_active; i += blockDim.x)
        {
          intervals_shared[i] = offsets[active_vertices[range.z + i]];
        }
        __syncthreads();

        int iActive[VT];
        mgpu::DeviceSharedToReg<NT, VT>(move_shared, tid, iActive);
//        if (blockIdx.x == 0 && threadIdx.x == 0)
//        {
//          printf("move_count=%d\n", move_count);
//          for (int i = 0; i < VT; i++)
//          {
//            printf("iActive[%d]=%d\n", i, iActive[i]);
//
//          }
//        }
        typename Program::MiscType local_misc_values[VT];
        typename Program::expand_edge expand_edge_functor;

        // Make a second pass through shared memory. Grab the start indices of
        // the interval for each item and add the scan into it for the gather
        // index.
#pragma unroll
        for (int i = 0; i < VT; ++i)
          gather[i] = intervals_shared2[interval[i]] + rank[i];
        __syncthreads();

        // Gather the data into register.
        typename Program::VertexId data[VT];

        mgpu::DeviceGather<NT, VT>(move_count, indices, gather, tid, data, false);
#pragma unroll
        for (int i = 0; i < VT; ++i)
        {
          int index = NT * i + tid;
          if (index < move_count)
          {
            typename Program::VertexId vertex_id = active_vertices[iActive[i]];
            typename Program::VertexId edge_id = gather[i];
            typename Program::VertexId neighbor = indices[edge_id];
            int read_edge_id;
            if (d_edgeCSC_indices)
              read_edge_id = d_edgeCSC_indices[edge_id];
            else
              read_edge_id = edge_id;
//            printf("blockIdx.x=%d, threadIdx.x=%d, vertex_id=%d, edge_id=%d, neighbor=%d\n", blockIdx.x, threadIdx.x, vertex_id, edge_id, neighbor);
            expand_edge_functor(true, 0, vertex_id, neighbor, read_edge_id,
                vertex_list, edge_list, data[i], local_misc_values[i]);
          }
        }

        mgpu::DeviceRegToGlobal<NT, VT>(move_count, data, tid,
            edge_frontier + range.x);

        mgpu::DeviceRegToGlobal<NT, VT>(move_count, local_misc_values, tid,
            misc_values + range.x);
      }

      template<typename Program>
      __global__ void apply(long long iteration, int frontier_size,
          typename Program::VertexId* frontier,
          typename Program::GatherType *gather_values,
          typename Program::VertexType vertex_list,
          typename Program::EdgeType edge_list, char* changed) // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
      {
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;

        typename Program::apply apply_functor;

        for (int i = tidx; i < frontier_size; i += gridDim.x * blockDim.x)
        {
          int v = frontier[i];
          typename Program::GatherType gather_value = gather_values[v];
          apply_functor(v, iteration, gather_value, vertex_list, edge_list,
              changed[v]);
        }
      }

      template<typename Program, int VT, int NT>
      __global__ void kernel_gather_mgpu(
          typename Program::SizeT num_active,
          const typename Program::VertexId *active_vertices,
          const typename Program::SizeT numBlocks,
          typename Program::SizeT move_count,
          const typename Program::SizeT *edge_count_scan,
          const typename Program::VertexId *mp_global,
          const typename Program::SizeT *offsets,
          const typename Program::VertexId *indices,
          typename Program::VertexType vertex_list,
          typename Program::EdgeType edge_list,
          typename Program::VertexId* d_edgeCSC_indices,
          typename Program::VertexId *pred_vertices,
          typename Program::GatherType* gather_results)
      {
        __shared__ int indices_shared[NT * (VT + 1)];
        int tid = threadIdx.x;
        int block = blockIdx.x;
//
//        // Load balance the move IDs (counting_iterator) over the scan of the
//        // interval sizes.
        int4 range = mgpu::CTALoadBalance<NT, VT>(move_count, edge_count_scan,
            num_active, block, tid, mp_global, indices_shared, true);

        // The interval indices are in the left part of shared memory (moveCount).
        // The scan of interval counts are in the right part (intervalCount).
        move_count = range.y - range.x;
        num_active = range.w - range.z;

        int* move_shared = indices_shared;
        int* intervals_shared = indices_shared + move_count;
        int* intervals_shared2 = intervals_shared - range.z;

        //        if (blockIdx.x == 0 && threadIdx.x == 0)
        //        {
        //          for (int i = 0; i < move_count + num_active; i++)
        //          {
        //            printf("indices_shared[%d]=%d\n", i, indices_shared[i]);
        //
        //          }
        //        }

        // Read out the interval indices and scan offsets.
        int interval[VT], rank[VT];
#pragma unroll
        for (int i = 0; i < VT; ++i)
        {
          int index = NT * i + tid;
          int gid = range.x + index;
          interval[i] = range.z; //initialize interval to range.z
          if (index < move_count)
          {
            interval[i] = move_shared[index];
            rank[i] = gid - intervals_shared2[interval[i]];
          }
        }
        __syncthreads();

        // Load and distribute the gather and scatter indices.
        int gather[VT];

        for (int i = tid; i < num_active; i += blockDim.x)
        {
          intervals_shared[i] = offsets[active_vertices[range.z + i]];
        }
        __syncthreads();

        int iActive[VT];
        mgpu::DeviceSharedToReg<NT, VT>(move_shared, tid, iActive);
        //        if (blockIdx.x == 0 && threadIdx.x == 0)
        //        {
        //          printf("move_count=%d\n", move_count);
        //          for (int i = 0; i < VT; i++)
        //          {
        //            printf("iActive[%d]=%d\n", i, iActive[i]);
        //
        //          }
        //        }

        typename Program::gather_edge gather_edge_functor;
        typename Program::GatherType result[VT];
        typename Program::VertexId predecessors[VT];

        // Make a second pass through shared memory. Grab the start indices of
        // the interval for each item and add the scan into it for the gather
        // index.
#pragma unroll
        for (int i = 0; i < VT; ++i)
        {
          gather[i] = intervals_shared2[interval[i]] + rank[i];
          predecessors[i] = active_vertices[iActive[i]];
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < VT; ++i)
        {
          int index = NT * i + tid;
          if (index < move_count)
          {
            typename Program::VertexId edge_id = gather[i];
            typename Program::VertexId neighbor = indices[edge_id];
            typename Program::VertexId vertex_id = predecessors[i];

            if (offsets[vertex_id + 1] - offsets[vertex_id] > 0)
            {
              if (d_edgeCSC_indices)
                gather_edge_functor(vertex_id, d_edgeCSC_indices[edge_id],
                    neighbor, vertex_list, edge_list, result[i]);
              else
                gather_edge_functor(vertex_id, edge_id, neighbor,
                    vertex_list, edge_list, result[i]);
            }
            else
              result[i] = Program::INIT_VALUE;
          }
        }

        mgpu::DeviceRegToGlobal<NT, VT>(move_count, result, tid,
            gather_results + range.x);

        mgpu::DeviceRegToGlobal<NT, VT>(move_count, predecessors, tid,
            pred_vertices + range.x);
      }

      template<typename KernelPolicy, typename Program>
      __global__ void reset_gather_result(int iteration, int num_elements,
          int* frontier, typename Program::VertexType vertex_list,
          typename Program::EdgeType edge_list,
          typename Program::GatherType* gather_tmp,
          typename KernelPolicy::VisitedMask* d_visited_mask) // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
      {

        //        if(blockIdx.x==0 && threadIdx.x==0)
        //          printf("reset_gather: num_elements=%d\n", num_elements);
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        typename Program::post_apply post_apply_functor;

        for (int i = tidx; i < num_elements; i += gridDim.x * blockDim.x)
        {
          int v = frontier[i];
          typename Program::SizeT mask_byte_offset = (v
              & KernelPolicy::VERTEX_ID_MASK) >> 3;
          d_visited_mask[mask_byte_offset] = 0;
//          printf("reset: v=%d, mask_byte_offset=%d\n", v, mask_byte_offset);
          //              d_gather_results[v] = 100000000;
          //              d_dists[v] = d_dists_out[v];
          post_apply_functor(v, vertex_list, edge_list, gather_tmp);
        }
      }

      template<typename KernelPolicy, typename Program>
      __global__ void reset_gather_result(int iteration, int nodes,
          typename Program::VertexType vertex_list,
          typename Program::EdgeType edge_list,
          typename Program::GatherType* gather_tmp,
          typename KernelPolicy::VisitedMask* d_visited_mask) // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
      {
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        typename Program::post_apply post_apply_functor;

        for (int v = tidx; v < nodes; v += gridDim.x * blockDim.x)
        {
          typename Program::SizeT mask_byte_offset = (v
              & KernelPolicy::VERTEX_ID_MASK) >> 3;
          d_visited_mask[mask_byte_offset] = 0;
          post_apply_functor(v, vertex_list, edge_list, gather_tmp);
        }
      }

      template<typename InputIt, typename PredicateIt, typename OutputIt>
      __global__
      void kernel_copy_if(InputIt in, int N, PredicateIt pred, int *d_map,
          OutputIt output)
      {
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        for (int i = tid; i < N; i += blockDim.x * gridDim.x)
        {
          if (pred[i])
          {
            output[d_map[i]] = in[i];
          }
        }
      }

    }      //namespace mgpukernel
  }      // namespace vertex_centric
} // namespace GASengine

