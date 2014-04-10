/*
 * kernel.cuh
 *
 *  Created on: Apr 4, 2014
 *      Author: zhisong
 */

#ifndef KERNEL_CUH_
#define KERNEL_CUH_

namespace MPI
{
  namespace mpikernel
  {

    template<typename Program>
    __global__ void frontier2flag(int frontier_size, int nodes, typename Program::VertexId* frontier, char* flags)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < frontier_size; i += gridDim.x * blockDim.x)
      {
        typename Program::VertexId v = frontier[i];
        flags[v] = 1;
      }
    }

    template<typename Program>
    __global__ void flag2bitmap(int nodes, int byte_size, char* flags, char* bitmap)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < byte_size; i += gridDim.x * blockDim.x)
      {
        bitmap[i] = 0;
        for (int j = 0; j < 8; j++)
        {
          int v = i * 8 + j;
          if (v < nodes)
          {
            char f = flags[v];
            if (f == 1)
            {
              int byte_offset = i;
              char mask_byte = 1 << j;

              bitmap[byte_offset] |= mask_byte;
//                printf("v=%d, byte_offset=%d, mask_byte=%d, bitmap[byte_offset]=%d\n", v, byte_offset, mask_byte, bitmap[byte_offset]);
            }
          }
        }

      }
    }

    template<typename Program>
    __global__ void bitmap2flag(int byte_size, char* bitmap, char* flags)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < byte_size; i += gridDim.x * blockDim.x)
      {
        char b = bitmap[i];
        char mask = 1;
        for (int j = 0; j < 8; j++)
        {
          mask <<= j;
          if(b & mask)
          {
            flags[8*i + j] = 1;
          }
          else
            flags[8*i + j] = 0;
        }
      }
    }

    //c = a - b
    __global__ void bitsubstract(int n, char* a, const char* b, char* c)          // Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < n; i += gridDim.x * blockDim.x)
      {
        char tmpa = a[i];
        char tmpb = b[i];
        c[i] = (~tmpb) & tmpa;
      }
    }

    //c = union(a, b)
    __global__ void bitunion(int n, char* a, const char* b, char* c)
    {
      int tidx = blockIdx.x * blockDim.x + threadIdx.x;

      for (int i = tidx; i < n; i += gridDim.x * blockDim.x)
      {
        char tmpa = a[i];
        char tmpb = b[i];
        c[i] = tmpb | tmpa;
      }

    }

  }
}
#endif /* KERNEL_CUH_ */
