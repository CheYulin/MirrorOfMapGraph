/* 
 * File:   adaptiveBC.h
 * Author: Zhisong Fu
 *
 * Created on July 9, 2013, 9:49 AM
 */

#ifndef ADAPTIVEBC_H
#define	ADAPTIVEBC_H

template<int HOST>
__host__ __device__ int
new_atomicAdd(int *addr, int val)
{
  if ( HOST )
  {
    int old = addr[0];
    addr[0] = old + val ;
    return old ;
  }
  else
  {
#ifdef __CUDA_ARCH__
#if  __CUDA_ARCH__ >= 110
    return atomicAdd( addr, val ) ;
#else
    printf("AtomicAdd does not support sm1!\n");
#endif
#else
    return 0 ;
#endif
  }
}

struct adaptiveBC {
  struct VertexType {
    int dist;
    int sigma;
    bool changed;
    double BC;

    //the starting value should technically be infinity, but we want to avoid overflow when we add to it
    //we could handle this by doing the addition as a long and then clamping or something to that effect
    //this approach is fine for now
    __host__ __device__
    VertexType() : dist(10000000), changed(false), sigma(0), BC(0.0){}

    __host__ __device__
    VertexType(int dist_, int s_, bool changed_, double BC_) : dist(dist_), sigma(s_), changed(changed_), BC(BC_) {}
  };

  static GatherEdges gatherOverEdges() {
    return GATHER_IN_EDGES;
  }

  struct gather {
    __host__ __device__
      int operator()(const VertexType dst, const VertexType src, const int edge_length, const int flag) {
        return src.dist + edge_length;
      }
  };

  struct sum {
    __host__ __device__
      int operator()(int left, int right) {
        int r = min(left, right);
        return r;
      }
  };

  struct apply {
    __host__ __device__
      void operator()(VertexType &curVertex, const int newDistance = 0) {
        int newVertexVal = min(newDistance, curVertex.dist);
        if (newDistance == curVertex.dist)
          curVertex.changed = false;
        else
          curVertex.changed = true;

        curVertex.dist = newVertexVal;
      }
  };

  static ScatterEdges scatterOverEdges() {
    return SCATTER_OUT_EDGES;
  }

  struct scatter {
    __host__ __device__
      int operator()(VertexType &dst, VertexType &src, int &e) {
         
        if(src.dist == dst.dist-1)
        {
          new_atomicAdd<0>(&dst.sigma, src.sigma);
        }
       
        return src.changed;
      }
  };

};

struct adaptiveBC_backward {
  typedef typename adaptiveBC::VertexType VertexType;

  static GatherEdges gatherOverEdges() {
    return GATHER_IN_EDGES;
  }

  struct gather {
    __host__ __device__
      double operator()(const VertexType dst, const VertexType src, const int edge_length, const int flag) {
      if(flag && dst.dist == src.dist-1)
      {
        return (double)dst.sigma / src.sigma * (1.0 + src.BC);
      }
      else 
        return 0.0;
      }
  };

  struct sum {
    __host__ __device__
      double operator()(double left, double right) {
        return left + right;
      }
  };

  struct apply {
    __host__ __device__
      void operator()(VertexType &curVertex, const double delta = 0) {
        double newVertexVal = delta + curVertex.BC;
        curVertex.BC += newVertexVal;
      }
  };

  static ScatterEdges scatterOverEdges() {
    return SCATTER_OUT_EDGES;
  }

  struct scatter {
    __host__ __device__
      int operator()(VertexType &dst, VertexType &src, int &e) {

        return dst.dist == src.dist-1;
      }
  };
};

__global__ void kernel_reinit_active(int nv, adaptiveBC::VertexType* vertex_data, int* flags, int diameter)
{
  int tidx = blockDim.x*blockIdx.x + threadIdx.x;
  for(int v = tidx; v<nv; v+= gridDim.x*blockDim.x)
  {
    if(vertex_data[v].dist == diameter - 2)
    {
      flags[v] = 1;
      vertex_data[v].changed = true;
    }
    else 
    {
      flags[v] = 0;
      vertex_data[v].changed = false;
    }
  }   
}
void reinit_active_flags(thrust::device_vector<adaptiveBC::VertexType>& d_vertex_data, thrust::device_vector<int> &flags, int diameter)
{
  int nthreads = 256;
  int nv = d_vertex_data.size();
  int nblocks = min( (int)65535, (int)ceil( (double)nv/nthreads));
  kernel_reinit_active<<<nblocks, nthreads>>>(nv, thrust::raw_pointer_cast(&d_vertex_data[0]), thrust::raw_pointer_cast(&flags[0]), diameter);
}


__global__ void kernel_accum_BC(int nv, double* bc, adaptiveBC::VertexType* vertex_data, int startVertex)
{
  int tidx = blockDim.x*blockIdx.x + threadIdx.x;
  for(int v = tidx; v<nv; v+= gridDim.x*blockDim.x)
  {
    if(v != startVertex) bc[v] += vertex_data[v].BC;
  }   
}
void accum_BC(thrust::device_vector<double>& d_BC, thrust::device_vector<adaptiveBC::VertexType>& d_vertex_data, int startVertex)
{
  int nthreads = 256;
  int nv = d_vertex_data.size();
  int nblocks = min( (int)65535, (int)ceil( (double)nv/nthreads));
  kernel_accum_BC<<<nblocks, nthreads>>>(nv, thrust::raw_pointer_cast(&d_BC[0]), thrust::raw_pointer_cast(&d_vertex_data[0]), startVertex);
}

#endif	/* ADAPTIVEBC_H */

