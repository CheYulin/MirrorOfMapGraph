/* 
 * File:   predextract.h
 * Author: zhisong
 *
 * Created on October 16, 2014, 1:46 PM
 */

#ifndef PREDEXTRACT_H
#define	PREDEXTRACT_H

#include <GASengine/problem_type.cuh>
#include <GASengine/csr_problem.cuh>
//#include <b40c/graph/csr_graph.cuh>
#include <GASengine/vertex_centric/mgpukernel/kernel.cuh>
//#include <GASengine/enactor_vertex_centric.cuh>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <vector>
#include <iterator>
#include <moderngpu.cuh>
//#include <util.h>
#include <util/mgpucontext.h>
#include <mgpuenums.h>

#include "bfs.h"

using namespace GASengine;
using namespace std;

struct predextract
{
  typedef int DataType;
  typedef DataType MiscType;
  typedef DataType GatherType;
  typedef int VertexId;
  typedef int SizeT;
  
//  typedef bfs::VertexType VertexType;
//  typedef bfs::EdgeType EdgeType;

  static const DataType INIT_VALUE = -1;
  static const bool allow_duplicates = true;

  struct VertexType
  {
    int* d_labels;
    int nodes;
    int edges;

    VertexType() :
        d_labels(NULL), nodes(0), edges(0)
    {
    }
  };

  struct EdgeType
  {
    int nodes; // #of nodes.
    int edges; // #of edges.

    EdgeType() :
        nodes(0), edges(0)
    {
    }
  };

  static void Initialize(const int directed, const int nodes, const int edges, int num_srcs,
      int* srcs, int* d_row_offsets, int* d_column_indices, int* d_column_offsets, int* d_row_indices, int* d_edge_values,
      VertexType &vertex_list, EdgeType &edge_list, int* d_frontier_keys[3],
      MiscType* d_frontier_values[3])
  {
 
  }

  static SrcVertex srcVertex()
  {
    return SINGLE;
  }

  static GatherEdges gatherOverEdges()
  {
    return NO_GATHER_EDGES;
  }

  static ApplyVertices applyOverEdges()
  {
    return NO_APPLY_VERTICES;
  }

  static ExpandEdges expandOverEdges()
  {
    return EXPAND_OUT_EDGES;
  }

  static PostApplyVertices postApplyOverEdges()
  {
    return NO_POST_APPLY_VERTICES;
  }

  /**
   * the binary operator
   */
  struct gather_sum
  {
    __device__
    GatherType operator()(GatherType left, GatherType right)
    {
      return left + right;
    }
  };

  /**
   * For each vertex in the frontier,
   */
  struct gather_vertex
  {
    __device__
    void operator()(const int vertex_id, const GatherType final_value,
        VertexType &vertex_list, EdgeType &edge_list)
    {

    }
  };

//  struct gather_edge
//  {
//    __device__
//    void operator()(const int vertex_id, const int edge_id, const int neighbor_id_in,
//        VertexType &vertex_list, EdgeType &edge_list, GatherType& new_value)
//    {
//
//    }
//  };
  
  struct gather_edge
  {

    __device__
            void operator()(const int vertex_id, const int edge_id, const int neighbor_id_in,
                            VertexType &vertex_list, EdgeType &edge_list, GatherType & new_value)
    {
//      printf("tid=%d, vertex_id=%d, neighbor_id_in=%d, nb_label=%d, my_label=%d, new_value=%d\n", threadIdx.x, vertex_id, neighbor_id_in, nb_label, my_label, new_value);
      int nb_label = vertex_list.d_labels[neighbor_id_in];
      int my_label = vertex_list.d_labels[vertex_id];
      new_value = (my_label - nb_label) == 1? neighbor_id_in: -1;
//      printf("tid=%d, vertex_id=%d, neighbor_id_in=%d, nb_label=%d, my_label=%d, new_value=%d\n", threadIdx.x, vertex_id, neighbor_id_in, nb_label, my_label, new_value);
    }
  };

  struct apply
  {
    __device__
    void operator()(const int vertex_id, const int iteration, GatherType gathervalue,
        VertexType& vertex_list, EdgeType& edge_list, char& changed)
    {
    }
  };

  /** post-apply function (invoked after threads in apply() synchronize at a memory barrier). */
  struct post_apply
  {
    __device__
    void operator()(const int vertex_id, VertexType& vertex_list, EdgeType& edge_list, GatherType* gather_tmp)
    {
    }
  };

  struct expand_vertex
  {
    __device__
    bool operator()(const int vertex_id, const char changed, VertexType &vertex_list, EdgeType& edge_list)
    {
      return true;
    }
  };

  struct expand_edge
  {
    __device__
    void operator()(const bool changed, const int iteration,
        const int vertex_id, const int neighbor_id_in, const int edge_id,
        VertexType& vertex_list, EdgeType& edge_list, int& frontier, int& misc_value)
    {
    }
  };

  struct contract
  {
    __device__
    void operator()(const int iteration, int &vertex_id,
        VertexType &vertex_list, EdgeType &edge_list, GatherType* gather_tmp, int& misc_value)
    {
      
    }
  };

  static void extractResult(VertexType& vertex_list, DataType* h_output)
  {
  }

};

struct ReduceFunctor : std::binary_function<int, int, int>
{

  __device__ int operator()(const int &left,
                            const int & right)
  {
    return max(left, right);
  }
};

struct EdgeCountIterator : public std::iterator<std::input_iterator_tag, int>
{
  int *m_offsets;
  int *m_active;

  __host__ __device__ EdgeCountIterator(int *offsets, int *active) :
  m_offsets(offsets), m_active(active)
  {
  }
  ;

  __device__
          int operator[](int i)const
  {
    int active = m_active[i];
    return max(m_offsets[active + 1] - m_offsets[active], 1);
  }

  __device__ EdgeCountIterator operator +(int i)const
  {
    return EdgeCountIterator(m_offsets, m_active + i);
  }
};

//void predextract(GASengine::CsrProblem<Program, VertexId, SizeT, Value, g_mark_predecessor, g_with_value> &csr_problem, int device_id, int* m_gatherTmp)
void pred_extract(int nodes, int* d_column_offsets, int* d_row_indices, int* d_labels, int device_id, int* m_gatherTmp)
{
  
//  typedef GASengine::CsrProblem<bfs, VertexId, SizeT, Value, g_mark_predecessor, g_with_value> CsrProblem;
//  typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

  mgpu::ContextPtr m_mgpuContext;
  m_mgpuContext = mgpu::CreateCudaDevice(device_id);
  int* d_seq;
  cudaMalloc((void**)&d_seq, nodes * sizeof (int));
  thrust::device_ptr<int> d_seq_ptr(d_seq);
  thrust::sequence(thrust::device, d_seq_ptr, d_seq_ptr + nodes);
  
  
//  int* test_vid = new int[graph_slice->nodes];
//  cudaMemcpy(test_vid, d_seq, (graph_slice->nodes) * sizeof (int), cudaMemcpyDeviceToHost);
//  printf("d_seq: ");
//  for (int i = 0; i < (graph_slice->nodes); ++i)
//  {
//    printf("%d, ", test_vid[i]);
//  }
//  printf("\n");
//  delete[] test_vid;
  

  int n_active_edges;
  int* d_edgeCountScan;
  cudaMalloc((void**) &d_edgeCountScan, (nodes + 1) * sizeof(int));
  EdgeCountIterator ecIterator(d_column_offsets, d_seq);
  mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
                                                                                   ecIterator,
                                                                                   nodes,
                                                                                   0,
                                                                                   mgpu::plus<int>(),
                                                                                   (int*)NULL,
                                                                                   &n_active_edges,
                                                                                   d_edgeCountScan,
                                                                                   *m_mgpuContext);

  //  int n_active_edges;
  //  cudaMemcpy(&n_active_edges, m_deviceMappedValue,
  //             sizeof (int),
  //             cudaMemcpyDeviceToHost);

  //          printf("n_active_edges = %d, frontier_size = %d\n", n_active_edges, frontier_size);

  const int nThreadsPerBlock = 128;
  MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
          (mgpu::counting_iterator<int>(0), n_active_edges, d_edgeCountScan, nodes,
           nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);

  int nBlocks = (n_active_edges + nodes + nThreadsPerBlock - 1) / nThreadsPerBlock;

  int* m_gatherDstsTmp;
  int* m_gatherMapTmp;
//  int* m_gatherTmp;
  cudaMalloc((void**)&m_gatherDstsTmp, (n_active_edges) * sizeof (int));
  cudaMalloc((void**)&m_gatherMapTmp, (n_active_edges) * sizeof (int));
//  cudaMalloc((void**)&m_gatherTmp, graph_slice->nodes * sizeof (int));
  const int VT = 1;

  predextract::VertexType vertex_list;
  predextract::EdgeType edge_list;
  vertex_list.d_labels = d_labels;
  vertex_centric::mgpukernel::kernel_gather_mgpu<predextract, VT, nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(nodes,
                                                                                                                   d_seq,
                                                                                                                   nBlocks,
                                                                                                                   n_active_edges,
                                                                                                                   d_edgeCountScan,
                                                                                                                   partitions->get(),
                                                                                                                   d_column_offsets,
                                                                                                                   d_row_indices,
                                                                                                                   vertex_list,
                                                                                                                   edge_list,
                                                                                                                   (int*)NULL,
                                                                                                                   m_gatherDstsTmp,
                                                                                                                   m_gatherMapTmp);
  
  

//  int* test_vid = new int[n_active_edges];
//  cudaMemcpy(test_vid, m_gatherDstsTmp, n_active_edges * sizeof (int), cudaMemcpyDeviceToHost);
//  printf("m_gatherDstsTmp: ");
//  for (int i = 0; i < n_active_edges; ++i)
//  {
//    printf("%d, ", test_vid[i]);
//  }
//  printf("\n");
//  delete[] test_vid;
//  
//  test_vid = new int[n_active_edges];
//  cudaMemcpy(test_vid, m_gatherMapTmp, n_active_edges * sizeof (int), cudaMemcpyDeviceToHost);
//  printf("m_gatherMapTmp: ");
//  for (int i = 0; i < n_active_edges; ++i)
//  {
//    printf("%d, ", test_vid[i]);
//  }
//  printf("\n");
//  delete[] test_vid;

  mgpu::ReduceByKey(
                    m_gatherDstsTmp,
                    m_gatherMapTmp,
                    n_active_edges,
                    predextract::INIT_VALUE,
                    ReduceFunctor(),
                    mgpu::equal_to<int > (),
                    (int *)NULL,
                    m_gatherTmp,
                    //                        graph_slice->m_gatherTmp,
                    NULL,
                    NULL,
                    *m_mgpuContext);
  
//  test_vid = new int[nodes];
//  cudaMemcpy(test_vid, m_gatherTmp, (nodes) * sizeof (int), cudaMemcpyDeviceToHost);
//  printf("m_gatherTmp: ");
//  for (int i = 0; i < (nodes); ++i)
//  {
//    printf("%d, ", test_vid[i]);
//  }
//  printf("\n");
//  delete[] test_vid;
  
  cudaFree(d_seq);
  cudaFree(d_edgeCountScan);
}

#endif	/* PREDEXTRACT_H */

