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

using namespace GASengine;
using namespace std;

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

template<typename Program, typename VertexId, typename SizeT,
typename Value, bool g_mark_predecessor, // Whether to mark predecessors (vs. mark distance from source)
bool g_with_value>
void predextract(GASengine::CsrProblem<Program, VertexId, SizeT, Value, g_mark_predecessor, g_with_value> &csr_problem, int device_id, int* m_gatherTmp)
{
  //      cudaDeviceSynchronize();
  //      MPI_Barrier(MPI_COMM_WORLD);
  typedef GASengine::CsrProblem<bfs, VertexId, SizeT, Value, g_mark_predecessor, g_with_value> CsrProblem;
  typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];
  ////      if (pi == pj)
  ////      {
  ////        cudaMemcpy(graph_slice->vertex_list.d_labels_src, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof (int), cudaMemcpyDeviceToDevice);
  ////      }
  ////
  ////      MPI_Bcast(graph_slice->vertex_list.d_labels_src, graph_slice->nodes, MPI_INT, pj, w.new_col_comm);

  //  int byte_size = (graph_slice->nodes + 8 - 1) / 8;
  //  ////        MPI_Recv(graph_slice->d_bitmap_in, byte_size, MPI_CHAR, src_proc, tag, MPI_COMM_WORLD, &status);//receive broadcast
  //  int nthreads = 256;
  //  int nblocks = (byte_size + nthreads - 1) / nthreads;
  //  MPI::mpikernel::bitmap2flag<Program> << <nblocks, nthreads >> >(byte_size, graph_slice->d_bitmap_assigned, graph_slice->d_visit_flags);
  //  util::B40CPerror(cudaDeviceSynchronize(), "bitmap2flag", __FILE__, __LINE__);
  //
  //  copy_if_mgpu(graph_slice->nodes,
  //               graph_slice->d_visit_flags,
  //               graph_slice->frontier_queues.d_keys[0],
  //               NULL,
  //               &frontier_size,
  //               m_mgpuContext);



  mgpu::ContextPtr m_mgpuContext;
  m_mgpuContext = mgpu::CreateCudaDevice(device_id);
  int* d_seq;
  cudaMalloc((void**)&d_seq, graph_slice->nodes * sizeof (int));
  thrust::device_ptr<int> d_seq_ptr(d_seq);
  thrust::sequence(thrust::device, d_seq_ptr, d_seq_ptr + graph_slice->nodes);
  
  
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
  EdgeCountIterator ecIterator(graph_slice->d_column_offsets, d_seq);
  mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, int, mgpu::plus<int>, int*>(
                                                                                   ecIterator,
                                                                                   graph_slice->nodes,
                                                                                   0,
                                                                                   mgpu::plus<int>(),
                                                                                   (int*)NULL,
                                                                                   &n_active_edges,
                                                                                   graph_slice->d_edgeCountScan,
                                                                                   *m_mgpuContext);

  //  int n_active_edges;
  //  cudaMemcpy(&n_active_edges, m_deviceMappedValue,
  //             sizeof (int),
  //             cudaMemcpyDeviceToHost);

  //          printf("n_active_edges = %d, frontier_size = %d\n", n_active_edges, frontier_size);

  const int nThreadsPerBlock = 128;
  MGPU_MEM(int)partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper >
          (mgpu::counting_iterator<int>(0), n_active_edges, graph_slice->d_edgeCountScan, graph_slice->nodes,
           nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);

  SizeT nBlocks = (n_active_edges + graph_slice->nodes + nThreadsPerBlock - 1) / nThreadsPerBlock;

  int* m_gatherDstsTmp;
  int* m_gatherMapTmp;
//  int* m_gatherTmp;
  cudaMalloc((void**)&m_gatherDstsTmp, (n_active_edges) * sizeof (int));
  cudaMalloc((void**)&m_gatherMapTmp, (n_active_edges) * sizeof (int));
//  cudaMalloc((void**)&m_gatherTmp, graph_slice->nodes * sizeof (int));
  const int VT = 1;

  vertex_centric::mgpukernel::kernel_gather_mgpu<Program, VT, nThreadsPerBlock> << <nBlocks, nThreadsPerBlock >> >(graph_slice->nodes,
                                                                                                                   d_seq,
                                                                                                                   nBlocks,
                                                                                                                   n_active_edges,
                                                                                                                   graph_slice->d_edgeCountScan,
                                                                                                                   partitions->get(),
                                                                                                                   graph_slice->d_column_offsets,
                                                                                                                   graph_slice->d_row_indices,
                                                                                                                   graph_slice->vertex_list,
                                                                                                                   graph_slice->edge_list,
                                                                                                                   (int*)NULL,
                                                                                                                   m_gatherDstsTmp,
                                                                                                                   m_gatherMapTmp);
  
  

//  test_vid = new int[n_active_edges];
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
                    Program::INIT_VALUE,
                    ReduceFunctor(),
                    mgpu::equal_to<VertexId > (),
                    (VertexId *)NULL,
                    m_gatherTmp,
                    //                        graph_slice->m_gatherTmp,
                    NULL,
                    NULL,
                    *m_mgpuContext);
  
//  test_vid = new int[graph_slice->nodes];
//  cudaMemcpy(test_vid, m_gatherTmp, (graph_slice->nodes) * sizeof (int), cudaMemcpyDeviceToHost);
//  printf("m_gatherTmp: ");
//  for (int i = 0; i < (graph_slice->nodes); ++i)
//  {
//    printf("%d, ", test_vid[i]);
//  }
//  printf("\n");
//  delete[] test_vid;
  //  graph_slice->predecessor_size = graph_slice->nodes;
  //      thrust::device_ptr<int> m_gatherTmp_ptr(graph_slice->m_gatherTmp);
  //      long long pred_sum = thrust::reduce(m_gatherTmp_ptr, m_gatherTmp_ptr + graph_slice->nodes);
  //      printf("rank_id %d pred_sum %lld\n", rank_id, pred_sum);

  //      if (rank_id == 1)
  //      {
  //        char* test_vid3 = new char[graph_slice->nodes];
  //        cudaMemcpy(test_vid3, graph_slice->d_visit_flags, graph_slice->nodes, cudaMemcpyDeviceToHost);
  //        printf("d_visit_flags: ");
  //        for (int i = 0; i < graph_slice->nodes; ++i)
  //        {
  //          printf("%d, ", test_vid3[i]);
  //        }
  //        printf("\n");
  //        delete[] test_vid3;
  //
  //        int* test_vid2 = new int[n_active_edges];
  //        cudaMemcpy(test_vid2, graph_slice->m_gatherMapTmp, n_active_edges * sizeof (int), cudaMemcpyDeviceToHost);
  //        printf("m_gatherMapTmp: ");
  //        for (int i = 0; i < n_active_edges; ++i)
  //        {
  //          printf("%d, ", test_vid2[i]);
  //        }
  //        printf("\n");
  //        delete[] test_vid2;
  //
  //        test_vid2 = new int[n_active_edges];
  //        cudaMemcpy(test_vid2, graph_slice->m_gatherDstsTmp, n_active_edges * sizeof (int), cudaMemcpyDeviceToHost);
  //        printf("m_gatherDstsTmp: ");
  //        for (int i = 0; i < n_active_edges; ++i)
  //        {
  //          printf("%d, ", test_vid2[i]);
  //        }
  //        printf("\n");
  //        delete[] test_vid2;
  //
  //        test_vid2 = new int[graph_slice->nodes];
  //        cudaMemcpy(test_vid2, graph_slice->vertex_list.d_labels, graph_slice->nodes * sizeof (int), cudaMemcpyDeviceToHost);
  //        printf("d_labels: ");
  //        for (int i = 0; i < graph_slice->nodes; ++i)
  //        {
  //          printf("%d, ", test_vid2[i]);
  //        }
  //        printf("\n");
  //        delete[] test_vid2;
  //
  //        test_vid2 = new int[frontier_size];
  //        cudaMemcpy(test_vid2, graph_slice->frontier_queues.d_keys[0], frontier_size * sizeof (int), cudaMemcpyDeviceToHost);
  //        printf("d_keys: ");
  //        for (int i = 0; i < frontier_size; ++i)
  //        {
  //          printf("%d, ", test_vid2[i]);
  //        }
  //        printf("\n");
  //        delete[] test_vid2;
  //
  //        test_vid2 = new int[frontier_size];
  //        cudaMemcpy(test_vid2, graph_slice->m_gatherTmp, frontier_size * sizeof (int), cudaMemcpyDeviceToHost);
  //        printf("m_gather: ");
  //        for (int i = 0; i < frontier_size; ++i)
  //        {
  //          printf("%d, ", test_vid2[i]);
  //        }
  //        printf("\n");
  //        delete[] test_vid2;
  //
  //      }
  //
  //      fflush(stdout);
  //      usleep(1000);
}

#endif	/* PREDEXTRACT_H */

