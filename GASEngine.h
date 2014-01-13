/*
   Copyright (C) SYSTAP, LLC 2006-2014.  All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef GASENGINE_H__
#define GASENGINE_H__

#define INDEBUG

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <omp.h>

// Utilities and correctness-checking
#include <test/b40c_test_util.h>

// Graph construction utils

#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/random.cuh>

// BFS includes
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/bfs/enactor_contract_expand.cuh>
#include <b40c/graph/bfs/enactor_expand_contract.cuh>
#include <b40c/graph/bfs/enactor_two_phase.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>
#include <b40c/graph/bfs/enactor_multi_gpu.cuh>

#include <b40c/graph/GASengine/csr_problem.cuh>
#include <b40c/graph/GASengine/enactor_vertex_centric.cuh>

using namespace b40c;
using namespace graph;
using namespace std;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

enum Strategy
{
  HOST = -1, EXPAND_CONTRACT, CONTRACT_EXPAND, TWO_PHASE, HYBRID, MULTI_GPU, VERTEX_CENTRIC
};

bool g_verbose;
bool g_verbose2;
const bool g_undirected = true; // Whether to add a "backedge" for every edge parsed/generated
bool g_quick; // Whether or not to perform CPU traversal as reference
const bool g_stream_from_host = false; // Whether or not to stream CSR representation from host mem
const bool g_with_value = true; // Whether or not to load value

enum GatherEdges
{
  NO_GATHER_EDGES, GATHER_IN_EDGES, GATHER_OUT_EDGES, GATHER_ALL_EDGES
};

enum ScatterEdges
{
  NO_SCATTER_EDGES, SCATTER_IN_EDGES, SCATTER_OUT_EDGES, SCATTER_ALL_EDGES
};

#ifdef GPU_DEVICE_NUMBER
__device__ __constant__ int d_iterations;
#else
int d_iterations;
#endif

template<typename Program, typename VertexType, typename EdgeType, typename GatherType, typename ReduceType>
class GASEngine
{
public:

  //  struct final_changed_transform : thrust::unary_function<thrust::tuple<VertexType&, int&>, int>
  //  {
  //    __device__
  //    int operator()(const thrust::tuple<VertexType&, int&> &t)
  //    {
  //      const VertexType& v = thrust::get < 0 > (t);
  //      const int& flag = thrust::get < 1 > (t);
  //      if(v.changed && flag) return 1;
  //      else return 0;
  //    }
  //  } ;

  void myPrint(const thrust::device_vector<int>& d_a)
  {
    thrust::host_vector<int> h_a = d_a;
    for (int i = 0; i < h_a.size(); i++)
    {
      printf("%d ", h_a[i]);
    }
    printf("\n");
  }

  struct set_one: thrust::binary_function<int, int, int>
  {

    __device__
    int operator()(const int& t1, const int& t2)
    {
      return 1;
    }
  };

  struct alledge: thrust::unary_function<thrust::tuple<int&, int&>, bool>
  {

    __device__
    bool operator()(const thrust::tuple<int&, int&> &t)
    {
      const int& dstactive = thrust::get < 0 > (t);
      const int& srcactive = thrust::get < 1 > (t);
      return (dstactive == 1 || srcactive == 1);
    }
  };

  struct graph_gather: thrust::unary_function<thrust::tuple<VertexType&, VertexType&, EdgeType&, int&>, GatherType>
  {

    __device__ GatherType operator()(const thrust::tuple<VertexType&, VertexType&, EdgeType&, int&> &t)
    {
      const int &flag = thrust::get < 3 > (t);
      if (!flag)
      {
        return 100000000;
      }
      else
      {
        const VertexType &dst = thrust::get < 0 > (t);
        const VertexType &src = thrust::get < 1 > (t);
        const EdgeType &edge = thrust::get < 2 > (t);
        typename Program::gather gather_functor;
        return gather_functor(dst, src, edge, flag);
      }
    }
  };

  struct graph_sum: thrust::binary_function<GatherType, GatherType, ReduceType>
  {

    __device__ ReduceType operator()(const GatherType &left, const GatherType & right)
    {
      typename Program::sum op;
      return op(left, right);
    }
  };

  //note the return type is arbitrary since it is ignored
  //but making the return type void doesn't seem to work

  struct graph_apply_gather: thrust::binary_function<VertexType, ReduceType, int>
  {

    __device__
    int operator()(VertexType &vertex_val, const ReduceType & accum)
    {
      typename Program::apply apply_functor;
      apply_functor(vertex_val, accum);
      return 0; //return value will be ignored, this is here to shut the compiler up
    }
  };

  struct graph_apply_nogather: thrust::unary_function<VertexType, int>
  {

    __device__ int operator()(VertexType & vertex_val)
    {
      typename Program::apply apply_functor;
      apply_functor(vertex_val);
      return 0; //return value will be ignored, this is here to shut the compiler up
    }
  };

  //edge state not currently represented
  //return value is discarded

  struct graph_scatter: thrust::binary_function<thrust::tuple<VertexType&, VertexType&, EdgeType&>, int &, int>
  {

    __device__
    int operator()(thrust::tuple<VertexType &, VertexType&, EdgeType&> &dst_src_edge, int &activeFlag)
    {
      VertexType &dst = thrust::get < 0 > (dst_src_edge);
      VertexType &src = thrust::get < 1 > (dst_src_edge);
      EdgeType &e = const_cast<EdgeType &>(thrust::get < 2 > (dst_src_edge));
      typename Program::scatter scatter_functor;

      int flag = scatter_functor(dst, src, e);
      if (flag) activeFlag = 1;

      return 0;
    }
  };

  struct graph_scatter_all: thrust::binary_function<thrust::tuple<VertexType&, VertexType&, EdgeType&, int&>, int &, int>
  {

    __device__
    int operator()(thrust::tuple<VertexType &, VertexType&, EdgeType&, int&> &dst_src_edge, int &activeFlag)
    {
      VertexType &dst = thrust::get < 0 > (dst_src_edge);
      VertexType &src = thrust::get < 1 > (dst_src_edge);
      EdgeType &e = const_cast<EdgeType &>(thrust::get < 2 > (dst_src_edge));
      int& srcflag = thrust::get < 3 > (dst_src_edge);
      typename Program::scatter scatter_functor;

      int flag = scatter_functor(dst, src, e);
      if (flag && srcflag) activeFlag = 1;

      return 0;
    }
  };

  //wrapper function to avoid needing to provide
  //edge state if the algorithm doesn't need it

  std::vector<int> run(CsrGraph<int, EdgeType, int>& csr_graph/*, thrust::device_vector<int> &d_edge_dst_vertex, thrust::device_vector<int> &d_edge_src_vertex,
   thrust::device_vector<VertexType> &d_vertex_vals, std::vector<thrust::device_vector<int> > &d_active_vertex_flags, int maxiter*/)
  {
    thrust::device_vector<int> d_dummy_edge_vals;

    return run(csr_graph/*, d_edge_dst_vertex, d_edge_src_vertex, d_vertex_vals, d_dummy_edge_vals, d_active_vertex_flags, maxiter*/);
  }

  std::vector<int> run(GASengine::CsrProblem<int, int, int, false, false>& csr_problem/*
   , thrust::device_vector<int> &d_edge_dst_vertex, thrust::device_vector<int> &d_edge_src_vertex,
   thrust::device_vector<VertexType> &d_vertex_vals, thrust::device_vector<EdgeType> &d_edge_vals, std::vector<thrust::device_vector<int> > &d_active_vertex_flags, int maxiter
   */)
  {
    typedef int VertexId;
    typedef int SizeT;
    typedef int Value;
    const bool MARK_PREDECESSORS = false;
    const bool WITH_VALUE = false;
    const bool INSTRUMENT = true;

    VertexId src = -1;	// Use whatever the specified graph-type's default is
    bool mark_pred = false; // Whether or not to mark src-distance vs. parent vertices
    bool with_value = false; // Whether or not to include edge/node computation
    int test_iterations = 1;
    int max_grid_size = 0; // Maximum grid size (0: leave it up to the enactor)
    int num_gpus = 1; // Number of GPUs for multi-gpu enactor to use
//    double max_queue_sizing = 1.3; // Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).
    double max_queue_sizing = 4.5;
    std::vector<int> strategies(1, VERTEX_CENTRIC);
//    typedef GASengine::CsrProblem<VertexId, SizeT, Value, MARK_PREDECESSORS, WITH_VALUE> CsrProblem;

    // Allocate host-side label array (for both reference and gpu-computed results)
//    VertexId* reference_labels = (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
//    VertexId* h_labels = (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
//    int* h_dists = (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
//    VertexId* reference_check = (g_quick) ? NULL : reference_labels;
//
//    //Allocate host-side node_value array (both ref and gpu-computed results)
//    Value* ref_node_values = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
//    Value* h_node_values = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
//    Value* ref_node_value_check = (g_quick) ? NULL : ref_node_values;
//
//    //Allocate host-side sigma value array (both ref and gpu-computed results)
//    Value* ref_sigmas = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
//    Value* h_sigmas = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
//    Value* ref_sigmas_check = (g_quick) ? NULL : ref_sigmas;
//    Value* h_deltas = (Value*) malloc(sizeof(Value) * csr_graph.nodes);

    // Allocate BFS enactor map
    GASengine::EnactorVertexCentric<INSTRUMENT> vertex_centric(g_verbose);

//    // Allocate problem on GPU
//    CsrProblem csr_problem;
//    if (csr_problem.FromHostProblem(g_stream_from_host, csr_graph.nodes, csr_graph.edges,
//                                    csr_graph.column_indices, csr_graph.row_offsets, csr_graph.edge_values,
//                                    csr_graph.row_indices, csr_graph.column_offsets, csr_graph.node_values, num_gpus))
//      exit(1);

//    printf("got here.\n");

    cudaError_t retval = cudaSuccess;

    retval = vertex_centric.EnactIterativeSearch(csr_problem, src, max_grid_size, max_queue_sizing);

    if (retval && (retval != cudaErrorInvalidDeviceFunction))
    {
      exit(1);
    }

//    csr_problem.ExtractResults(h_dists, h_node_values, h_sigmas, h_deltas);
//
//    for(int i=0; i<csr_graph.nodes; i++)
//    {
//      printf("after dist[%d] = %d\n", i, h_dists[i]);
//    }

    std::vector<int> ret(2, 0);
//	        ret[0] = iterations;
//	        ret[1] = selector;
    return ret;

    /*   typedef typename thrust::device_vector<VertexType>::iterator VertexValueIterator;
     typedef thrust::device_vector<int>::iterator IndexIterator;
     typedef thrust::permutation_iterator<VertexValueIterator, IndexIterator> permuteIterator;
     typedef thrust::permutation_iterator<IndexIterator, IndexIterator> indexPermuteIterator;
     typedef thrust::tuple<permuteIterator, permuteIterator, typename thrust::device_vector<EdgeType>::iterator> permuteIteratorTuple;
     typedef thrust::zip_iterator<permuteIteratorTuple> zipIterator;

     typedef thrust::tuple<permuteIterator, permuteIterator, typename thrust::device_vector<EdgeType>::iterator, indexPermuteIterator> permuteIteratorTuple2;
     typedef thrust::zip_iterator<permuteIteratorTuple2> zipIterator2;

     const int numEdges = d_edge_dst_vertex.size();
     const int numVertices = d_vertex_vals.size();

     //create some temporary storage we'll need
     thrust::device_vector<ReduceType> d_vertex_accum(numVertices);
     thrust::device_vector<int> d_participating_vertices(numVertices);

     permuteIterator srcValsIt(d_vertex_vals.begin(), d_edge_src_vertex.begin());
     permuteIterator dstValsIt(d_vertex_vals.begin(), d_edge_dst_vertex.begin());
     thrust::transform_iterator<graph_gather, zipIterator, GatherType> graph_gather_iterator(thrust::make_zip_iterator(thrust::make_tuple(dstValsIt, srcValsIt, d_edge_vals.begin())), graph_gather());
     permuteIterator vertexVals(d_vertex_vals.begin(), d_participating_vertices.begin());


     //sort by src
     thrust::device_vector<int> d_edge_dst_vertex2 = d_edge_dst_vertex;
     thrust::device_vector<int> d_edge_src_vertex2 = d_edge_src_vertex;
     thrust::device_vector<EdgeType> &d_edge_vals2 = d_edge_vals;

     thrust::sort_by_key(d_edge_src_vertex2.begin(), d_edge_src_vertex2.end(), thrust::make_zip_iterator(
     thrust::make_tuple(
     d_edge_dst_vertex2.begin(),
     d_edge_vals2.begin())));

     permuteIterator srcValsIt2(d_vertex_vals.begin(), d_edge_src_vertex2.begin());
     permuteIterator dstValsIt2(d_vertex_vals.begin(), d_edge_dst_vertex2.begin());

     int selector = 0;
     int iterations = 0;
     thrust::device_vector<int> seq(numEdges);
     thrust::device_vector<int> edge_front(numEdges);
     thrust::sequence(seq.begin(), seq.end());
     thrust::device_vector<int> edge_flags(numEdges, 0);

     #ifdef INDEBUG
     double gathertime = 0.0;
     //double applytime = 0.0;
     double scattertime = 0.0;
     double startTime;
     #endif

     for (int i = 0; i < maxiter; i++)
     //    for (;;)
     {
     #ifdef GPU_DEVICE_NUMBER
     cudaMemcpyToSymbol(d_iterations, &iterations, sizeof (int));
     #else
     d_iterations = iterations;
     #endif

     #ifdef INDEBUG
     startTime = omp_get_wtime();
     #endif
     //gather
     if (Program::gatherOverEdges() == GATHER_IN_EDGES)
     {
     //        indexPermuteIterator flagIt(d_active_vertex_flags[selector].begin(), d_edge_dst_vertex.begin());
     //        thrust::transform_iterator<graph_gather, zipIterator2, GatherType> graph_gather_iterator2(thrust::make_zip_iterator(thrust::make_tuple(dstValsIt, srcValsIt, d_edge_vals.begin(), flagIt)), graph_gather());
     //
     //        thrust::fill(edge_flags.begin(), edge_flags.end(), 0);
     //        //compute active edges
     //        thrust::transform_if(edge_flags.begin(), edge_flags.end(), edge_flags.begin(), flagIt, edge_flags.begin(), set_one(), thrust::identity<int>());
     //
     //#ifdef INDEBUG
     //        //        thrust::host_vector<int> tmp = edge_flags;
     //        //        for (int i = 0; i < tmp.size(); i++)
     //        //        {
     //        //          printf("edge_flags[%d] = %d\n", i, tmp[i]);
     //        //        }
     //#endif
     //
     //
     //        thrust::device_vector<int>::iterator new_end = thrust::copy_if(seq.begin(), seq.end(), edge_flags.begin(), edge_front.begin(), thrust::identity<int>());
     //        int num_active_edges = new_end - edge_front.begin();
     //
     //#ifdef INDEBUG
     //        printf("num_active_edges = %d\n", num_active_edges);
     //#endif
     //
     //#ifdef GPU_DEVICE_NUMBER
     //        cudaMemcpyToSymbol(d_iterations, &iterations, sizeof (int));
     //#else
     //        d_iterations = iterations;
     //#endif
     //
     //        indexPermuteIterator frontIt(d_edge_dst_vertex.begin(), edge_front.begin());
     //        thrust::permutation_iterator<thrust::transform_iterator<graph_gather, zipIterator2, GatherType>, IndexIterator> graph_gather_permute_iterator(graph_gather_iterator2, edge_front.begin());
     //        //        thrust::pair<IndexIterator, typename thrust::device_vector<ReduceType>::iterator> it =
     //        //                thrust::reduce_by_key(d_edge_dst_vertex.begin(),
     //        //                                      d_edge_dst_vertex.end(),
     //        //                                      graph_gather_iterator2,
     //        //                                      d_participating_vertices.begin(),
     //        //                                      d_vertex_accum.begin(),
     //        //                                      thrust::equal_to<int>(),
     //        //                                      graph_sum());
     //
     //        thrust::pair<IndexIterator, typename thrust::device_vector<ReduceType>::iterator> it =
     //                thrust::reduce_by_key(frontIt,
     //                                      frontIt + num_active_edges,
     //                                      graph_gather_permute_iterator,
     //                                      d_participating_vertices.begin(),
     //                                      d_vertex_accum.begin(),
     //                                      thrust::equal_to<int>(),
     //                                      graph_sum());
     //
     //        const int numParticipatingVertces = it.first - d_participating_vertices.begin();
     //
     //#ifdef INDEBUG
     //        //        printf("numParticipatingVertces = %d\n", numParticipatingVertces);
     //#endif
     //
     //        thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     //                thrust::device_vector<int>::iterator>
     //                activeVertices(d_active_vertex_flags[selector].begin(), d_participating_vertices.begin());
     //
     //        //apply
     //        thrust::transform_if(vertexVals,
     //                             vertexVals + numParticipatingVertces,
     //                             d_vertex_accum.begin(),
     //                             activeVertices,
     //                             thrust::make_discard_iterator(),
     //                             graph_apply_gather(),
     //                             thrust::identity<int>());

     indexPermuteIterator flagIt(d_active_vertex_flags[selector].begin(), d_edge_dst_vertex.begin());
     thrust::transform_iterator<graph_gather, zipIterator2, GatherType> graph_gather_iterator2(thrust::make_zip_iterator(thrust::make_tuple(dstValsIt, srcValsIt, d_edge_vals.begin(), flagIt)), graph_gather());
     #ifdef GPU_DEVICE_NUMBER
     cudaMemcpyToSymbol(d_iterations, &iterations, sizeof (int));
     #else
     d_iterations = iterations;
     #endif
     thrust::pair<IndexIterator, typename thrust::device_vector<ReduceType>::iterator> it =
     thrust::reduce_by_key(d_edge_dst_vertex.begin(),
     d_edge_dst_vertex.end(),
     graph_gather_iterator2,
     d_participating_vertices.begin(),
     d_vertex_accum.begin(),
     thrust::equal_to<int>(),
     graph_sum());

     const int numParticipatingVertces = it.first - d_participating_vertices.begin();

     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     activeVertices(d_active_vertex_flags[selector].begin(), d_participating_vertices.begin());

     //apply
     thrust::transform_if(vertexVals,
     vertexVals + numParticipatingVertces,
     d_vertex_accum.begin(),
     activeVertices,
     thrust::make_discard_iterator(),
     graph_apply_gather(),
     thrust::identity<int>());
     }
     else if (Program::gatherOverEdges() == GATHER_OUT_EDGES)
     {
     indexPermuteIterator flagIt(d_active_vertex_flags[selector].begin(), d_edge_src_vertex.begin());
     thrust::transform_iterator<graph_gather, zipIterator2, GatherType> graph_gather_iterator2(thrust::make_zip_iterator(thrust::make_tuple(dstValsIt, srcValsIt, d_edge_vals.begin(), flagIt)), graph_gather());
     #ifdef GPU_DEVICE_NUMBER
     cudaMemcpyToSymbol(d_iterations, &iterations, sizeof (int));
     #else
     d_iterations = iterations;
     #endif
     thrust::pair<IndexIterator, typename thrust::device_vector<ReduceType>::iterator> it =
     thrust::reduce_by_key(d_edge_src_vertex.begin(),
     d_edge_src_vertex.end(),
     graph_gather_iterator2,
     d_participating_vertices.begin(),
     d_vertex_accum.begin(),
     thrust::equal_to<int>(),
     graph_sum());

     const int numParticipatingVertces = it.first - d_participating_vertices.begin();

     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     activeVertices(d_active_vertex_flags[selector].begin(), d_participating_vertices.begin());

     //apply
     thrust::transform_if(vertexVals,
     vertexVals + numParticipatingVertces,
     d_vertex_accum.begin(),
     activeVertices,
     thrust::make_discard_iterator(),
     graph_apply_gather(),
     thrust::identity<int>());
     }
     else
     {
     //NO EDGES, no gather, just transform
     //apply
     thrust::transform_if(d_vertex_vals.begin(),
     d_vertex_vals.end(),
     d_active_vertex_flags[selector].begin(),
     thrust::make_discard_iterator(),
     graph_apply_nogather(),
     thrust::identity<int>());
     }

     #ifdef INDEBUG
     #ifdef GPU_DEVICE_NUMBER
     cudaDeviceSynchronize();
     #endif
     double elapsed1 = (omp_get_wtime() - startTime)*1000;

     startTime = omp_get_wtime();

     #endif
     //scatter phase, go over each (active) edge and set new active vals
     if (Program::scatterOverEdges() == SCATTER_OUT_EDGES)
     {
     //        //compute front edges
     //        indexPermuteIterator wasActiveFlagIt(d_active_vertex_flags[selector].begin(), d_edge_src_vertex.begin());
     //
     //#ifdef INDEBUG
     //        thrust::host_vector<int> h_wasActiveFlagIt(numEdges);
     //        thrust::copy(wasActiveFlagIt, wasActiveFlagIt+numEdges, h_wasActiveFlagIt.begin());
     //        for(int i=0; i<numEdges; i++)
     //          printf("wasActiveFlagIt[%d] = %d\n", i, h_wasActiveFlagIt[i]);
     //        myPrint(d_active_vertex_flags[selector]);
     //#endif
     //
     //        thrust::fill(edge_flags.begin(), edge_flags.end(), 0);
     //        //compute active edges
     //        thrust::transform_if(edge_flags.begin(), edge_flags.end(), edge_flags.begin(), wasActiveFlagIt, edge_flags.begin(), set_one(), thrust::identity<int>());
     //        thrust::device_vector<int>::iterator new_end = thrust::copy_if(seq.begin(), seq.end(), edge_flags.begin(), edge_front.begin(), thrust::identity<int>());
     //        int num_active_edges = new_end - edge_front.begin();
     //
     //#ifdef INDEBUG
     //        printf("Scatter: num_active_edges = %d\n", num_active_edges);
     //#endif
     //
     //        thrust::fill(d_active_vertex_flags[selector ^ 1].begin(), d_active_vertex_flags[selector ^ 1].end(), 0);
     //
     //        thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     //                thrust::device_vector<int>::iterator>
     //                willBeActiveFlagIt(d_active_vertex_flags[selector ^ 1].begin(), d_edge_dst_vertex.begin());
     //
     //        thrust::permutation_iterator<zipIterator, IndexIterator> graph_scatter_iterator(thrust::make_zip_iterator(thrust::make_tuple(dstValsIt, srcValsIt, d_edge_vals.begin())), edge_front.begin());
     //        thrust::permutation_iterator<thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     //                thrust::device_vector<int>::iterator>, IndexIterator> willBeActiveFlagPermuteIt(willBeActiveFlagIt, edge_front.begin());
     //
     //        thrust::transform(graph_scatter_iterator, graph_scatter_iterator + num_active_edges,
     //                          willBeActiveFlagPermuteIt,
     //                          thrust::make_discard_iterator(),
     //                          graph_scatter());

     thrust::fill(d_active_vertex_flags[selector ^ 1].begin(), d_active_vertex_flags[selector ^ 1].end(), 0);

     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     wasActiveFlagIt(d_active_vertex_flags[selector].begin(), d_edge_src_vertex.begin());
     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     willBeActiveFlagIt(d_active_vertex_flags[selector ^ 1].begin(), d_edge_dst_vertex.begin());

     thrust::transform_if(thrust::make_zip_iterator(thrust::make_tuple(
     dstValsIt, srcValsIt, d_edge_vals.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(
     dstValsIt + numEdges, srcValsIt + numEdges, d_edge_vals.end())),
     willBeActiveFlagIt,
     wasActiveFlagIt,
     thrust::make_discard_iterator(),
     graph_scatter(),
     thrust::identity<int>());


     }
     else if (Program::scatterOverEdges() == SCATTER_IN_EDGES)
     {
     thrust::fill(d_active_vertex_flags[selector ^ 1].begin(), d_active_vertex_flags[selector ^ 1].end(), 0);

     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     wasActiveFlagIt(d_active_vertex_flags[selector].begin(), d_edge_dst_vertex.begin());
     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     willBeActiveFlagIt(d_active_vertex_flags[selector ^ 1].begin(), d_edge_src_vertex.begin());

     thrust::transform_if(thrust::make_zip_iterator(thrust::make_tuple(
     dstValsIt, srcValsIt, d_edge_vals.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(
     dstValsIt + numEdges, srcValsIt + numEdges, d_edge_vals.end())),
     willBeActiveFlagIt,
     wasActiveFlagIt,
     thrust::make_discard_iterator(),
     graph_scatter(),
     thrust::identity<int>());

     }
     else if (Program::scatterOverEdges() == SCATTER_ALL_EDGES)
     {
     thrust::fill(d_active_vertex_flags[selector ^ 1].begin(), d_active_vertex_flags[selector ^ 1].end(), 0);

     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     wasActiveFlagItDst(d_active_vertex_flags[selector].begin(), d_edge_dst_vertex.begin());
     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     wasActiveFlagItSrc(d_active_vertex_flags[selector].begin(), d_edge_src_vertex.begin());

     thrust::permutation_iterator<thrust::device_vector<int>::iterator,
     thrust::device_vector<int>::iterator>
     willBeActiveFlagIt(d_active_vertex_flags[selector ^ 1].begin(), d_edge_dst_vertex.begin());

     thrust::transform_if(thrust::make_zip_iterator(thrust::make_tuple(
     dstValsIt, srcValsIt, d_edge_vals.begin(), wasActiveFlagItSrc)),
     thrust::make_zip_iterator(thrust::make_tuple(
     dstValsIt + numEdges, srcValsIt + numEdges, d_edge_vals.end(), wasActiveFlagItSrc)),
     willBeActiveFlagIt,
     thrust::make_zip_iterator(thrust::make_tuple(wasActiveFlagItDst, wasActiveFlagItSrc)),
     thrust::make_discard_iterator(),
     graph_scatter_all(),
     alledge());

     }
     else
     {
     }

     #ifdef INDEBUG
     double elapsed2 = (omp_get_wtime() - startTime)*1000;
     #endif
     selector ^= 1;

     int numActive = thrust::reduce(d_active_vertex_flags[selector].begin(),
     d_active_vertex_flags[selector].end());
     //      std::vector<int> tmp(d_active_vertex_flags[selector].size());
     //      thrust::copy(d_active_vertex_flags[selector].begin(), d_active_vertex_flags[selector].end(), tmp.begin());
     //
     //            std::cout << "numActive: " << numActive << ", iteration: " << iterations << "\n";

     if (numActive == 0)
     break;

     #ifdef INDEBUG
     gathertime += elapsed1;
     scattertime += elapsed2;
     #endif

     iterations++;
     }

     #ifdef INDEBUG
     printf("gather time is: %f\n", gathertime);
     printf("scatter time is: %f\n", scattertime);
     #endif

     std::vector<int> ret(2);
     ret[0] = iterations;
     ret[1] = selector;
     return ret;
     */
  }
};

#endif
