/*********************************************************

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

**********************************************************/

/* Written by Erich Elsen and Vishal Vaidyanathan
   of Royal Caliber, LLC
   Contact us at: info@royal-caliber.com
*/


#ifndef GASENGINE_H__
#define GASENGINE_H__

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <iostream>

enum GatherEdges { NO_GATHER_EDGES, GATHER_IN_EDGES };

enum ScatterEdges { NO_SCATTER_EDGES, SCATTER_OUT_EDGES };

__device__ __constant__ int d_iterations;

template<typename Program,
         typename VertexType,
         typename EdgeType,
         typename GatherType,
         typename ReduceType>
class GASEngine {
  public:
  struct graph_gather : thrust::unary_function<thrust::tuple<VertexType&, VertexType&, EdgeType&>, GatherType> {
    __device__
      GatherType operator()(const thrust::tuple<VertexType&, VertexType&, EdgeType&> &t) {
        const VertexType &dst = thrust::get<0>(t);
        const VertexType &src = thrust::get<1>(t);
        const EdgeType  &edge = thrust::get<2>(t);
        typename Program::gather gather_functor;
        return gather_functor(dst, src, edge);
      }
  };

  struct graph_sum : thrust::binary_function<GatherType, GatherType, ReduceType> {
    __device__
      ReduceType operator()(const GatherType &left, const GatherType &right) {
        typename Program::sum op;
        return op(left, right);
      }
  };

  //note the return type is arbitrary since it is ignored
  //but making the return type void doesn't seem to work
  struct graph_apply_gather : thrust::binary_function<VertexType, ReduceType, int> {
    __device__
      int operator()(VertexType &vertex_val, const ReduceType &accum) {
        typename Program::apply apply_functor;
        apply_functor(vertex_val, accum);
        return 0; //return value will be ignored, this is here to shut the compiler up
      }
  };

  struct graph_apply_nogather : thrust::unary_function<VertexType, int> {
    __device__
      int operator()(VertexType &vertex_val) {
        typename Program::apply apply_functor;
        apply_functor(vertex_val);
        return 0; //return value will be ignored, this is here to shut the compiler up
      }
  };

  //edge state not currently represented
  //return value is discarded
  struct graph_scatter : thrust::binary_function<thrust::tuple<VertexType&, VertexType&, EdgeType&>, int &, int> {
    __device__
      int operator()(const thrust::tuple<VertexType &, VertexType&, EdgeType&> &dst_src_edge, int &activeFlag) {
        const VertexType &dst = thrust::get<0>(dst_src_edge);
        const VertexType &src = thrust::get<1>(dst_src_edge);
        EdgeType &e           = const_cast<EdgeType &>(thrust::get<2>(dst_src_edge));
        typename Program::scatter scatter_functor;

        int flag = scatter_functor(dst, src, e);
        if (flag)
          activeFlag = 1;

        return 0;
      }
  };

  //wrapper function to avoid needing to provide
  //edge state if the algorithm doesn't need it
  int run(thrust::device_vector<int> &d_edge_dst_vertex,
          thrust::device_vector<int> &d_edge_src_vertex,
          thrust::device_vector<VertexType> &d_vertex_vals,
          std::vector<thrust::device_vector<int> > &d_active_vertex_flags)
  {
    thrust::device_vector<int> d_dummy_edge_vals;

    return run(d_edge_dst_vertex,
               d_edge_src_vertex,
               d_vertex_vals,
               d_dummy_edge_vals,
               d_active_vertex_flags);
  }


  int run(thrust::device_vector<int> &d_edge_dst_vertex,
          thrust::device_vector<int> &d_edge_src_vertex,
          thrust::device_vector<VertexType> &d_vertex_vals,
          thrust::device_vector<EdgeType>   &d_edge_vals,
          std::vector<thrust::device_vector<int> > &d_active_vertex_flags)
  {
    typedef typename thrust::device_vector<VertexType>::iterator VertexValueIterator;
    typedef thrust::device_vector<int>::iterator IndexIterator;
    typedef thrust::permutation_iterator<VertexValueIterator, IndexIterator> permuteIterator;
    typedef thrust::tuple<permuteIterator, permuteIterator, typename thrust::device_vector<EdgeType>::iterator> permuteIteratorTuple;
    typedef thrust::zip_iterator<permuteIteratorTuple> zipIterator;

    const int numEdges = d_edge_dst_vertex.size();
    const int numVertices = d_vertex_vals.size();

    //create some temporary storage we'll need
    thrust::device_vector<ReduceType> d_vertex_accum(numVertices);
    thrust::device_vector<int>        d_participating_vertices(numVertices);

    permuteIterator srcValsIt(d_vertex_vals.begin(), d_edge_src_vertex.begin());
    permuteIterator dstValsIt(d_vertex_vals.begin(), d_edge_dst_vertex.begin());
    thrust::transform_iterator<graph_gather, zipIterator, GatherType> graph_gather_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(dstValsIt, srcValsIt, d_edge_vals.begin())), graph_gather());

    permuteIterator vertexVals(d_vertex_vals.begin(), d_participating_vertices.begin());

    int selector = 0;
    int iterations = 0;

    for (;;) {
      cudaMemcpyToSymbol(d_iterations, &iterations, sizeof(int));
      if (Program::gatherOverEdges() == GATHER_IN_EDGES) {
        //gather
        thrust::pair<IndexIterator, typename thrust::device_vector<ReduceType>::iterator> it =
          thrust::reduce_by_key(d_edge_dst_vertex.begin(),
                                d_edge_dst_vertex.end(),
                                graph_gather_iterator,
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
      else {
        //NO EDGES, no gather, just transform
        //apply
        thrust::transform_if(d_vertex_vals.begin(),
                             d_vertex_vals.end(),
                             d_active_vertex_flags[selector].begin(),
                             thrust::make_discard_iterator(),
                             graph_apply_nogather(),
                             thrust::identity<int>());
      }

      //scatter phase, go over each (active) edge and set new active vals

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

      selector ^= 1;

      int numActive = thrust::reduce(d_active_vertex_flags[selector].begin(),
                                     d_active_vertex_flags[selector].end());

      std::cout << "numActive: " << numActive << "\n";

      if (numActive == 0)
        break;

      iterations++;
    }
    return iterations;
  }

};

#endif
