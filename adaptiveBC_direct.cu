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

typedef unsigned int uint;

#include "GASEngine.h"
#include "adaptiveBC_direct.h"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "graphio.h"
#include <cstdlib>

void generateRandomGraph(std::vector<int> &h_edge_src_vertex,
                         std::vector<int> &h_edge_dst_vertex,
                         int numVertices, int avgEdgesPerVertex) {
  thrust::minstd_rand rng;
  thrust::random::experimental::normal_distribution<float> n_dist(avgEdgesPerVertex, sqrtf(avgEdgesPerVertex));
  thrust::uniform_int_distribution<int> u_dist(0, numVertices - 1);

  for (int v = 0; v < numVertices; ++v) {
    int numEdges = min(max((int)roundf(n_dist(rng)), 1), 1000);
    for (int e = 0; e < numEdges; ++e) {
      uint dst_v = u_dist(rng);
      h_edge_src_vertex.push_back(v);
      h_edge_dst_vertex.push_back(dst_v);
    }
  }
}

int main(int argc, char **argv) {

  int numVertices;
  const char* outFileName = 0;

  //generate simple random graph
  std::vector<int> h_edge_src_vertex;
  std::vector<int> h_edge_dst_vertex;
  std::vector<int> h_edge_data;

  if (argc == 1) {
    numVertices = 1000000;
    const int avgEdgesPerVertex = 10;
    generateRandomGraph(h_edge_src_vertex, h_edge_dst_vertex, numVertices, avgEdgesPerVertex);
    h_edge_data.reserve(h_edge_src_vertex.size());
    for (int i = 0; i < h_edge_src_vertex.size(); ++i) {
      h_edge_data.push_back(rand() % 100);
    }
  }
  else if (argc == 2 || argc == 3) {
    loadGraph( argv[1], numVertices, h_edge_src_vertex, h_edge_dst_vertex, &h_edge_data );
    if (argc == 3)
      outFileName = argv[2];
  }
  else {
    std::cerr << "Too many arguments!" << std::endl;
    exit(1);
  }

  const uint numEdges = h_edge_src_vertex.size();

  thrust::device_vector<int> d_edge_src_vertex = h_edge_src_vertex; //sort by dst
  thrust::device_vector<int> d_edge_dst_vertex = h_edge_dst_vertex; //sort by dst
  thrust::device_vector<int> d_edge_data = h_edge_data;             //sort by dst
  
  thrust::device_vector<int> d_edge_src_vertex2 = h_edge_src_vertex; //sort by src
  thrust::device_vector<int> d_edge_dst_vertex2 = h_edge_dst_vertex; //sort by src
  thrust::device_vector<int> d_edge_data2 = h_edge_data;             //sort by src

  //use PSW ordering
  thrust::sort_by_key(d_edge_dst_vertex.begin(), d_edge_dst_vertex.end(), thrust::make_zip_iterator(
                                                                          thrust::make_tuple(
                                                                            d_edge_src_vertex.begin(),
                                                                            d_edge_data.begin())));
  
  //sort by source
  thrust::sort_by_key(d_edge_src_vertex2.begin(), d_edge_src_vertex2.end(), thrust::make_zip_iterator(
                                                                          thrust::make_tuple(
                                                                            d_edge_dst_vertex2.begin(),
                                                                            d_edge_data2.begin())));

  thrust::device_vector<adaptiveBC::VertexType> d_vertex_data(numVertices); //each vertex value starts at "infinity"

  std::vector<thrust::device_vector<int> > d_active_vertex_flags;
  {
    thrust::device_vector<int> foo;
    d_active_vertex_flags.push_back(foo);
    d_active_vertex_flags.push_back(foo);
  }

  //find max out degree vertex to start
  int startVertex;

  //one vertex starts active in sssp
  d_active_vertex_flags[0].resize(numVertices, 0);
  d_active_vertex_flags[1].resize(numVertices, 0);

  GASEngine<adaptiveBC, adaptiveBC::VertexType, int, int, int> forward_engine;
  GASEngine<adaptiveBC_backward, adaptiveBC_backward::VertexType, int, double, double> backward_engine;
  thrust::device_vector<double> d_vertex_BC(numVertices, 0.0);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  for(int seed = 0; seed < numVertices; seed++)
  {
    thrust::fill(d_active_vertex_flags[0].begin(), d_active_vertex_flags[0].end(), 0);
    thrust::fill(d_active_vertex_flags[1].begin(), d_active_vertex_flags[1].end(), 0);
    thrust::fill(d_vertex_data.begin(), d_vertex_data.end(), adaptiveBC::VertexType(10000000, 0, false, 0.0));
    
    startVertex = seed;
    d_active_vertex_flags[0][startVertex] = 1;
    d_vertex_data[startVertex] = adaptiveBC::VertexType(0, 1, true, 0.0);
    int diameter = forward_engine.run(d_edge_dst_vertex,
                            d_edge_src_vertex,
                            d_vertex_data,
                            d_edge_data,
                            d_active_vertex_flags, 1000000);
  
  //reinit active
    reinit_active_flags(d_vertex_data, d_active_vertex_flags[0], diameter);
    thrust::fill(d_active_vertex_flags[1].begin(), d_active_vertex_flags[1].end(), 0);

    int diameter2 = backward_engine.run(d_edge_dst_vertex2,
                              d_edge_src_vertex2,
                              d_vertex_data,
                              d_edge_data2,
                              d_active_vertex_flags,100000000);
    accum_BC(d_vertex_BC, d_vertex_data, startVertex);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  std::cout << "Took: " << elapsed << " ms" << std::endl;

  if (outFileName) {
    FILE* f = fopen(outFileName, "w");
    std::vector<adaptiveBC::VertexType> h_vertex_data(d_vertex_data.size());
    thrust::copy(d_vertex_data.begin(), d_vertex_data.end(), h_vertex_data.begin());
    std::vector<double> h_bc(d_vertex_BC.size());
    thrust::copy(d_vertex_BC.begin(), d_vertex_BC.end(), h_bc.begin());

    for ( int i = 0; i < numVertices; ++i)
    {
      fprintf( f, "%d\t%d\t%d\t%d\t%f\n", i, h_vertex_data[i].dist, h_vertex_data[i].sigma, h_vertex_data[i].changed, h_bc[i]);
    }

    fclose(f);
  }
  return 0;
}

