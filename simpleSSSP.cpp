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
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "graphio.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
using namespace std;
#include "sssp.h"

void generateRandomGraph(std::vector<int> &h_edge_src_vertex,
                         std::vector<int> &h_edge_dst_vertex,
                         int numVertices, int avgEdgesPerVertex)
{
  thrust::minstd_rand rng;
  thrust::random::normal_distribution<float> n_dist(avgEdgesPerVertex, sqrtf(avgEdgesPerVertex));
  thrust::uniform_int_distribution<int> u_dist(0, numVertices - 1);

  for (int v = 0; v < numVertices; ++v) {
    int numEdges = std::min(std::max((int) roundf(n_dist(rng)), 1), 1000);
    for (int e = 0; e < numEdges; ++e) {
      uint dst_v = u_dist(rng);
      h_edge_src_vertex.push_back(v);
      h_edge_dst_vertex.push_back(dst_v);
    }
  }
}

void run_test(thrust::device_vector<int> &d_edge_dst_vertex,
              thrust::device_vector<int> &d_edge_src_vertex,
              thrust::device_vector<sssp::VertexType> &d_vertex_data,
              thrust::device_vector<int> d_edge_data,
              std::vector<thrust::device_vector<int> > &d_active_vertex_flags, int startVertex, int numEdges)
{
  thrust::fill(d_active_vertex_flags[0].begin(), d_active_vertex_flags[0].end(), 0);
  thrust::fill(d_active_vertex_flags[1].begin(), d_active_vertex_flags[1].end(), 0);
  d_active_vertex_flags[0][startVertex] = 1;
  d_vertex_data[startVertex] = sssp::VertexType(0, true);

  std::vector<int> ret(2);
  GASEngine<sssp, sssp::VertexType, int, int, int> engine;

  double startTime = omp_get_wtime();

  ret = engine.run(d_edge_dst_vertex,
                   d_edge_src_vertex,
                   d_vertex_data,
                   d_edge_data,
                   d_active_vertex_flags, INT_MAX);

#ifdef GPU_DEVICE_NUMBER
  cudaDeviceSynchronize();
#endif
  double elapsed = (omp_get_wtime() - startTime)*1000;

  int diameter = ret[0];
  std::cout << "Took: " << elapsed << " ms" << std::endl;
  std::cout << "Iterations to convergence: " << diameter << std::endl;
}

int main(int argc, char **argv)
{

  int numVertices;
  const char* outFileName = 0;

#ifdef GPU_DEVICE_NUMBER
  cudaSetDevice(GPU_DEVICE_NUMBER);
  cerr << "Running on device " << GPU_DEVICE_NUMBER << endl;
#endif

  //generate simple random graph
  std::vector<int> h_edge_src_vertex;
  std::vector<int> h_edge_dst_vertex;
  std::vector<int> h_edge_data;
  int ispattern;
  int startVertex = 0;

  if (argc == 1) {
    numVertices = 1000000;
    const int avgEdgesPerVertex = 10;
    generateRandomGraph(h_edge_src_vertex, h_edge_dst_vertex, numVertices, avgEdgesPerVertex);
    h_edge_data.reserve(h_edge_src_vertex.size());
    for (int i = 0; i < h_edge_src_vertex.size(); ++i) {
      h_edge_data.push_back(rand() % 100);
    }
  }
  else if (argc == 4) {
    ispattern = loadGraph(argv[1], numVertices, h_edge_src_vertex, h_edge_dst_vertex, &h_edge_data);
    outFileName = argv[2];
    startVertex = atoi(argv[3]);
  }
  else {
    std::cerr << "Usage: ./simpleSSSP graph (.mtx) output_filename start_vertex" << std::endl;
    exit(1);
  }

  const uint numEdges = h_edge_src_vertex.size();
  if (ispattern == 0) // if it is pattern mtx
    h_edge_data = std::vector<int>(numEdges, 1);

  thrust::device_vector<int> d_edge_src_vertex = h_edge_src_vertex;
  thrust::device_vector<int> d_edge_dst_vertex = h_edge_dst_vertex;
  thrust::device_vector<int> d_edge_data = h_edge_data;

  //use PSW ordering
  thrust::sort_by_key(d_edge_dst_vertex.begin(), d_edge_dst_vertex.end(), thrust::make_zip_iterator(
                                                                                                    thrust::make_tuple(
                                                                                                                       d_edge_src_vertex.begin(),
                                                                                                                       d_edge_data.begin())));
  std::vector<thrust::device_vector<int> > d_active_vertex_flags;
  {
    thrust::device_vector<int> foo;
    d_active_vertex_flags.push_back(foo);
    d_active_vertex_flags.push_back(foo);
  }

  //find max out degree vertex to start
  std::vector<char> existing_vertices(numVertices, 0);

  std::vector<int> h_out_edges(numVertices);
  for (int i = 0; i < h_edge_src_vertex.size(); ++i) {
    h_out_edges[h_edge_src_vertex[i]]++;
    existing_vertices[h_edge_src_vertex[i]] = 1;
    existing_vertices[h_edge_dst_vertex[i]] = 1;
  }

  if (argc == 1) {
    startVertex = std::max_element(h_out_edges.begin(), h_out_edges.end()) - h_out_edges.begin();
  }
  
  if(h_out_edges[startVertex] == 0)
  {
    printf("Start vertex has 0 out edges!\n");
    exit(1);
  }
  //startVertex = 0;
  std::cout << "Starting at " << startVertex << " of " << numVertices << std::endl;

  //one vertex starts active in sssp
  d_active_vertex_flags[0].resize(numVertices, 0);
  d_active_vertex_flags[1].resize(numVertices, 0);

  //run test
  thrust::device_vector<sssp::VertexType> d_vertex_data(numVertices); //each vertex value starts at "infinity"
  run_test(d_edge_dst_vertex,
           d_edge_src_vertex,
           d_vertex_data,
           d_edge_data,
           d_active_vertex_flags, startVertex, numEdges);

  //  d_active_vertex_flags[0][startVertex] = 1;
  //  d_vertex_data[startVertex] = sssp::VertexType(0, true);
  //
  //  std::vector<int> ret(2);
  //  GASEngine<sssp, sssp::VertexType, int, int, int> engine;
  //
  //  double startTime = omp_get_wtime();
  //
  //  ret = engine.run(d_edge_dst_vertex,
  //                   d_edge_src_vertex,
  //                   d_vertex_data,
  //                   d_edge_data,
  //                   d_active_vertex_flags, INT_MAX);
  //
  //#ifdef GPU_DEVICE_NUMBER
  //  cudaDeviceSynchronize();
  //#endif
  //  double elapsed = (omp_get_wtime() - startTime)*1000;
  //
  //  int diameter = ret[0];
  //  std::cout << "Took: " << elapsed << " ms" << std::endl;
  //  std::cout << "Iterations to convergence: " << diameter << std::endl;

  if (outFileName) {
    FILE* f = fopen(outFileName, "w");
    std::vector<sssp::VertexType> h_vertex_data(d_vertex_data.size());
    thrust::copy(d_vertex_data.begin(), d_vertex_data.end(), h_vertex_data.begin());

    for (int i = 0; i < numVertices; ++i) {
      fprintf(f, "%d\t%d\n", i, h_vertex_data[i].dist);
    }

    fclose(f);
  }

  return 0;
}
