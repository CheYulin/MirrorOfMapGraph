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
#include "bfs.h"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "graphio.h"
#include <iostream>

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

  if (argc == 1) {
    numVertices = 8000;
    const int avgEdgesPerVertex = 10;
    generateRandomGraph(h_edge_src_vertex, h_edge_dst_vertex, numVertices, avgEdgesPerVertex);
  }
  else if (argc == 2 || argc == 3) {
    loadGraph( argv[1], numVertices, h_edge_src_vertex, h_edge_dst_vertex );
    if (argc == 3)
      outFileName = argv[2];
  }
  else {
    std::cerr << "Too many arguments!" << std::endl;
    exit(1);
  }

  const uint numEdges = h_edge_src_vertex.size();

  thrust::device_vector<int> d_edge_src_vertex = h_edge_src_vertex;
  thrust::device_vector<int> d_edge_dst_vertex = h_edge_dst_vertex;

  //use PSW ordering
  //thrust::sort_by_key(d_edge_dst_vertex.begin(), d_edge_dst_vertex.end(), d_edge_src_vertex.begin());
  thrust::sort_by_key(d_edge_src_vertex.begin(), d_edge_src_vertex.end(), d_edge_dst_vertex.begin());

  thrust::device_vector<int> d_vertex_vals(numVertices, -1);

  std::vector<thrust::device_vector<int> > d_active_vertex_flags;
  {
    thrust::device_vector<int> foo;
    d_active_vertex_flags.push_back(foo);
    d_active_vertex_flags.push_back(foo);
  }
  d_active_vertex_flags[0].resize(numVertices, 0);
  d_active_vertex_flags[1].resize(numVertices, 0);

  //set starting node for bfs
  int startVertex = 0;
  std::vector<char> existing_vertices(numVertices, 0);
  {
    std::vector<int> h_out_edges(numVertices);
    for(int e = 0; e < h_edge_src_vertex.size(); ++e) {
      h_out_edges[h_edge_src_vertex[e]]++;
      existing_vertices[h_edge_src_vertex[e]] = 1;
      existing_vertices[h_edge_dst_vertex[e]] = 1;
    }
    startVertex = std::max_element(h_out_edges.begin(), h_out_edges.end()) - h_out_edges.begin();
  }

  d_vertex_vals[startVertex] = 0;
  d_active_vertex_flags[0][startVertex] = 1;

  GASEngine<bfs, int, int, int, int> engine;

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  cudaEventRecord(start);

  int diameter = engine.run(d_edge_dst_vertex,
                            d_edge_src_vertex,
                            d_vertex_vals,
                            d_active_vertex_flags);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  std::cout << "Took: " << elapsed << " ms" << std::endl;
  std::cout << "Graph Diameter: " << diameter << std::endl;
  std::cout << "M-Edges / sec: " << numEdges / (elapsed * 1000.f) << std::endl;

  if( outFileName )
  {
    FILE* f = fopen( outFileName, "w" );
    thrust::host_vector<int> h_vertex_vals(numVertices);
    thrust::copy( d_vertex_vals.begin(), d_vertex_vals.end(), h_vertex_vals.begin() );
    for( int i = 0; i < existing_vertices.size(); ++i) {
      if (!existing_vertices[i])
        continue;
      fprintf( f, "%d\t%d\n", i, h_vertex_vals[i] );
    }
    fclose(f);
  }
  
  return 0;
}
