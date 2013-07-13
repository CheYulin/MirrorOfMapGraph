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

#include <cstdio>
#include <cmath>
#include "GASEngine.h"
#include "pagerank.h"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "graphio.h"
#include <algorithm>
#include <omp.h>
using namespace std;

void generateRandomGraph(std::vector<int> &h_edge_src_vertex,
                         std::vector<int> &h_edge_dst_vertex,
                         int numVertices, int avgEdgesPerVertex) {
  thrust::minstd_rand rng;
  thrust::random::normal_distribution<float> n_dist(avgEdgesPerVertex, sqrtf(avgEdgesPerVertex));
  thrust::uniform_int_distribution<int> u_dist(0, numVertices - 1);

  for (int v = 0; v < numVertices; ++v) {
    int numEdges = min(std::max((int)roundf(n_dist(rng)), 1), 1000);
    for (int e = 0; e < numEdges; ++e) {
      uint dst_v = u_dist(rng);
      h_edge_src_vertex.push_back(v);
      h_edge_dst_vertex.push_back(dst_v);
    }
  }
}

int main(int argc, char **argv) {

  int numVertices = 50000;
  const int avgEdgesPerVertex = 10;

#ifdef GPU_DEVICE_NUMBER
  cudaSetDevice(GPU_DEVICE_NUMBER);
  cerr << "Running on device " << GPU_DEVICE_NUMBER << endl;
#endif

  const char* outFileName = 0;

  //generate simple random graph
  std::vector<int> h_edge_src_vertex;
  std::vector<int> h_edge_dst_vertex;

  if (argc == 1)
    generateRandomGraph(h_edge_src_vertex, h_edge_dst_vertex, numVertices, avgEdgesPerVertex);
  else if (argc == 2 || argc == 3)
  {
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

  //need to figure out how many out edges per vertex
  //everytime a vertex is a source, that is an outbound edge
  std::vector<int> h_num_out_edges(numVertices, 0);
  std::vector<char> existing_vertices(numVertices, 0);
  {
    for(int e = 0; e < h_edge_src_vertex.size(); ++e) {
      h_num_out_edges[h_edge_src_vertex[e]]++;
      existing_vertices[h_edge_src_vertex[e]] = 1;
      existing_vertices[h_edge_dst_vertex[e]] = 1;
    }
  }

  //use PSW ordering
  thrust::sort_by_key(d_edge_dst_vertex.begin(), d_edge_dst_vertex.end(), d_edge_src_vertex.begin());

  //generate random starting values
  std::vector<pagerank::VertexType> h_vertex_vals(numVertices);
  
  {
    thrust::minstd_rand rng;
    thrust::uniform_real_distribution<float> u_dist(0.f, 1.f);
    for (int i = 0; i < numVertices; ++i) {
      h_vertex_vals[i].val = .15f;
      h_vertex_vals[i].num_out_edges = h_num_out_edges[i];
    }
  }

  thrust::device_vector<pagerank::VertexType> d_vertex_vals = h_vertex_vals;

  std::vector<thrust::device_vector<int> > d_active_vertex_flags;
  {
    thrust::device_vector<int> foo;
    d_active_vertex_flags.push_back(foo);
    d_active_vertex_flags.push_back(foo);
  }

  //all vertices start active in pagerank
  d_active_vertex_flags[0].resize(numVertices, 1);
  d_active_vertex_flags[1].resize(numVertices, 1);

  GASEngine<pagerank, pagerank::VertexType, int, float, float> engine;

  double startTime = omp_get_wtime();

  int diameter = engine.run(d_edge_dst_vertex,
                            d_edge_src_vertex,
                            d_vertex_vals,
                            d_active_vertex_flags);

#ifdef GPU_DEVICE_NUMBER
  cudaDeviceSynchronize();
#endif

  double elapsed = omp_get_wtime()-startTime;
  std::cout << "Took: " << elapsed << " ms" << std::endl;
  std::cout << "Number iterations to convergence: " << diameter << std::endl;

  if( outFileName )
  {
    FILE* f = fopen( outFileName, "w" );
    thrust::copy( d_vertex_vals.begin(), d_vertex_vals.end(), h_vertex_vals.begin() );
    for( int i = 0; i < existing_vertices.size(); ++i) 
    {
      if (!existing_vertices[i])
        continue;
      fprintf( f, "%d %f\n", i, h_vertex_vals[i].val );
    }
    fclose(f);
  }
  
  return 0;
}
