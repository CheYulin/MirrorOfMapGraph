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
#include "sampleBC.h"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "graphio.h"
#include <cstdlib>

struct final_changed_transform : thrust::unary_function<thrust::tuple<adaptiveBC::VertexType&, int&>, int>
  {
    __device__
    int operator()(const thrust::tuple<adaptiveBC::VertexType&, int&> &t)
    {
      const adaptiveBC::VertexType& v = thrust::get < 0 > (t);
      const int& flag = thrust::get < 1 > (t);
      if(v.changed && flag) return 1;
      else return 0;
    }
  } ;
  
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

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  for(device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
  }

  int numVertices;
  const char* outFileName = 0;
  double samplerate = 1.0;
  int issym;

  //generate simple random graph
  std::vector<int> h_edge_src_vertex;
  std::vector<int> h_edge_dst_vertex;
  std::vector<int> h_edge_data;

  if (argc == 2) {
    numVertices = 1000000;
    const int avgEdgesPerVertex = 10;
    generateRandomGraph(h_edge_src_vertex, h_edge_dst_vertex, numVertices, avgEdgesPerVertex);
    h_edge_data.reserve(h_edge_src_vertex.size());
    for (int i = 0; i < h_edge_src_vertex.size(); ++i) {
      h_edge_data.push_back(rand() % 100);
    }
    samplerate = atof(argv[1]);
  }
  else if (argc == 3 || argc == 4) {
    samplerate = atof(argv[1]);
    
    //TODO: now the loadGraph has problem parsing the matrix size information
    issym = loadGraph( argv[2], numVertices, h_edge_src_vertex, h_edge_dst_vertex, &h_edge_data );
    if (argc == 4)
      outFileName = argv[3];
  }
  else {
    std::cerr << "Wrong arguments!\n Usage: sampleBC sample-rate graph(.mtx) outputfilename(optional)" << std::endl;
    exit(1);
  }
  
  const uint numEdges = h_edge_src_vertex.size();
  printf("Graph number of vertices is: %d, number is edges is %d\n", numVertices, numEdges);
  
  h_edge_data = std::vector<int>(numEdges, 1);//enforce unweight graph
  
  if(issym == 0)
  {
    printf("This is a directed graph\n");
  }
  else
  {
    printf("This is a undirected graph\n");
  }

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
  
  thrust::device_vector<int> changed_array(numVertices);
  std::vector<int> ret(2, 1);
  std::vector<int> ret2(2, 1);
  
  srand (time(NULL));
  int num_srcs = (int)ceil( (double)numVertices * samplerate);
  printf("Number of random sources: %d\n", num_srcs);
  
   std::vector<bool> rand_flag(numVertices, false);
  std::vector<int> rand_numbers;
  int num_rand = 0;
  while(num_rand < num_srcs)
  {
       int randnum = rand() % numVertices;
      if(!rand_flag[randnum])
      {
          rand_numbers.push_back(randnum);
          rand_flag[randnum] = true;
          num_rand++;
      }
      
  }
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  for(int seed = 0; seed < num_srcs; seed++)
  {
    startVertex = rand_numbers[seed];
    printf("Seed vertex: %d\n", startVertex);
    
    thrust::fill(d_active_vertex_flags[0].begin(), d_active_vertex_flags[0].end(), 0);
    thrust::fill(d_active_vertex_flags[1].begin(), d_active_vertex_flags[1].end(), 0);
    thrust::fill(d_vertex_data.begin(), d_vertex_data.end(), adaptiveBC::VertexType(10000000, 0, false, 0.0));
    
    d_active_vertex_flags[0][startVertex] = 1;
    d_vertex_data[startVertex] = adaptiveBC::VertexType(0, 1, true, 0.0);
    ret = forward_engine.run(d_edge_dst_vertex,
                            d_edge_src_vertex,
                            d_vertex_data,
                            d_edge_data,
                            d_active_vertex_flags, 1000000000);
    
    int selector = ret[1];
    thrust::fill_n(changed_array.begin(), numVertices, 0);
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_vertex_data.begin(),d_active_vertex_flags[selector^1].begin())), 
                      thrust::make_zip_iterator(thrust::make_tuple(d_vertex_data.end(),d_active_vertex_flags[selector^1].end())), 
                      changed_array.begin(), final_changed_transform());
    int num_changed = reduce(changed_array.begin(), changed_array.end());
    int diameter = ret[0] - ((num_changed == 0) ? 1 : 0);

  //reinit active
    reinit_active_flags(d_vertex_data, d_active_vertex_flags[0], diameter);
    thrust::fill(d_active_vertex_flags[1].begin(), d_active_vertex_flags[1].end(), 0);

    ret2 = backward_engine.run(d_edge_dst_vertex2,
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
      fprintf( f, "%d\t%d\t%d\t%f\n", i, h_vertex_data[i].dist, h_vertex_data[i].sigma, h_bc[i]);
    }

    fclose(f);
  }
  return 0;
}
