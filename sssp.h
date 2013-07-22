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

#ifndef SSSP_H__
#define SSSP_H__

struct sssp {
  struct VertexType {
    int dist;
    bool changed;

    //the starting value should technically be infinity, but we want to avoid overflow when we add to it
    //we could handle this by doing the addition as a long and then clamping or something to that effect
    //this approach is fine for now
    __host__ __device__
    VertexType() : dist(10000000), changed(true) {}

    __host__ __device__
    VertexType(int dist_, bool changed_) : dist(dist_), changed(changed_) {}
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
        return min(left, right);
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
      int operator()(const VertexType &dst, const VertexType &src, const int &e) {
        return src.changed;
      }
  };

};

#endif
