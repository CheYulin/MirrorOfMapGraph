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

#ifndef BFS_H__
#define BFS_H__

//edge data not currently represented
struct bfs {
  static GatherEdges gatherOverEdges() {
    return NO_GATHER_EDGES;
  }

  struct gather {
    __device__
      int operator()(const int dst, const int src, const int e) {
        return src;
      }
  };

  struct sum {
    __device__
      int operator()(int left, int right) {
        return max(left, right);
      }
  };

  struct apply {
    __device__
      void operator()(int &cur_depth, const int parent_depth = 0) {
        cur_depth = d_iterations; //parent_depth + 1;
      }
  };

  static ScatterEdges scatterOverEdges() {
    return SCATTER_OUT_EDGES;
  }

  struct scatter {
    __host__ __device__
      int operator()(const int dst, const int src, const int e) {
        if (dst == -1)
          return 1;
        else
          return 0;
      }
  };

};

#endif
