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

#ifndef PAGERANK_H__
#define PAGERANK_H__

//the activation flag is now the high bit of the edge field
struct pagerank {
  struct VertexType {
    float val;
    int   num_out_edges;

    __host__ __device__
    VertexType() : val(0.f), num_out_edges(0x80000000) {}
  };

  static GatherEdges gatherOverEdges() {
    return GATHER_IN_EDGES;
  }

  struct gather {
    __host__ __device__
      float operator()(const VertexType &dst, const VertexType &src, const int &e) {
        return src.val / (src.num_out_edges & 0x7FFFFFF);
      }
  };

  struct sum {
    __host__ __device__
      float operator()(float left, float right) {
        return left + right;
      }
  };

  struct apply {
    __host__ __device__
      void operator()(VertexType &cur_val, const float accum = 0.f) {
        float rnew = .15f + (1.f - .15f) * accum;

        if ( fabs(rnew - cur_val.val) < .01f)
          cur_val.num_out_edges &= 0x7FFFFFFF; //guarantee high bit is 0
        else
          cur_val.num_out_edges |= 0x80000000; //guarantee high bit is 1

        cur_val.val = rnew;
      }
  };

  static ScatterEdges scatterOverEdges() {
    return SCATTER_OUT_EDGES;
  }

  struct scatter {
    __host__ __device__
      int operator()(const VertexType &dst, const VertexType &src, const int &e) {
        return (src.num_out_edges & 0x80000000);
      }
  };

};

#endif
