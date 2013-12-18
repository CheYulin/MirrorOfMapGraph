/*
 * bfs.h
 *
 *  Created on: Dec 2, 2013
 *      Author: zhisong
 */

#ifndef BFS_H_
#define BFS_H_

#include <b40c/graph/GASengine/csr_problem.cuh>

//TODO: edge data not currently represented
//TODO: initialize frontier
struct bfs {
	static const int INIT_VALUE = 100000000;
	typedef int GatherType;
	struct VertexType {
		int* d_labels;
		int nn;
		int ne;

		VertexType () : d_labels(NULL), nn(0), ne(0){}

		void init(int nodes, int edges)
		{
			nn = nodes;
			ne = edges;
			b40c::util::B40CPerror(cudaMalloc((void**) &d_labels, nodes * sizeof(int)), "cudaMalloc VertexType::d_labels failed", __FILE__, __LINE__);

			int memset_block_size = 256;
			int memset_grid_size_max = 32 * 1024;	// 32K CTAs
			int memset_grid_size;

			// Initialize d_labels elements to -1
			memset_grid_size = B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);
			b40c::util::MemsetKernel<int><<<memset_grid_size, memset_block_size, 0, 0>>>(
				d_labels,
				-1,
				nodes);
		}
	  };

	  static GatherEdges gatherOverEdges() {
		return NO_GATHER_EDGES;
	  }

	  struct contract {
		  __device__
		  void operator()(int &row_id, VertexType &vertex_list, int& iteration, int &vertex_id)
		  {
			// Load label of node
			int label;
			label = vertex_list.d_labels[row_id];

			if (label != -1) {

				// Seen it
				vertex_id = -1;

			} else {

				// Update label with current iteration
				vertex_list.d_labels[row_id] = iteration;
			}
		  }
	  };

	  /**
	  	 * the binary operator
	  	 */
	  	  struct gather_sum {
	  		__device__
	  		GatherType operator()(GatherType left, GatherType right) {
	              return left + right;
	  		  }
	  	  };

	  	/**
	   * For each vertex in the frontier,
	   */
		  struct gather_vertex {
			__device__
			  void operator()(int row_id, GatherType final_value, VertexType &vertex_list) {

			  }
		  };

	  struct expand_vertex {
	  		  __device__
	  		  bool operator()(int &row_id, VertexType &vertex_list)
	  		  {
	  			  return true;
	  		  }
	  	  };

	  struct expand_edge {
		  __device__
		  void operator()(bool &changed, int &row_id, VertexType &vertex_list, int& neighbor_id_in, int&neighbor_id_out, int& frontier, int& predecessor_out)
		  {
			  neighbor_id_out = neighbor_id_in;
			  predecessor_out = row_id;
			  frontier = neighbor_id_out;
		  }
	  };

	  struct gather_edge {
		__device__
		  void operator()(int row_id, int neighbor_id_in, VertexType &vertex_list,  int& new_value) {

		  }
	  };

	  struct sum {
		__device__
		  int operator()(int left, int right) {
			return min(left, right);
		  }
	  };

	  struct reset {
		__device__
		  void operator()(VertexType& vertex_list, int v) {
		  }
	  };

	  struct apply {
		__device__
		  void operator()(VertexType& vertex_list, int iteration, int v) {
		  }
	  };

	  static ApplyVertices applyOverEdges() {
		return NO_APPLY_VERTICES;
	  }

	  static ExpandEdges expandOverEdges() {
		return EXPAND_OUT_EDGES;
	  }


};

#endif /* BFS_H_ */
