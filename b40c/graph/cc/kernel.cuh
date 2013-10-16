#pragma once

namespace b40c {
namespace graph {
namespace cc {


template <typename KernelPolicy>
struct Dispatch
{
    typedef typename KernelPolicy::VertexId                         VertexId;
    typedef typename KernelPolicy::ProblemType                      ProblemType;
    
/**
 *  
 *  Function to speedup the selection process in the first iteration
 *  The ancestor tree is initialized to the add the edge from larger
 *  edge to its smaller neighbour in this method.
 *  
 *  The process is random and each edge performs this task independently.
 *  select_winner_init
 *   
 */
static __device__ void SelectWinnerInit(
                    VertexId      *d_parent,
                    VertexId      *d_edge_list,
                    int           num_edges)
{
    int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	VertexId  t;
	unsigned int from, to, max_node, min_node;
	if (idx < num_edges) {
		t = d_edge_list[idx];
		from = t>>ProblemType::FROM_VERTEX_OFFSET;
		to = t&ProblemType::TO_VERTEX_ID_MASK;
		max_node = from > to ? from : to;
		min_node = from + to - max_node;
		d_parent[max_node] = min_node;
	}
	return;
}

/**
 * Function to hook from higher valued tree to lower valued tree.
 * For details, read the PPL Paper or LSPP paper or my master's thesis.
 * Following greener's algorithm, there are two iterations, one from
 * lower valued edges to higher values edges and the second iteration
 * goes vice versa. The performance of this is largely related to the input.
 */
static __device__ void SelectWinnerMin(
                    bool          *d_marks,
                    VertexId      *d_parent,
                    VertexId      *d_edge_list,
                    int           num_edges,
                    int           *flag)
{
 	int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	VertexId  t;
	unsigned int from, to, parent_from, parent_to, max_node, min_node;
	__shared__ int s_flag;
	if (threadIdx.x == 0)
		s_flag = 0;
	__syncthreads();
	if (idx < num_edges) {
		t = d_edge_list[idx];
		if (!d_marks[idx]) {
			from = t>>ProblemType::FROM_VERTEX_OFFSET;
			to = t&ProblemType::TO_VERTEX_ID_MASK;
			parent_from = d_parent[from];
			parent_to = d_parent[to];
			max_node = parent_from > parent_to ? parent_from : parent_to;
			min_node = parent_from + parent_to - max_node;
			if (max_node == min_node)
				d_marks[idx] = true;
			else {
				d_parent[min_node] = max_node;
				s_flag = 1;
			}
		}
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		if (s_flag == 1) {
			*flag = 1;
		}
	}
	return;
}


/**
 * Function to hook from lower valued tree to higher valued tree.
 */
static __device__ void SelectWinnerMax(
                    bool          *d_marks,
                    VertexId      *d_parent,
                    VertexId      *d_edge_list,
                    int           num_edges,
                    int           *flag)
{
   	int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	VertexId  t;
	unsigned int from, to, parent_from, parent_to, max_node, min_node;
	__shared__ int s_flag;
	if (threadIdx.x == 0)
		s_flag = 0;
	__syncthreads();
	if (idx < num_edges) {
		t = d_edge_list[idx];
		if (!d_marks[idx]) {
			from = t>>ProblemType::FROM_VERTEX_OFFSET;
			to = t&ProblemType::TO_VERTEX_ID_MASK;
			
			parent_from = d_parent[from];
			parent_to = d_parent[to];
			max_node = parent_from > parent_to ? parent_from : parent_to;
			min_node = parent_from + parent_to - max_node;
			if (max_node == min_node)
				d_marks[idx] = true;
			else {
				d_parent[max_node] = min_node;
				s_flag = 1;
			}
		}
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		if (s_flag == 1) {
			*flag = 1;
		}
	}
	return;
}

static __device__ void PointerJumping(
                VertexId      *d_parent,
                int           num_nodes,
                int           *flag)
{
	int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	__shared__ int s_flag;
	if (idx >= num_nodes)
		return;
	if (threadIdx.x == 0)
		s_flag = 0;
	__syncthreads();

    VertexId parent, grand_parent;
	if (idx < num_nodes) {
		parent = d_parent[idx];
		grand_parent = d_parent[parent];
		if (parent != grand_parent) {
			s_flag = 1;
			d_parent[idx] = grand_parent;
		}
	}
    __syncthreads();
	if (threadIdx.x == 0) {
		if (s_flag == 1) {
			*flag = 1;
		}
	}
}

static __device__ void PointerJumpingMasked(
                char          *d_masks,
                VertexId      *d_parent,
                int           num_nodes,
                int           *flag)
{
    int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	__shared__ int s_flag;
	if (idx >= num_nodes)
		return;
	if (threadIdx.x == 0)
	{
		s_flag = 0;
	}
	__syncthreads();

    VertexId parent, grand_parent;
	if (d_masks[idx]==0) {
		parent = d_parent[idx];
		grand_parent = d_parent[parent];
		if (parent != grand_parent) {
			s_flag = 1;
			d_parent[idx] = grand_parent;
		}
		else {
			d_masks[idx] = -1;
		}
	}

	if (threadIdx.x == 0) {
		if (s_flag == 1) {
			*flag = 1;
		}
	}
}
 
static __device__ void PointerJumpingUnmasked(
                char          *d_masks,
                VertexId      *d_parent,
                int           num_nodes)
{
  
	int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	if (idx >= num_nodes)
		return;
	__syncthreads();

    VertexId parent, grand_parent;
	if (d_masks[idx]==1) {
		parent = d_parent[idx];
		grand_parent = d_parent[parent];
		d_parent[idx] = grand_parent;
		}
}


static __device__ void UpdateParent(
                VertexId      *d_parent,
                int           num_nodes)
{
 
	int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	if (idx >= num_nodes)
		return;
	d_parent[idx] = (VertexId)idx;
	return;
}

static __device__ void UpdateMark(
                bool          *d_marks,
                int           num_edges)
{
   
	int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	if (idx >= num_edges)
		return;
	d_marks[idx] = false;
	return;
}

static __device__ void UpdateMask(
                char          *d_masks,
                VertexId      *d_parents,
                int           num_nodes)
{
  
	int idx = blockIdx.x*KernelPolicy::THREADS+threadIdx.x;
	if (idx >= num_nodes)
		return;
	d_masks[idx] = (d_parents[idx] == idx)?0:1;
	return;
}

}; //struct Dispatch

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void SelectWinnerInit(
    typename KernelPolicy::VertexId     *d_parent,
    typename KernelPolicy::VertexId     *d_edge_list,
    int                                 num_edges)
{
	Dispatch<KernelPolicy>::SelectWinnerInit(
	    d_parent,
	    d_edge_list,
	    num_edges);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void SelectWinnerMin(
    bool                                *d_marks,
    typename KernelPolicy::VertexId     *d_parent,
    typename KernelPolicy::VertexId     *d_edge_list,
    int                                 num_edges,
    int                                 *flag)
{
	Dispatch<KernelPolicy>::SelectWinnerMin(
	    d_marks,
	    d_parent,
	    d_edge_list,
	    num_edges,
	    flag);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void SelectWinnerMax(
    bool                                *d_marks,
    typename KernelPolicy::VertexId     *d_parent,
    typename KernelPolicy::VertexId     *d_edge_list,
    int                                 num_edges,
    int                                 *flag)
{
	Dispatch<KernelPolicy>::SelectWinnerMax(
	    d_marks,
	    d_parent,
	    d_edge_list,
	    num_edges,
	    flag);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void PointerJumping(
    typename KernelPolicy::VertexId     *d_parent,
    int                                 num_nodes,
    int                                 *flag)
{
	Dispatch<KernelPolicy>::PointerJumping(
	    d_parent,
	    num_nodes,
	    flag);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void PointerJumpingMasked(
    char                                *d_masks,
    typename KernelPolicy::VertexId     *d_parent,
    int                                 num_nodes,
    int                                 *flag)
{
	Dispatch<KernelPolicy>::PointerJumpingMasked(
	    d_masks,
	    d_parent,
	    num_nodes,
	    flag);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void PointerJumpingUnmasked(
    char                                *d_masks,
    typename KernelPolicy::VertexId     *d_parent,
    int                                 num_nodes)
{
	Dispatch<KernelPolicy>::PointerJumpingUnmasked(
	    d_masks,
	    d_parent,
	    num_nodes);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void UpdateParent(
    typename KernelPolicy::VertexId     *d_parent,
    int                                 num_nodes)
{
	Dispatch<KernelPolicy>::UpdateParent(
	    d_parent,
	    num_nodes);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void UpdateMark(
    bool                                *d_marks,
    int                                 num_edges)
{
	Dispatch<KernelPolicy>::UpdateMark(
	    d_marks,
	    num_edges);
}

template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void UpdateMask(
    char                                *d_masks,
    typename KernelPolicy::VertexId     *d_parent,
    int                                 num_nodes)
{
	Dispatch<KernelPolicy>::UpdateMask(
	    d_masks,
	    d_parent,
	    num_nodes);
}
} // namespace cc
} // namespace graph
} // namespace b40c
