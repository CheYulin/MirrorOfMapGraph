/******************************************************************************
 * GPU COO storage management structure for CC problem data
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>

#include <b40c/graph/cc/problem_type.cuh>


namespace b40c {
namespace graph {
namespace cc {


/**
 * CSR storage management structure for BFS problems.  
 */
template <
	typename _VertexId>
struct CooProblem
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef ProblemType<
		_VertexId>				// VertexId
			ProblemType;

	typedef typename ProblemType::VertexId 			            VertexId;


	//---------------------------------------------------------------------
	// Helper structures
	//---------------------------------------------------------------------

	/**
	 * Graph slice on GPU
	 */
	struct GraphSlice
	{
		//Device index
		int             gpu_index;

		// Standard COO device storage arrays
		VertexId 		*d_edge_tuples;         //edge tuples 63-62bit:edge mark, 61-31bit:from vertex id, 30-0bit:to vertex id
		VertexId        *d_node_parents;        //node's parent array. 32bit:node mask, 31-0bit:node parent vertex id
		char            *d_masks;               //mask to show if a tree is stagnant
		bool            *d_marks;               //mark to show if an edge's two vertices belong to different ccs 

		// Number of nodes and edges in slice
		VertexId		nodes;
		VertexId		edges;

		// CUDA stream to use for processing this slice
		cudaStream_t 	stream;

		/**
		 * Constructor
		 */
		GraphSlice(cudaStream_t stream) :
			d_edge_tuples(NULL),
			d_node_parents(NULL),
			d_masks(NULL),
			d_marks(NULL),
			nodes(0),
			edges(0),
			gpu_index(0),
			stream(stream)
		{
		}

		/**
		 * Destructor
		 */
		virtual ~GraphSlice()
		{
			// Set device
			util::B40CPerror(cudaSetDevice(gpu_index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

			// Free pointers
			if (d_edge_tuples) 				util::B40CPerror(cudaFree(d_edge_tuples), "GpuSlice cudaFree d_edge_tuples failed", __FILE__, __LINE__);
			if (d_node_parents) 			util::B40CPerror(cudaFree(d_node_parents), "GpuSlice cudaFree d_node_parents failed", __FILE__, __LINE__);
			if (d_masks) 			        util::B40CPerror(cudaFree(d_masks), "GpuSlice cudaFree d_masks failed", __FILE__, __LINE__);
			if (d_marks) 			        util::B40CPerror(cudaFree(d_marks), "GpuSlice cudaFree d_marks failed", __FILE__, __LINE__);
            // Destroy stream
			if (stream) {
				util::B40CPerror(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
			}
		}
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

    //GPU index
    int                             gpu_index;

	// Size of the graph
	unsigned int 					nodes;
	unsigned int					edges;
	unsigned int                    num_components;
	

	// Set of graph slices (one for each GPU)
	GraphSlice* 	graph_slice;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	CooProblem() :
	    gpu_index(0),
		nodes(0),
		edges(0),
		num_components(0)
	{}


	/**
	 * Destructor
	 */
	virtual ~CooProblem()
	{
		// Cleanup graph slice on the heap
		delete graph_slice;
	}


	/**
	 * Extract into a single host vector the CC results
	 */
	cudaError_t ExtractResults(VertexId *h_component_label, VertexId *h_edge_list, bool *h_masks, VertexId *h_cc_bins)
	{
		cudaError_t retval = cudaSuccess;

		do {

				// Set device
				if (util::B40CPerror(cudaSetDevice(gpu_index),
					"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;;

				if (retval = util::B40CPerror(cudaMemcpy(
						h_component_label,
						graph_slice->d_node_parents,
						sizeof(VertexId) * graph_slice->nodes,
						cudaMemcpyDeviceToHost),
					"CsrProblem cudaMemcpy d_node_parents failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMemcpy(
						h_edge_list,
						graph_slice->d_edge_tuples,
						sizeof(VertexId) * graph_slice->edges,
						cudaMemcpyDeviceToHost),
					"CsrProblem cudaMemcpy d_node_parents failed", __FILE__, __LINE__)) break;
                
                if (retval = util::B40CPerror(cudaMemcpy(
						h_masks,
						graph_slice->d_masks,
						sizeof(char) * graph_slice->nodes,
						cudaMemcpyDeviceToHost),
					"CsrProblem cudaMemcpy d_masks failed", __FILE__, __LINE__)) break;
                
                num_components = 0;
				for (int i = 0; i < nodes; ++i)
				{
					if (h_component_label[i] == i)
                    {
                        h_cc_bins[num_components] = i;
						num_components++;
                    }
				}

                if (retval) break;

			} while(0);

		return retval;
	}

 //TODO: compute size of each connected component
 //sort them and output the top ten connected component size
 //output in file the following format:
 //connected component 1: nodei, nodei+1, ..., nodek
 //connected component 2: nodej, nodej+1, ..., nodem
 // h_cc_bins needs to be added into ExtractResult function too
   void CountComponentSize(VertexId *h_cc_labels, unsigned int num_components,
                           unsigned int *h_cc_histograms, VertexId *h_cc_bins)
   {
        for (int i = 0; i < num_components; ++i)
        {
            h_cc_histograms[i] = 0;
        }
        for (int i = 0; i < nodes; ++i)
        {
            for (int j = 0; j < num_components; ++j)
            {
                if (h_cc_labels[i] == h_cc_bins[j])
                {
                    h_cc_histograms[j]++;
                    break;
                }
            }
        }
   }



	/**
	 * Initialize from host CSR problem
	 */
	cudaError_t FromHostProblem(
		int 		    nodes,
		int 		    edges,
		long long int 	*h_from_vertex_ids,
		long long int 	*h_to_vertex_ids,
        long long int   *h_node_parents,
		int 		    gpu_index)
	{
		cudaError_t retval 			= cudaSuccess;
		this->nodes					= nodes;
		this->edges 				= edges;
		this->gpu_index 			= gpu_index;

		do {
				// Create a single GPU slice for the currently-set gpu
				int gpu;
				if (retval = util::B40CPerror(cudaGetDevice(&gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
				graph_slice = new GraphSlice(0);
				graph_slice->nodes = nodes;
				graph_slice->edges = edges;

				VertexId *edge_queue = new VertexId[edges];

				for (int i = 0; i < edges; ++i)
				{
					edge_queue[i]  = 0;
					edge_queue[i] += h_to_vertex_ids[i]&((1<<33)-1);
					edge_queue[i] += ((h_from_vertex_ids[i]&((1<<33)-1))<<32);                    
					//leave mark 0
				}

				// Allocate and initialize d_edge_tuples
				printf("GPU %d edge_tuples: %lld elements (%lld bytes)\n",
					graph_slice->gpu_index,
					(unsigned long long) (graph_slice->edges),
					(unsigned long long) (graph_slice->edges * sizeof(VertexId)));

				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slice->d_edge_tuples,
						graph_slice->edges * sizeof(VertexId)),
					"CooProblem cudaMalloc d_edge_tuples failed", __FILE__, __LINE__)) break;

				if (retval = util::B40CPerror(cudaMemcpy(
						graph_slice->d_edge_tuples,
						edge_queue,
						graph_slice->edges * sizeof(VertexId),
						cudaMemcpyHostToDevice),
					"CooProblem cudaMemcpy d_edge_tuples failed", __FILE__, __LINE__)) break;
				
				// Allocate and initialize d_node_parents
				printf("GPU %d node_parents: %lld elements (%lld bytes)\n",
					graph_slice->gpu_index,
					(unsigned long long) (graph_slice->nodes),
					(unsigned long long) (graph_slice->nodes) * sizeof(VertexId));

				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slice->d_node_parents,
						(graph_slice->nodes) * sizeof(VertexId)),
					"CooProblem cudaMalloc d_node_parents failed", __FILE__, __LINE__)) break;

				if (retval = util::B40CPerror(cudaMemcpy(
						graph_slice->d_node_parents,
						h_node_parents,
						(graph_slice->nodes) * sizeof(VertexId),
						cudaMemcpyHostToDevice),
					"CsrProblem cudaMemcpy d_node_parents failed", __FILE__, __LINE__)) break;

			    if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slice->d_masks,
						(graph_slice->nodes) * sizeof(char)),
					"CooProblem cudaMalloc d_masks failed", __FILE__, __LINE__)) break;

                if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slice->d_marks,
						(graph_slice->edges) * sizeof(VertexId)),
					"CooProblem cudaMalloc d_marks failed", __FILE__, __LINE__)) break;

				delete[] edge_queue;

		} while (0);

		return retval;
	}
	
};


} // namespace cc
} // namespace graph
} // namespace b40c
