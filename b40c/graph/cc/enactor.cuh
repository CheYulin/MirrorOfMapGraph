#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/cc/problem_type.cuh>
#include <b40c/graph/cc/coo_problem.cuh>

#include <b40c/graph/cc/kernel.cuh>
#include <b40c/graph/cc/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace cc {

template <bool INSTRUMENT>
class Enactor
{
	//----------------------------------
	// Members
	//----------------------------------
	
protected:

    //Device properties
    util::CudaProperties cuda_props;

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime kernel_stats;

    unsigned long long total_runtime;       // Total time "worked" by each cta
    unsigned long long total_lifetime;      // Total time elapsed by each cta

    /**
     * Throttle state. Keep a pinned, mapped word that the kernels will
     * signal when done.
     */
    int    *done;
    int             *d_done;

    int    *flag;
    int             *d_flag;

    /**
     * Current iteration (mapped into GPU space so that it can
     * be modified by multi-iteration kernel lauches)
     */
    long long      iteration;

public:
    // Allows display to stdout of kernel running details
    bool DEBUG;
    
    //----------------------------------
	// Methods
	//----------------------------------

protected:

    /**
     * Prepare enactor for search. Must be called prior to each search
     */
    template <typename CooProblem>
    cudaError_t Setup(
        CooProblem &coo_problem,
        int grid_size)
    {
    	typedef typename CooProblem::VertexId       VertexId;

    	cudaError_t retval = cudaSuccess;

    	do {
    		if (!done) {
    			done = new int;
    			if (retval = util::B40CPerror(cudaMalloc(
    				    (void**)&d_done,
    				    sizeof(int)),
    				    "cudaMalloc d_done failed", __FILE__, __LINE__)) break;
			}

            if (!flag) {
                flag = new int;
    			if (retval = util::B40CPerror(cudaMalloc(
    				    (void**)&d_flag,
    				    sizeof(int)),
    				    "cudaMalloc d_flag failed", __FILE__, __LINE__)) break;
            }

			if (retval = kernel_stats.Setup(grid_size)) break;

			// Reset statistics
			iteration           = 0;
			total_runtime      = 0;
			total_lifetime     = 0;
			done[0]             = -1;
			flag[0]             = -1;
		} while (0);

		return retval;
	}


public:

    /**
     * Constructor
     */
    Enactor(bool DEBUG = false) :
        iteration(NULL),
        done(NULL),
        d_done(NULL),
        flag(NULL),
        d_flag(NULL)
    {}

    /**
     * Destructor
     */
    virtual ~Enactor()
    {
		if (done) {
			delete done;
			util::B40CPerror(cudaFree((void *) d_done),
					"Enactor cudaFreeHost done failed", __FILE__, __LINE__);
		}

        if (flag) {
        	delete flag;
			util::B40CPerror(cudaFree((void *) d_flag),
					"Enactor cudaFreeHost done failed", __FILE__, __LINE__);
		}
	}

	/**
	 * Obtain statistics about the last CC enacted
	 */
	template <typename VertexId>
	void GetStatistics(
	    VertexId total_iteration,
	    double &avg_duty)
	{
		cudaThreadSynchronize();

		total_iteration = this->iteration;

		avg_duty = (total_lifetime > 0) ?
		    double(total_runtime) / total_lifetime :
		    0.0;
	}

	/**
	 * Utility function: Returns the default maximum number of threadblocks
	 * this enactor class can launch
	 */
	int MaxGridSize(int cta_occupancy, int max_grid_size = 0)
	{
		if (max_grid_size <= 0) {
			// No override: Fully populate all SMs
			max_grid_size = this->cuda_props.device_props.multiProcessorCount * cta_occupancy;
		}

		return max_grid_size;
	}

	/**
	 * Enacts connected component labeling. Invokes
	 * hooking and ptr_jumping kernels for each iteration.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template<
	    typename KernelPolicy,
	    typename CooProblem>
	cudaError_t EnactCC(
	    CooProblem          &coo_problem,
	    int                 max_grid_size = 0)
	{
		typedef typename CooProblem::VertexId   VertexId;

		cudaError_t retval = cudaSuccess;

		do {
			// Determine grid size
			int kernel_occupancy        = KernelPolicy::CTA_OCCUPANCY;
			int kernel_grid_size        = MaxGridSize(kernel_occupancy, max_grid_size);

			if (DEBUG) {
				printf("CC kernel occupancy %d, level-grid size %d\n",
				    kernel_occupancy, kernel_grid_size);
			}

			// Lazy initialization
			if (retval = Setup(coo_problem, kernel_grid_size)) break;

			// Single-gpu graph slice
			typename CooProblem::GraphSlice *graph_slice = coo_problem.graph_slice;

			int n_nodes = graph_slice->nodes;
			int n_edges = graph_slice->edges;
			int nn_block = n_nodes/KernelPolicy::THREADS + ((n_nodes%KernelPolicy::THREADS==0)?0:1);
			int ne_block = n_edges/KernelPolicy::THREADS + ((n_edges%KernelPolicy::THREADS==0)?0:1);

           //Start kernel calls to compute connected component labeling
			UpdateMark<KernelPolicy>
			    <<<ne_block, KernelPolicy::THREADS>>>(
			        graph_slice->d_marks,
			        graph_slice->edges);
			UpdateParent<KernelPolicy>
			    <<<nn_block, KernelPolicy::THREADS>>>(
			        graph_slice->d_node_parents,
			        graph_slice->nodes);
            
            if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "kernel call to UpdateParent/Mark failed ", __FILE__, __LINE__))) break;
			SelectWinnerInit<KernelPolicy>
			    <<<ne_block, KernelPolicy::THREADS>>>(
			        graph_slice->d_node_parents,
			        graph_slice->d_edge_tuples,
			        graph_slice->edges);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "kernel call to SelectWinnerInit failed ", __FILE__, __LINE__))) break;
			do {
				done[0] = 0;
                if (retval = util::B40CPerror(cudaMemcpy(
                	d_done,
                	done,
                	sizeof(int),
                	cudaMemcpyHostToDevice),
                	"cudaMemcpy done failed", __FILE__, __LINE__)) break;

				PointerJumping<KernelPolicy>
				    <<<nn_block, KernelPolicy::THREADS>>>(
				        graph_slice->d_node_parents,
				        graph_slice->nodes,
				        d_done);

                printf("ptr_jump\n");
				
                if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "kernel call to PointerJumping failed ", __FILE__, __LINE__))) break;
                if (retval = util::B40CPerror(cudaMemcpy(
                	done,
                	d_done,
                	sizeof(int),
                	cudaMemcpyDeviceToHost),
                	"cudaMemcpy d_done failed", __FILE__, __LINE__)) break;
            } while (done[0]);

			UpdateMask<KernelPolicy>
			    <<<nn_block, KernelPolicy::THREADS>>>(
			        graph_slice->d_masks,
			        graph_slice->d_node_parents,
			        graph_slice->nodes);
       
            iteration = 1;
			do {
                printf("%d iteration\n", iteration);
                done[0] = 0;
                if (retval = util::B40CPerror(cudaMemcpy(
                	d_done,
                	done,
                	sizeof(int),
                	cudaMemcpyHostToDevice),
                	"cudaMemcpy done failed", __FILE__, __LINE__)) break;

				if (iteration!=0) {
					SelectWinnerMax<KernelPolicy>
					    <<<ne_block, KernelPolicy::THREADS>>>(
					        graph_slice->d_marks,
					        graph_slice->d_node_parents,
					        graph_slice->d_edge_tuples,
					        graph_slice->edges,
					        d_done);
					iteration++;
					iteration=iteration%4;
				}
				else {
                    SelectWinnerMin<KernelPolicy>
					    <<<ne_block, KernelPolicy::THREADS>>>(
					        graph_slice->d_marks,
					        graph_slice->d_node_parents,
					        graph_slice->d_edge_tuples,
					        graph_slice->edges,
					        d_done);
					iteration++;
                    iteration=iteration%4;
				}

                if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "kernel call to SelectWinnerMin/Max failed", __FILE__, __LINE__))) break;
                if (retval = util::B40CPerror(cudaMemcpy(
                	done,
                	d_done,
                	sizeof(int),
                	cudaMemcpyDeviceToHost),
                	"cudaMemcpy d_done failed", __FILE__, __LINE__)) break;
                
				if (done[0] == 0)
					break;

               do {
					flag[0] = 0;
                    if (retval = util::B40CPerror(cudaMemcpy(
                	d_flag,
                	flag,
                	sizeof(int),
                	cudaMemcpyHostToDevice),
                	"cudaMemcpy flag failed", __FILE__, __LINE__)) break;

					PointerJumpingMasked<KernelPolicy>
					    <<<nn_block, KernelPolicy::THREADS>>>(
					        graph_slice->d_masks,
					        graph_slice->d_node_parents,
					        graph_slice->nodes,
					        d_flag);

                    printf("ptr_jump_mask\n");
                    if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "kernel call to PointerJumpingMasked failed ", __FILE__, __LINE__))) break;
                    if (retval = util::B40CPerror(cudaMemcpy(
                	flag,
                	d_flag,
                	sizeof(int),
                	cudaMemcpyDeviceToHost),
                	"cudaMemcpy d_flag failed", __FILE__, __LINE__)) break;

				} while (flag[0]);

				PointerJumpingUnmasked<KernelPolicy>
				    <<<nn_block, KernelPolicy::THREADS>>>(
				        graph_slice->d_masks,
				        graph_slice->d_node_parents,
				        graph_slice->nodes);
                if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "kernel call to PointerJumpingUnmasked failed", __FILE__, __LINE__))) break;
				UpdateMask<KernelPolicy>
				    <<<nn_block, KernelPolicy::THREADS>>>(
				        graph_slice->d_masks,
				        graph_slice->d_node_parents,
				        graph_slice->nodes);
                if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "kernel call to UpdateMask failed", __FILE__, __LINE__))) break;

			} while (done[0]);

		} while(0);

		if (DEBUG) printf("\n");

		return retval;
	}

    /**
	 * Enacts connected component labeling. Invokes
	 * hooking and ptr_jumping kernels for each iteration.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template<typename CooProblem>
	cudaError_t EnactCC(
	    CooProblem          &coo_problem,
	    int                 max_grid_size = 0)
	{
        typedef typename CooProblem::VertexId   VertexId;

        // GF100
        if (this->cuda_props.device_sm_version >= 200) {
        	typedef KernelPolicy<
        	    typename CooProblem::ProblemType,   //ProblemType
        	    200,                                //CUDA_ARCH
        	    INSTRUMENT,                         //INSTRUMENT
        	    8,                                  //MIN_CTA_OCCUPANCY
        	    9>                                  //LOG_THREADS
        	kernelPolicy;

        return EnactCC<kernelPolicy>(
            coo_problem, max_grid_size);
        }

	    printf("Not yet tuned for this architeture\n");
	    return cudaErrorInvalidDeviceFunction;
	}
};

} //namespace cc
} //namespace graph
} //namespace b40c



			
    	

    

    			
	


