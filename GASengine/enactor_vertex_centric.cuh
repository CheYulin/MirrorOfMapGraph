#pragma once

#include <stdlib.h>

#include <config.h>

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/global_barrier.cuh>

#include <GASengine/problem_type.cuh>
#include <GASengine/csr_problem.cuh>
#include <GASengine/enactor_base.cuh>

#include <GASengine/vertex_centric/gather/kernel.cuh>
#include <GASengine/vertex_centric/gather/kernel_policy.cuh>
#include <GASengine/vertex_centric/expand_atomic/kernel.cuh>
#include <GASengine/vertex_centric/expand_atomic/kernel_policy.cuh>
#include <GASengine/vertex_centric/contract_atomic/kernel.cuh>
#include <GASengine/vertex_centric/contract_atomic/kernel_policy.cuh>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

using namespace b40c;

using namespace std;

namespace GASengine
{
  template<bool INSTRUMENT>
  class EnactorVertexCentric: public EnactorBase
  {
    //---------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------

  protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime expand_kernel_stats;
    util::KernelRuntimeStatsLifetime contract_kernel_stats;

    unsigned long long total_runtimes; // Total time "worked" by each cta
    unsigned long long total_lifetimes; // Total time elapsed by each cta
    unsigned long long total_queued;

    volatile int *done;
    int *d_done;
    cudaEvent_t throttle_event;
    Config cfg;

    /**
     * Mechanism for implementing software global barriers from within
     * a single grid invocation
     */
    util::GlobalBarrierLifetime global_barrier;

    /**
     * Current iteration (mapped into GPU space so that it can
     * be modified by multi-iteration kernel launches)
     */
    volatile long long *iteration;
    long long *d_iteration;

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

  protected:

    /**
     * Prepare enactor for search.  Must be called prior to each search.
     */
    template<typename CsrProblem>
    cudaError_t Setup(CsrProblem &csr_problem, int expand_grid_size,
        int contract_grid_size, int iter)
    {
      typedef typename CsrProblem::SizeT SizeT;
      typedef typename CsrProblem::VertexId VertexId;
      typedef typename CsrProblem::VisitedMask VisitedMask;

      cudaError_t retval = cudaSuccess;

      do
      {

        // Make sure host-mapped "done" is initialized
        if (!done)
        {
          int flags = cudaHostAllocMapped;

          // Allocate pinned memory for done
          if (retval = util::B40CPerror(
              cudaHostAlloc((void **) &done, sizeof(int) * 1, flags),
              "EnactorContractExpand cudaHostAlloc done failed",
              __FILE__, __LINE__))
            break;

          // Map done into GPU space
          if (retval =
              util::B40CPerror(
                  cudaHostGetDevicePointer((void **) &d_done,
                      (void *) done, 0),
                  "EnactorContractExpand cudaHostGetDevicePointer done failed",
                  __FILE__, __LINE__))
            break;

          // Create throttle event
          if (retval =
              util::B40CPerror(
                  cudaEventCreateWithFlags(&throttle_event,
                      cudaEventDisableTiming),
                  "EnactorContractExpand cudaEventCreateWithFlags throttle_event failed",
                  __FILE__, __LINE__))
            break;
        }

        // Make sure host-mapped "iteration" is initialized
        if (!iteration)
        {

          int flags = cudaHostAllocMapped;

          // Allocate pinned memory
          if (retval = util::B40CPerror(
              cudaHostAlloc((void **) &iteration,
                  sizeof(long long) * 1, flags),
              "EnactorContractExpand cudaHostAlloc iteration failed",
              __FILE__, __LINE__))
            break;

          // Map into GPU space
          if (retval =
              util::B40CPerror(
                  cudaHostGetDevicePointer((void **) &d_iteration,
                      (void *) iteration, 0),
                  "EnactorContractExpand cudaHostGetDevicePointer iteration failed",
                  __FILE__, __LINE__))
            break;
        }

        // Reset statistics
        iteration[0] = iter;
        total_queued = 0;
        done[0] = -1;

        // Single-gpu graph slice
        typename CsrProblem::GraphSlice *graph_slice =
            csr_problem.graph_slices[0];

        // Bind bitmask texture
        int bytes = (graph_slice->nodes + 8 - 1) / 8;
        cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
        if (retval =
            util::B40CPerror(
                cudaBindTexture(0,
                    vertex_centric::contract_atomic::BitmaskTex<
                        VisitedMask>::ref,
                    graph_slice->d_visited_mask, bitmask_desc,
                    bytes),
                "EnactorVertexCentric cudaBindTexture bitmask_tex_ref failed",
                __FILE__, __LINE__))
          break;
      }
      while (0);

      return retval;
    }

  public:

    /**
     * Constructor
     */
    EnactorVertexCentric(Config cfg, bool DEBUG = false) :
        cfg(cfg), EnactorBase(EDGE_FRONTIERS, DEBUG), iteration(NULL), d_iteration(
            NULL), total_queued(0), done(NULL), d_done(NULL)
    {
    }

    /**
     * Destructor
     */
    virtual ~EnactorVertexCentric()
    {
      if (iteration)
      {
        util::B40CPerror(cudaFreeHost((void *) iteration),
            "EnactorVertexCentric cudaFreeHost iteration failed",
            __FILE__, __LINE__);
      }
      if (done)
      {
        util::B40CPerror(cudaFreeHost((void *) done),
            "EnactorVertexCentric cudaFreeHost done failed", __FILE__,
            __LINE__);

        util::B40CPerror(cudaEventDestroy(throttle_event),
            "EnactorVertexCentric cudaEventDestroy throttle_event failed",
            __FILE__, __LINE__);
      }
    }

    template<typename ExpandPolicy, typename GatherPolicy,
        typename ContractPolicy, typename Program, typename CsrProblem>
    cudaError_t EnactIterativeSearch(CsrProblem &csr_problem,
        typename CsrProblem::SizeT* h_row_offsets,
        int directed, int num_srcs, int* srcs, int iter_num)
    {
      typedef typename CsrProblem::SizeT SizeT;
      typedef typename CsrProblem::VertexId VertexId;
      typedef typename CsrProblem::EValue EValue;
      typedef typename CsrProblem::VisitedMask VisitedMask;

      DEBUG = cfg.getParameter<int>("verbose");
      cudaError_t retval = cudaSuccess;

      // Determine grid size(s)
      int expand_occupancy = ExpandPolicy::CTA_OCCUPANCY;
      int expand_grid_size = MaxGridSize(expand_occupancy);

      int gather_occupancy = GatherPolicy::CTA_OCCUPANCY;
      int gather_grid_size = MaxGridSize(gather_occupancy);

      int contract_occupancy = ContractPolicy::CTA_OCCUPANCY;
      int contract_grid_size = MaxGridSize(contract_occupancy);

      typename CsrProblem::GraphSlice *graph_slice =
          csr_problem.graph_slices[0];

      double max_queue_sizing = cfg.getParameter<double>("max_queue_sizing");

      if (retval = csr_problem.Reset(GetFrontierType(), max_queue_sizing))
        return retval;

      Program::Initialize(graph_slice->nodes, graph_slice->edges, num_srcs,
          srcs, graph_slice->d_row_offsets, graph_slice->d_column_indices, graph_slice->d_column_offsets, graph_slice->d_row_indices,
          graph_slice->d_edge_values,
          graph_slice->vertex_list, graph_slice->edge_list,
          graph_slice->frontier_queues.d_keys,
          graph_slice->frontier_queues.d_values);

      if (retval = Setup(csr_problem, expand_grid_size,
          contract_grid_size, 0))
        return retval;

      SizeT queue_length;
      VertexId queue_index = 0;
      int selector = 0;

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      double startcontract, endcontract;
      double startexpand, endexpand;
      double startgather, endgather;

      double elapsedcontract = 0.0;
      double elapsedexpand = 0.0;
      double elapsedgather = 0.0;

      cudaEventRecord(start);
      double startTime = omp_get_wtime();

      for (int i = 0; i < iter_num; i++)
      {
        if (DEBUG)
          printf("Iteration: %lld\n", (long long) iteration[0]);
        if (DEBUG)
          startcontract = omp_get_wtime();
        //
        // Contraction
        //
        vertex_centric::contract_atomic::Kernel<ContractPolicy, Program><<<
        contract_grid_size, ContractPolicy::THREADS>>>(
            iteration[0],
            num_srcs, // initial num_elements
            queue_index,// queue counter index
            queue_index,// steal counter index
            1,// number of GPUs
            d_done,
            graph_slice->frontier_queues.d_keys[selector ^ 1],// filtered edge frontier in
            graph_slice->frontier_queues.d_keys[selector],// vertex frontier out
            graph_slice->frontier_queues.d_values[selector ^ 1],// predecessor in
            graph_slice->vertex_list,
            graph_slice->edge_list,
            graph_slice->d_preds,// prtedecessor out
            graph_slice->d_visited_mask,
            this->work_progress,
            graph_slice->frontier_elements[selector ^ 1],// max filtered edge frontier vertices
            graph_slice->frontier_elements[selector]);

        if (DEBUG
            && (retval = util::B40CPerror(cudaThreadSynchronize(),
                "contract_atomic::Kernel failed ", __FILE__,
                __LINE__)))
          break;
        cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

        if (DEBUG)
        {
          cudaDeviceSynchronize();
          endcontract = omp_get_wtime();
          elapsedcontract += endcontract - startcontract;
        }

        queue_index++;
        selector ^= 1;

        if (DEBUG)
        {
          if (retval = work_progress.GetQueueLength(queue_index,
              queue_length))
            break;
          printf("queue_length after contraction: %lld\n",
              (long long) queue_length);
//
//          printf("Frontier size after contract: %d\n", queue_length);
//          VertexId* test_vid = new VertexId[queue_length];
//          cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost);
//          printf("Frontier after contract: ");
//          for (int i = 0; i < queue_length; ++i)
//          {
//            printf("%d, ", test_vid[i]);
//          }
//          printf("\n");
//          delete[] test_vid;
        }

        // Throttle
        if (iteration[0] & 1)
        {
          if (retval =
              util::B40CPerror(cudaEventRecord(throttle_event),
                  "EnactorVertexCentric cudaEventRecord throttle_event failed",
                  __FILE__, __LINE__))
            break;
        }
        else
        {
          if (retval =
              util::B40CPerror(
                  cudaEventSynchronize(throttle_event),
                  "EnactorVertexCentric cudaEventSynchronize throttle_event failed",
                  __FILE__, __LINE__))
            break;
        };

        // Check if done
        if (done[0] == 0)
          break;

        if (DEBUG)
        {
          cudaDeviceSynchronize();
          startgather = omp_get_wtime();
        }

        //
        //Gather stage
        //
        if (directed == 0)
        {
          vertex_centric::gather::Kernel<GatherPolicy, Program><<<
          gather_grid_size, GatherPolicy::THREADS>>>(
              queue_index,              // queue counter index
              queue_index,// steal counter index
              1,// number of GPUs
              d_done,
              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->d_column_indices,//pass in the CSC graph to gather for destination vertices
              graph_slice->d_row_offsets,//pass in the CSC graph to gather for destination vertices
              graph_slice->vertex_list,//
              graph_slice->edge_list,//
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector]);
        }
        else
        {
          if (Program::gatherOverEdges() == GATHER_IN_EDGES)
          {

            vertex_centric::gather::Kernel<GatherPolicy, Program><<<
            gather_grid_size, GatherPolicy::THREADS>>>(
                queue_index,              // queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                graph_slice->d_row_indices,//pass in the CSC graph to gather for destination vertices
                graph_slice->d_column_offsets,//pass in the CSC graph to gather for destination vertices
                graph_slice->vertex_list,//
                graph_slice->edge_list,//
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);

          }
          else if (Program::gatherOverEdges() == GATHER_OUT_EDGES)
          {
            vertex_centric::gather::Kernel<GatherPolicy, Program><<<
            gather_grid_size, GatherPolicy::THREADS>>>(
                queue_index,              // queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                graph_slice->d_column_indices,//pass in the CSC graph to gather for destination vertices
                graph_slice->d_row_offsets,//pass in the CSC graph to gather for destination vertices
                graph_slice->vertex_list,//
                graph_slice->edge_list,//
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);

          }
          else if (Program::gatherOverEdges() == GATHER_ALL_EDGES)
          {
            vertex_centric::gather::Kernel<GatherPolicy, Program><<<
            gather_grid_size, GatherPolicy::THREADS>>>(
                queue_index,              // queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                graph_slice->d_row_indices,//pass in the CSC graph to gather for destination vertices
                graph_slice->d_column_offsets,//pass in the CSC graph to gather for destination vertices
                graph_slice->vertex_list,//
                graph_slice->edge_list,//
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);

            if (DEBUG
                && (retval = util::B40CPerror(
                        cudaThreadSynchronize(),
                        "gather1::Kernel failed ", __FILE__,
                        __LINE__)))
            break;

            vertex_centric::gather::Kernel<GatherPolicy, Program><<<
            gather_grid_size, GatherPolicy::THREADS>>>(
                queue_index,// queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                graph_slice->d_column_indices,//pass in the CSC graph to gather for destination vertices
                graph_slice->d_row_offsets,//pass in the CSC graph to gather for destination vertices
                graph_slice->vertex_list,//
                graph_slice->edge_list,//
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);

            if (DEBUG
                && (retval = util::B40CPerror(
                        cudaThreadSynchronize(),
                        "gather2::Kernel failed ", __FILE__,
                        __LINE__)))
            break;
          }
        }

        cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates
        if (DEBUG)
        {
//          printf("gather value: \n");
//          float* test_vid = new float[graph_slice->nodes];
//          cudaMemcpy(test_vid, graph_slice->vertex_list.d_min_dists, graph_slice->nodes * sizeof(float), cudaMemcpyDeviceToHost);
////          printf("Frontier after expand: ");
//          for (int i = 0; i < graph_slice->nodes; ++i)
//          {
//            printf("%f, ", test_vid[i]);
//          }
//          printf("\n");
//          delete[] test_vid;
          cudaDeviceSynchronize();
          endgather = omp_get_wtime();
          elapsedgather += endgather - startgather;
        }

        if (DEBUG)
        {
          if (work_progress.GetQueueLength(queue_index, queue_length))
            break;
          total_queued += queue_length;
        }

        if (DEBUG)
          startexpand = omp_get_wtime();

        if (Program::applyOverEdges() == APPLY_FRONTIER)
        {
          //
          //apply stage
          //
          vertex_centric::gather::apply<GatherPolicy, Program><<<
          gather_grid_size, GatherPolicy::THREADS>>>(
              iteration[0], queue_index, this->work_progress,
              graph_slice->frontier_queues.d_keys[selector ^ 1],
              graph_slice->vertex_list, graph_slice->edge_list);
        }

        if (Program::postApplyOverEdges() == POST_APPLY_FRONTIER)
        {

          //
          //reset dists and gather_results
          //
          vertex_centric::gather::reset_gather_result<GatherPolicy,
              Program><<<gather_grid_size, GatherPolicy::THREADS>>>(
              iteration[0], queue_index, this->work_progress,
              graph_slice->frontier_queues.d_keys[selector ^ 1],
              graph_slice->vertex_list, graph_slice->edge_list,
              graph_slice->d_visited_mask);

          if (DEBUG
              && (retval = util::B40CPerror(
                      cudaThreadSynchronize(),
                      "gather::reset_changed Kernel failed ",
                      __FILE__, __LINE__)))
          break;
        }
        else if (Program::postApplyOverEdges() == POST_APPLY_ALL)
        {
          vertex_centric::gather::reset_gather_result<GatherPolicy,
          Program><<<gather_grid_size, GatherPolicy::THREADS>>>(
              iteration[0], graph_slice->nodes,
              graph_slice->vertex_list, graph_slice->edge_list,
              graph_slice->d_visited_mask);
        }

        //
        // Expansion
        //
        if (directed == 0)
        {
          vertex_centric::expand_atomic::Kernel<ExpandPolicy, Program><<<
          expand_grid_size, ExpandPolicy::THREADS>>>(
              iteration[0],
              queue_index,              // queue counter index
              queue_index,
              1,// number of GPUs
              d_done,
              graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
              graph_slice->frontier_queues.d_keys[selector],// edge frontier out
              graph_slice->frontier_queues.d_values[selector],// predecessor out
              graph_slice->vertex_list,//
              graph_slice->edge_list,
              graph_slice->d_column_indices,
              graph_slice->d_row_offsets,
              this->work_progress,
              graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
              graph_slice->frontier_elements[selector]);

        }
        else
        {
          if (Program::expandOverEdges() == EXPAND_OUT_EDGES)
          {

            vertex_centric::expand_atomic::Kernel<ExpandPolicy,
            Program><<<expand_grid_size,
            ExpandPolicy::THREADS>>>(iteration[0],
                queue_index,              // queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                graph_slice->frontier_queues.d_values[selector],// predecessor out
                graph_slice->vertex_list,//
                graph_slice->edge_list,
                graph_slice->d_column_indices,
                graph_slice->d_row_offsets,
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);
          }
          else if (Program::expandOverEdges() == EXPAND_IN_EDGES)
          {
            vertex_centric::expand_atomic::Kernel<ExpandPolicy,
            Program><<<expand_grid_size,
            ExpandPolicy::THREADS>>>(iteration[0],
                queue_index,              // queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                graph_slice->frontier_queues.d_values[selector],// predecessor out
                graph_slice->vertex_list,//
                graph_slice->edge_list,
                graph_slice->d_row_indices,
                graph_slice->d_column_offsets,
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);
          }
          else if (Program::expandOverEdges() == EXPAND_ALL_EDGES)
          {
            int queue_length1 = 0, queue_length2 = 0;
            vertex_centric::expand_atomic::Kernel<ExpandPolicy,
            Program><<<expand_grid_size,
            ExpandPolicy::THREADS>>>(iteration[0],
                queue_index,              // queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[selector],// edge frontier out
                graph_slice->frontier_queues.d_values[selector],// predecessor out
                graph_slice->vertex_list,//
                graph_slice->edge_list,
                graph_slice->d_column_indices,
                graph_slice->d_row_offsets,
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);

            if (work_progress.GetQueueLength(queue_index + 1,
                    queue_length1))
            break;

            work_progress.SetQueueLength(queue_index + 1, 0);

            vertex_centric::expand_atomic::Kernel<ExpandPolicy,
            Program><<<expand_grid_size,
            ExpandPolicy::THREADS>>>(iteration[0],
                queue_index,// queue counter index
                queue_index,// steal counter index
                1,// number of GPUs
                d_done,
                graph_slice->frontier_queues.d_keys[selector ^ 1],// vertex frontier in
                graph_slice->frontier_queues.d_keys[2],// edge frontier out
                graph_slice->frontier_queues.d_values[2],// predecessor out
                graph_slice->vertex_list,//
                graph_slice->edge_list,
                graph_slice->d_row_indices,
                graph_slice->d_column_offsets,
                this->work_progress,
                graph_slice->frontier_elements[selector ^ 1],// max vertex frontier vertices
                graph_slice->frontier_elements[selector]);

            //combine two edge frontier out
            if (work_progress.GetQueueLength(queue_index + 1,
                    queue_length2))
            break;

            cudaMemcpy(
                graph_slice->frontier_queues.d_keys[selector]
                + queue_length1,
                graph_slice->frontier_queues.d_keys[2],
                queue_length2 * sizeof(VertexId),
                cudaMemcpyDeviceToDevice);

            cudaMemcpy(
                graph_slice->frontier_queues.d_values[selector]
                + queue_length1,
                graph_slice->frontier_queues.d_values[2],
                queue_length2 * sizeof(VertexId),
                cudaMemcpyDeviceToDevice);

            work_progress.SetQueueLength(queue_index + 1,
                queue_length1 + queue_length2);

          }
        }

        if (DEBUG
            && (retval = util::B40CPerror(cudaThreadSynchronize(),
                "expand_atomic::Kernel failed ", __FILE__,
                __LINE__)))
          break;
        cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

        if (DEBUG)
        {
          cudaDeviceSynchronize();
          endexpand = omp_get_wtime();
          elapsedexpand += endexpand - startexpand;
        }

        queue_index++;
        selector ^= 1;
        iteration[0]++;

        if (DEBUG)
        {
          if (work_progress.GetQueueLength(queue_index, queue_length))
            break;

          total_queued += queue_length;
          printf("queue_length after expansion: %lld\n",
              (long long) queue_length);

//          printf("Frontier size after expand: %d\n", queue_length);
//          VertexId* test_vid = new VertexId[queue_length];
//          cudaMemcpy(test_vid, graph_slice->frontier_queues.d_keys[selector ^ 1], queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost);
//          printf("Frontier after expand: ");
//          for (int i = 0; i < queue_length; ++i)
//          {
//            printf("%d, ", test_vid[i]);
//          }
//          printf("\n");
//          delete[] test_vid;
        }

        // Check if done
        if (done[0] == 0)
          break;
      }

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaDeviceSynchronize();
      double endTime = omp_get_wtime();
      double elapsed_wall = (omp_get_wtime() - startTime) * 1000;

      float elapsed;
      cudaEventElapsedTime(&elapsed, start, stop);
      std::cout << "Kernel time took: " << elapsed << " ms" << std::endl;
      std::cout << "Wall time took: " << elapsed_wall << " ms"
          << std::endl;
      std::cout << "Contract time took: " << elapsedcontract * 1000
          << " ms" << std::endl;
      std::cout << "Gather time took: " << elapsedgather * 1000
          << " ms" << std::endl;
      std::cout << "Expand time took: " << elapsedexpand * 1000 << " ms"
          << std::endl;

      printf("Total iteration: %lld\n", (long long) iteration[0]);

      // Check if any of the frontiers overflowed due to redundant expansion
      bool overflowed = false;
      retval = work_progress.CheckOverflow<SizeT>(overflowed);
      if (overflowed)
      {
        retval =
            util::B40CPerror(cudaErrorInvalidConfiguration,
                "Frontier queue overflow.  Please increase queue-sizing factor. ",
                __FILE__, __LINE__);
        retval = (cudaError_t) 1;
      }
      std::cout << "retval: " << retval << std::endl;

      return retval;
    }

    template<typename CsrProblem, typename Program>
    cudaError_t EnactIterativeSearch(CsrProblem &csr_problem,
        typename CsrProblem::SizeT* h_row_offsets,
        int directed, int num_srcs, int* srcs, int iter_num)
    {
      typedef typename CsrProblem::VertexId VertexId;
      typedef typename CsrProblem::SizeT SizeT;

      // GF100
      if (this->cuda_props.device_sm_version >= 200)
      {

        // Expansion kernel config
        typedef vertex_centric::expand_atomic::KernelPolicy<Program,
            typename CsrProblem::ProblemType, 200, // CUDA_ARCH
            INSTRUMENT, // INSTRUMENT
            1, // CTA_OCCUPANCY
            9, // LOG_THREADS
            0, // LOG_LOAD_VEC_SIZE
            0, // LOG_LOADS_PER_TILE
            5, // LOG_RAKING_THREADS
            util::io::ld::cg, // QUEUE_READ_MODIFIER,
            util::io::ld::NONE, // COLUMN_READ_MODIFIER,
            util::io::ld::NONE, // EDGE_VALUES_READ_MODIFIER,
            util::io::ld::cg, // ROW_OFFSET_ALIGNED_READ_MODIFIER,
            util::io::ld::NONE, // ROW_OFFSET_UNALIGNED_READ_MODIFIER,
            util::io::st::cg, // QUEUE_WRITE_MODIFIER,
            false, // WORK_STEALING
            32, // WARP_GATHER_THRESHOLD
            128 * 4, // CTA_GATHER_THRESHOLD,
            7> // LOG_SCHEDULE_GRANULARITY
        ExpandPolicy;

        // Gather kernel config
        typedef vertex_centric::gather::KernelPolicy<Program,
            typename CsrProblem::ProblemType, 200, // CUDA_ARCH
            INSTRUMENT, // INSTRUMENT
            1, // CTA_OCCUPANCY
            9, // LOG_THREADS
            0, // LOG_LOAD_VEC_SIZE
            0, // LOG_LOADS_PER_TILE
            5, // LOG_RAKING_THREADS
            util::io::ld::cg, // QUEUE_READ_MODIFIER,
            util::io::ld::NONE, // COLUMN_READ_MODIFIER,
            util::io::ld::NONE, // EDGE_VALUES_READ_MODIFIER,
            util::io::ld::cg, // ROW_OFFSET_ALIGNED_READ_MODIFIER,
            util::io::ld::NONE, // ROW_OFFSET_UNALIGNED_READ_MODIFIER,
            util::io::st::cg, // QUEUE_WRITE_MODIFIER,
            false, // NO WORK_STEALING : changed from true
            32, // WARP_GATHER_THRESHOLD
            128 * 4, // CTA_GATHER_THRESHOLD,
            7> // LOG_SCHEDULE_GRANULARITY
        GatherPolicy;

        // Contraction kernel config
        typedef vertex_centric::contract_atomic::KernelPolicy<Program,
            typename CsrProblem::ProblemType, 200, // CUDA_ARCH
            INSTRUMENT, // INSTRUMENT
            0, // SATURATION_QUIT
            true, // DEQUEUE_PROBLEM_SIZE
            8, // CTA_OCCUPANCY
            7, // LOG_THREADS
            1, // LOG_LOAD_VEC_SIZE
            0, // LOG_LOADS_PER_TILE
            5, // LOG_RAKING_THREADS
            util::io::ld::NONE, // QUEUE_READ_MODIFIER,
            util::io::st::NONE, // QUEUE_WRITE_MODIFIER,
            false, // WORK_STEALING
            -1, // END_BITMASK_CULL 0 to never perform bitmask filtering, -1 to always perform bitmask filtering
            8> // LOG_SCHEDULE_GRANULARITY
        ContractPolicy;

        return EnactIterativeSearch<ExpandPolicy, GatherPolicy,
            ContractPolicy, Program>(csr_problem,
            h_row_offsets, directed, num_srcs, srcs, iter_num);
      }

      printf("Not yet tuned for this architecture\n");
      return cudaErrorInvalidDeviceFunction;
    }
  };

} // namespace GASengine

