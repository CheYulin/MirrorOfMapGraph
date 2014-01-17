
/******************************************************************************
 * Base Search Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>

#include <GASengine/csr_problem.cuh>

using namespace b40c;

namespace GASengine
{
  class EnactorBase
  {
  protected:

    //Device properties
    util::CudaProperties cuda_props;

    // Queue size counters and accompanying functionality
    util::CtaWorkProgressLifetime work_progress;

    FrontierType frontier_type;

  public:

    // Allows display to stdout of search details
    int DEBUG;

    FrontierType GetFrontierType()
    {
      return frontier_type;
    }

  protected:

    /**
     * Constructor.
     */
    EnactorBase(FrontierType frontier_type, bool DEBUG) :
        frontier_type(frontier_type),
            DEBUG(DEBUG)
    {
      // Setup work progress (only needs doing once since we maintain
      // it in our kernel code)
      work_progress.Setup();
    }

    /**
     * Utility function: Returns the default maximum number of threadblocks
     * this enactor class can launch.
     */
    int MaxGridSize(int cta_occupancy, int max_grid_size = 0)
    {
      if (max_grid_size <= 0)
      {
        // No override: Fully populate all SMs
        max_grid_size = this->cuda_props.device_props.multiProcessorCount * cta_occupancy;
      }

      return max_grid_size;
    }

    /**
     * Utility method to display the contents of a device array
     */
    template<typename T>
    void DisplayDeviceResults(
        T *d_data,
        size_t num_elements)
    {
      // Allocate array on host and copy back
      T *h_data = (T*) malloc(num_elements * sizeof(T));
      cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

      // Display data
      for (int i = 0; i < num_elements; i++)
      {
        PrintValue(h_data[i]);
        printf(", ");
      }
      printf("\n\n");

      // Cleanup
      if (h_data) free(h_data);
    }
  };

} // namespace GASengine
