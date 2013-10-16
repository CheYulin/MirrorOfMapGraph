#pragma once

namespace b40c {
namespace graph {
namespace cc {

/**
 * Kernel configuration policy for connected component labeling kernels.
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type. By
 * incorporating this type into the kernel code itselt, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */

template<
    // ProblemType type parameters
    typename _ProblemType,                      //CC problem type (e.g., b40c::graph::cc::ProblemType)

    // Machine parameters
    int CUDA_ARCH,                              //CUDA SM architecture to generate code for

    // Behavioral control parameters
    bool _INSTRUMENT,

    // Tunable parameters (generic)
    int MIN_CTA_OCCUPANCY,                      //Lower bound on number of CTAs to have resident per SM (influences per-CTA smem cache sizes and register allocation/spills)
    int _LOG_THREADS>                           //Number of threads per CTA (log)

struct KernelPolicy : _ProblemType
{
	//----------------------------------------------------
	// Constants and typedefs
	//----------------------------------------------------
	
	typedef _ProblemType                                            ProblemType;

    // Constants
	enum {
		INSTRUMENT                  = _INSTRUMENT,
		LOG_THREADS                 = _LOG_THREADS,
		THREADS                     = 1 << LOG_THREADS,

		THREAD_OCCUPANCY            = B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		CTA_OCCUPANCY               = B40C_MIN(MIN_CTA_OCCUPANCY, B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), THREAD_OCCUPANCY)),

		VALID                       = (CTA_OCCUPANCY > 0),
	};
};

} // namespace cc
} // namespace graph
} // namespace b40c

