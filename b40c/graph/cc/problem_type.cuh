/******************************************************************************
 * Connected component problem type
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace graph {
namespace cc {


/**
 * Type of CC problem
 */
template <typename 	_VertexId>						                                        // Type of signed integer to use as vertex id (e.g., uint64)
struct ProblemType
{
	typedef _VertexId   VertexId;

    static const _VertexId FROM_VERTEX_OFFSET = 32;
	static const _VertexId TO_VERTEX_ID_MASK	= 0xFFFFFFFF;								                // Bitmask for getting to vertex id in edge tuple
	static const _VertexId FROM_VERTEX_ID_MASK =  (TO_VERTEX_ID_MASK<<FROM_VERTEX_OFFSET);                    // Bitmask for getting from vertex id in edge tuple
};


} // namespace cc
} // namespace graph
} // namespace b40c

