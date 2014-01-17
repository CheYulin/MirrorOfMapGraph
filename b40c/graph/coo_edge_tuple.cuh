/******************************************************************************
 * Simple COO sparse graph data structure
 ******************************************************************************/

#pragma once

#include <time.h>
#include <stdio.h>

#include <algorithm>

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace graph {


/**
 * COO sparse format edge.  (A COO graph is just a list/array/vector of these.)
 */
template<typename VertexId, typename Value>
struct CooEdgeTuple {
	VertexId row;
	VertexId col;
	Value val;

	CooEdgeTuple(VertexId row, VertexId col, Value val) : row(row), col(col), val(val) {}

	void Val(Value &value)
	{
		value = val;
	}
};


template<typename VertexId>
struct CooEdgeTuple<VertexId, util::NullType> {
	VertexId row;
	VertexId col;

	template <typename Value>
	CooEdgeTuple(VertexId row, VertexId col, Value val) : row(row), col(col) {}

	template <typename Value>
	void Val(Value &value) {}
};


/**
 * Comparator for sorting COO sparse format edges
 */
template<typename CooEdgeTuple>
bool DimacsTupleCompare (
	CooEdgeTuple elem1,
	CooEdgeTuple elem2)
{
	if (elem1.row < elem2.row) {
		// Sort edges by source node (to make rows)
		return true;
/*
	} else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
		// Sort edgelists as well for coherence
		return true;
*/
	} 
	
	return false;
}

/**
 *  * Comparator for sorting COO sparse format edges
 *   */
template<typename CooEdgeTuple>
bool DimacsTupleCompare2 (
        CooEdgeTuple elem1,
        CooEdgeTuple elem2)
{
	if (elem1.col < elem2.col) {
	    // Sort edges by source node (to make rows)
	    return true;
	    /*
	       } else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
	    // Sort edgelists as well for coherence
	    return true;
	     */
	} 

	return false;
}


} // namespace graph
} // namespace b40c
