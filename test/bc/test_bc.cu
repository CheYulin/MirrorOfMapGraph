/******************************************************************************
 * Copyright 2010-2012 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 ******************************************************************************/


/******************************************************************************
 * Simple test driver program for BFS graph traversal.
 *
 * Useful for demonstrating how to integrate BFS traversal into your 
 * application. 
 ******************************************************************************/


#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <fstream>

//ignore cub for now
//#include <cub.cuh>

// Utilities and correctness-checking
#include <b40c_test_util.h>

// Graph construction utils
#include <b40c/graph/builder/dimacs.cuh>
#include <b40c/graph/builder/grid2d.cuh>
#include <b40c/graph/builder/grid3d.cuh>
#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/metis.cuh>
#include <b40c/graph/builder/rmat.cuh>
#include <b40c/graph/builder/random.cuh>
#include <b40c/graph/builder/rr.cuh>
#include <b40c/graph/builder/fromfile.cuh>


// BFS includes
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/bfs/enactor_contract_expand.cuh>
#include <b40c/graph/bfs/enactor_expand_contract.cuh>
#include <b40c/graph/bfs/enactor_two_phase.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>
#include <b40c/graph/bfs/enactor_multi_gpu.cuh>

#include <b40c/graph/bc/csr_problem.cuh>
#include <b40c/graph/bc/enactor_vertex_centric.cuh>

using namespace b40c;
using namespace graph;
using namespace std;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

enum Strategy {
	HOST = -1,
	EXPAND_CONTRACT,
	CONTRACT_EXPAND,
	TWO_PHASE,
	HYBRID,
	MULTI_GPU,
    VERTEX_CENTRIC
};


//#define __B40C_ERROR_CHECKING__		 

bool g_verbose;
bool g_verbose2;
bool g_undirected;			// Whether to add a "backedge" for every edge parsed/generated
bool g_quick;				// Whether or not to perform CPU traversal as reference
bool g_stream_from_host;	// Whether or not to stream CSR representation from host mem
bool g_with_value;          // Whether or not to load value


#define DEBUG 1
/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */

void Usage() 
{
	printf("\ntest_bfs <graph type> <graph type args> [--device=<device index>] "
			"[--v] [--instrumented] [--i=<num-iterations>] [--undirected]"
			"[--src=< <source idx> | randomize >] [--queue-sizing=<sizing factor>\n"
			"[--mark-pred] [--strategy=<strategy>[,<strategy>]*]\n"
			"\n"
			"Unless otherwise specified, all graph types use 4-byte vertex-identifiers.\n"
			"\n"
			"Graph types and args:\n"
			"  grid2d <width>\n"
			"    2D square grid lattice having width <width>.  Interior vertices \n"
			"    have 4 neighbors and 1 self-loop.  Default source vertex is the grid-center.\n"
			"  grid3d <side-length>\n"
			"    3D square grid lattice having width <width>.  Interior vertices \n"
			"    have 6 neighbors and 1 self-loop.  Default source vertex is the grid-center.\n"
			"  dimacs [<file>]\n"
			"    Reads a DIMACS-formatted graph of directed edges from stdin (or \n"
			"    from the optionally-specified file).  Default source vertex is random.\n"
			"  metis [<file>]\n"
			"    Reads a METIS-formatted graph of directed edges from stdin (or \n"
			"    from the optionally-specified file).  Default source vertex is random.\n"
			"  market [<file>]\n"
			"    Reads a Matrix-Market coordinate-formatted graph of directed edges from stdin (or \n"
			"    from the optionally-specified file).  Default source vertex is random.\n"
			"  random <n> <m>\n"
			"    A random graph generator that adds <m> edges to <n> nodes by randomly \n"
			"    choosing a pair of nodes for each edge.  There are possibilities of \n"
			"    loops and multiple edges between pairs of nodes. Default source vertex \n"
			"    is random.\n"
			"  rr <n> <d>\n"
			"    A random graph generator that adds <d> randomly-chosen edges to each\n"
			"    of <n> nodes.  There are possibilities of loops and multiple edges\n"
			"    between pairs of nodes. Default source vertex is random.\n"
			"  g500 <n>\n"
			"    An R-MAT graph generator that adds 16n undirected edges to <n> nodes in accordance with\n"
			"    the Graph500 problem specification (8-byte vertex identifiers, A=.57,B=.19,C=.19,D=.05 "
			"    skew parameters).\n"
			"  rmat <n> <m>\n"
			"    An R-MAT graph generator that adds <m> edges to <n> nodes in accordance with\n"
			"    the GTGraph generator defaults (A=.45,B=.15,C=.15,D=.25 skew parameters\n"
            "  fromfile <file>\n"
            "    Reads graph data from file. \n"
			"\n"
           "  akamai <file>\n"
            "    Reads a directed graph Akamai data from file. Default source vertex is random.\n"
			"\n"
			"--strategy  Specifies the strategies to evaluate when num-gpus specified <= 1.\n"
			"  Valid strategies are: {%d, %d, %d, %d}. Default: %d\n"
			"      \"%d\": expand-contract\n"
			"        - Two O(n) global ping-pong buffers (out-of-core vertex frontier)\n"
			"        - single kernel invocation (in-kernel software global barriers between BFS iterations)\n"
			"        - Predecessor-marking not implemented\n"
			"      \"%d\": contract-expand\n"
			"        - Two O(m) global ping-pong buffers (out-of-core edge frontier)\n"
			"        - Single kernel invocation (in-kernel software global barriers between BFS iterations)\n"
			"      \"%d\": two-phase\n"
			"        - Uneven O(n) and O(m) global ping-pong buffers (out-of-core vertex and edge frontiers)\n"
			"        - Two kernel invocations per BFS iteration (pipelined)\n"
			"      \"%d\": hybrid of contract-expand and two-phase strategies\n"
			"        - Uses high-throughput two-phase for BFS iterations with lots of concurrency,\n"
			"          switching to contract-expand when frontier falls below a certain threshold\n"
			"        - Two O(m) global ping-pong buffers \n"
			"\n"
			"--v  Verbose launch and statistical output is displayed to the console.\n"
			"\n"
			"--v2  Same as --v, but also displays the input graph to the console.\n"
			"\n"
			"--instrumented  Kernels keep track of queue-search_depth, redundant work (i.e., the \n"
			"    overhead of duplicates in the frontier), and average barrier duty (a \n"
			"    relative indicator of load imbalance.)\n"
			"\n"
			"--i  Performs <num-iterations> test-iterations of BFS traversals.\n"
			"    Default = 1\n"
			"\n"
			"--src  Begins BFS from the vertex <source idx>. Default is specific to \n"
			"    graph-type.  If alternatively specified as \"randomize\", each \n"
			"    test-iteration will begin with a newly-chosen random source vertex.\n"
			"\n"
			"--queue-sizing  Allocates a frontier queue sized at (graph-edges * <queue-sizing>).  Default\n"
			"    is 1.15.\n"
			"\n"
			"--mark-pred  Parent vertices are marked instead of source distances, i.e., it\n"
			"    creates an ancestor tree rooted at the source vertex.\n"
			"\n"
            "--with-value Will perform a simple computation which sums all the edge values to neighbors\n"
			"    and stores it as the node value\n"
			"\n"
			"--stream-from-host  Keeps the graph data (column indices, row offsets) on the host,\n"
			"    using zero-copy access to traverse it.\n"
			"\n"
			"--num-gpus  Number of GPUs to use\n"
			"\n"
			"--undirected  Edges are undirected.  Reverse edges are added to DIMACS and\n"
			"    random graphs, effectively doubling the CSR graph representation size.\n"
			"    Grid2d/grid3d graphs are undirected regardless of this flag, and rr \n"
			"    graphs are directed regardless of this flag.\n"
			"\n",
				EXPAND_CONTRACT, CONTRACT_EXPAND, TWO_PHASE, HYBRID,
				HYBRID,
				EXPAND_CONTRACT, CONTRACT_EXPAND, TWO_PHASE, HYBRID);
}

/**
 * Displays the BFS result (i.e., distance from source)
 */
template<typename VertexId, typename SizeT, typename Value>
void DisplaySolution(VertexId* source_path, Value* node_values, Value* sigmas, SizeT nodes)
{
	if (nodes <= 20)
	{
	    printf("search_depth[");
	    for (VertexId i = 0; i < nodes; ++i) {
		    PrintValue(i);
		    printf(":");
		    PrintValue(source_path[i]);
		    printf(", ");
	    }
	    printf("]\n");

        printf("bc value[");
	    for (VertexId i = 0; i < nodes; ++i) {
		    PrintValue(i);
		    printf(":");
		    PrintValue(node_values[i]);
		    printf(", ");
	    }
	    printf("]\n");

        printf("sigmas[");
	    for (VertexId i = 0; i < nodes; ++i) {
		    PrintValue(i);
		    printf(":");
		    PrintValue(sigmas[i]);
		    printf(", ");
	    }
	    printf("]\n");
	}
	else
	{
        /*std::ofstream of("bc_values.txt");
        cout << "NID\tsearch_depth\tbc_value" << endl;
        for (VertexId i = 0; i < nodes; ++i)
        {
        	of << i <<"\t"<<source_path[i]<<"\t"<<node_values[i]<<endl;
        }*/
	}
}



/******************************************************************************
 * Performance/Evaluation Statistics
 ******************************************************************************/

struct Statistic 
{
	double mean;
	double m2;
	int count;
	
	Statistic() : mean(0.0), m2(0.0), count(0) {}
	
	/**
	 * Updates running statistic, returning bias-corrected sample variance.
	 * Online method as per Knuth.
	 */
	double Update(double sample)
	{
		count++;
		double delta = sample - mean;
		mean = mean + (delta / count);
		m2 = m2 + (delta * (sample - mean));
		return m2 / (count - 1);					// bias-corrected 
	}
	
};

struct Stats {
	char *name;
	Statistic rate;
	Statistic search_depth;
	Statistic redundant_work;
	Statistic duty;
	
	Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
	Stats(char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};


template <typename SizeT>
struct HistogramLevel
{
	SizeT		discovered;
	SizeT		expanded;
	SizeT		unique_expanded;

	HistogramLevel() : discovered(0), expanded(0), unique_expanded(0) {}
};


/**
 * Displays a histogram of search behavior by level depth, i.e., expanded,
 * unique, and newly-discovered nodes at each level
 */
template <
	typename VertexId,
	typename Value,
	typename SizeT>
void Histogram(
	VertexId 								src,
	VertexId 								*reference_labels,					// reference answer
	const CsrGraph<VertexId, Value, SizeT> 	&csr_graph,	// reference host graph
	VertexId								search_depth)
{
	std::vector<HistogramLevel<SizeT> > histogram(search_depth + 1);
	std::vector<std::vector<VertexId> > frontier(search_depth + 1);

	// Establish basics
	histogram[0].expanded = 1;
	histogram[0].unique_expanded = 1;

	for (VertexId vertex = 0; vertex < csr_graph.nodes; vertex++) {

		VertexId distance = reference_labels[vertex];
		if (distance >= 0) {

			SizeT row_offset 	= csr_graph.row_offsets[vertex];
			SizeT row_oob 		= csr_graph.row_offsets[vertex + 1];
			SizeT neighbors 	= row_oob - row_offset;

			histogram[distance].discovered++;
			histogram[distance + 1].expanded += neighbors;
		}
	}

	// Allocate frontiers
	for (VertexId distance = 0; distance < search_depth; distance++) {
		frontier[distance].reserve(histogram[distance].expanded);
	}

	// Construct frontiers
	for (VertexId vertex = 0; vertex < csr_graph.nodes; vertex++) {

		VertexId distance = reference_labels[vertex];
		if (distance >= 0) {

			SizeT row_offset 	= csr_graph.row_offsets[vertex];
			SizeT row_oob 		= csr_graph.row_offsets[vertex + 1];

			frontier[distance].insert(
				frontier[distance].end(),
				csr_graph.column_indices + row_offset,
				csr_graph.column_indices + row_oob);
		}
	}

	printf("Work Histogram:\n");
	printf("Depth, Expanded, Unique-Expanded, Discovered\n");
	for (VertexId distance = 0; distance < search_depth; distance++) {

		// Sort
		std::sort(
			frontier[distance].begin(),
			frontier[distance].end());

		// Count unique elements
		histogram[distance + 1].unique_expanded =
			std::unique(frontier[distance].begin(), frontier[distance].end()) -
			frontier[distance].begin();

		printf("%lld, %d, %d, %d\n",
			(long long) distance,
			histogram[distance].expanded,
			histogram[distance].unique_expanded,
			histogram[distance].discovered);
	}
	printf("\n\n");
}


/**
 * Displays timing and correctness statistics 
 */
template <
	bool MARK_PREDECESSORS,
	bool WITH_VALUE,
	typename VertexId,
	typename Value,
	typename SizeT>
void DisplayStats(
	Stats 									&stats,
	VertexId 								src,
	VertexId 								*h_labels,							// computed answer
	VertexId 								*reference_labels,					// reference answer
	Value                                   *h_node_values,
	Value                                   *ref_node_values,
    Value                                   *h_sigmas,
    Value                                   *ref_sigmas,
	const CsrGraph<VertexId, Value, SizeT> 	&csr_graph,	// reference host graph
	double 									elapsed,
	VertexId								search_depth,
	long long 								total_queued,
	double 									avg_duty)
{
	// Compute nodes and edges visited
	SizeT edges_visited = 0;
	SizeT nodes_visited = 0;
	for (VertexId i = 0; i < csr_graph.nodes; i++) {
		if (h_labels[i] > -1) {
			nodes_visited++;
			edges_visited += csr_graph.row_offsets[i + 1] - csr_graph.row_offsets[i];
		}
	}
	
	double redundant_work = 0.0;
	if (total_queued > 0)  {
		redundant_work = ((double) total_queued - edges_visited) / edges_visited;		// measure duplicate edges put through queue
	}
	redundant_work *= 100;

	// Display test name
	printf("[%s] finished. ", stats.name);

	// Display correctness
	if (reference_labels != NULL) {
		printf("Validity: ");
		fflush(stdout);
		if (!MARK_PREDECESSORS) {

			// Simply compare with the reference source-distance
			CompareResults(h_labels, reference_labels, csr_graph.nodes, true);

            CompareResults(h_sigmas, ref_sigmas, csr_graph.nodes, true);

            if (WITH_VALUE)
			    CompareResults(h_node_values, ref_node_values, csr_graph.nodes, true);

		} else {
            if (WITH_VALUE)
                CompareResults(h_node_values, ref_node_values, csr_graph.nodes, true);

			// Verify plausibility of parent markings
			bool correct = true;

			for (VertexId node = 0; node < csr_graph.nodes; node++) {
				VertexId parent = h_labels[node];

				// Check that parentless nodes have zero or unvisited source distance
				VertexId node_dist = reference_labels[node];
				if (parent < 0) {
					if (reference_labels[node] > 0) {
						printf("INCORRECT: parentless node %lld (parent %lld) has positive distance distance %lld",
							(long long) node, (long long) parent, (long long) node_dist);
						correct = false;
						break;
					}
					continue;
				}

				// Check that parent has iteration one less than node
				VertexId parent_dist = reference_labels[parent];
				if (parent_dist + 1 != node_dist) {
					printf("INCORRECT: parent %lld has distance %lld, node %lld has distance %lld",
						(long long) parent, (long long) parent_dist, (long long) node, (long long) node_dist);
					correct = false;
					break;
				}

				// Check that parent is in fact a parent
				bool found = false;
				for (SizeT neighbor_offset = csr_graph.row_offsets[parent];
					neighbor_offset < csr_graph.row_offsets[parent + 1];
					neighbor_offset++)
				{
					if (csr_graph.column_indices[neighbor_offset] == node) {
						found = true;
						break;
					}
				}
				if (!found) {
					printf("INCORRECT: %lld is not a neighbor of %lld",
						(long long) parent, (long long) node);
					correct = false;
					break;
				}
			}

			if (correct) {
				printf("CORRECT");
			}

		}
	}
	printf("\n");

	// Display statistics
	if (nodes_visited < 5) {
		printf("Fewer than 5 vertices visited.\n");

	} else {
		
		// Display the specific sample statistics
		double m_teps = (double) edges_visited / (elapsed * 1000.0); 
		printf("  elapsed: %.3f ms, rate: %.3f MiEdges/s", elapsed, m_teps);
		if (search_depth != 0) printf(", search_depth: %lld", (long long) search_depth);
		if (avg_duty != 0) {
			printf("\n  avg cta duty: %.2f%%", avg_duty * 100);
		}
		printf("\n  src: %lld, nodes visited: %lld, edges visited: %lld",
			(long long) src, (long long) nodes_visited, (long long) edges_visited);
		if (total_queued > 0) {
			printf(", total queued: %lld", total_queued);
		}
		if (redundant_work > 0) {
			printf(", redundant work: %.2f%%", redundant_work);
		}
		printf("\n");

		// Display the aggregate sample statistics
		printf("  Summary after %lld test iterations (bias-corrected):\n", (long long) stats.rate.count + 1);

		double search_depth_stddev = sqrt(stats.search_depth.Update((double) search_depth));
		if (search_depth > 0) printf(		"    [Search depth]:      u: %.1f, s: %.1f, cv: %.4f\n",
			stats.search_depth.mean, search_depth_stddev, search_depth_stddev / stats.search_depth.mean);

		double redundant_work_stddev = sqrt(stats.redundant_work.Update(redundant_work));
		if (redundant_work > 0) printf(		"    [redundant work %%]: u: %.2f, s: %.2f, cv: %.4f\n",
			stats.redundant_work.mean, redundant_work_stddev, redundant_work_stddev / stats.redundant_work.mean);

		double duty_stddev = sqrt(stats.duty.Update(avg_duty * 100));
		if (avg_duty > 0) printf(			"    [Duty %%]:           u: %.2f, s: %.2f, cv: %.4f\n",
			stats.duty.mean, duty_stddev, duty_stddev / stats.duty.mean);

		double rate_stddev = sqrt(stats.rate.Update(m_teps));
		printf(								"    [Time (ms)]:         u: %.3f\n",
			double(edges_visited) / stats.rate.mean / 1000.0);
		printf(								"    [Rate MiEdges/s]:    u: %.3f, s: %.3f, cv: %.4f\n",
			stats.rate.mean, rate_stddev, rate_stddev / stats.rate.mean);
	}
	
	fflush(stdout);

}
		

/******************************************************************************
 * BFS Testing Routines
 ******************************************************************************/


/**
 * A simple CPU-based reference BFS ranking implementation.  
 * 
 * Computes the distance of each node from the specified source node. 
 */
template<
	typename VertexId,
	typename Value,
	typename SizeT>
void SimpleReferenceBfs(
    bool                                    with_value,
	int 									test_iteration,
	const CsrGraph<VertexId, Value, SizeT> 	&csr_graph,
	VertexId 								*source_path,
    Value                                   *node_values,
    Value                                   *sigmas,
	VertexId 								src,
	Stats									&stats)								// running statistics
{
	// (Re)initialize distances
	for (VertexId i = 0; i < csr_graph.nodes; i++) {
		source_path[i] = -1;
		node_values[i] = 0;
		sigmas[i] = 0;
	}
	if (src == -1) src = 0;
    printf("source: %d\n", src);
	source_path[src] = 0;
	VertexId search_depth = 0;
    sigmas[src] = 1;

	// Initialize queue for managing previously-discovered nodes
	std::deque<VertexId> frontier;
	frontier.push_back(src);

	//
	// Perform BFS 
	//
	
	CpuTimer cpu_timer;
	cpu_timer.Start();
	printf("reference BC forward:\n******************\n");
	while (!frontier.empty()) {
		
		// Dequeue node from frontier
		VertexId dequeued_node = frontier.front();
		frontier.pop_front();
		VertexId neighbor_dist = source_path[dequeued_node] + 1;
        node_values[dequeued_node] = 0;
		// Locate adjacency list
		int edges_begin = csr_graph.row_offsets[dequeued_node];
		int edges_end = csr_graph.row_offsets[dequeued_node + 1];

		for (int edge = edges_begin; edge < edges_end; edge++) {
			// Lookup neighbor and enqueue if undiscovered 
			VertexId neighbor = csr_graph.column_indices[edge];
			if (with_value)
                node_values[dequeued_node] += csr_graph.edge_values[edge];
			if (source_path[neighbor] == -1) {
				source_path[neighbor] = neighbor_dist;
                sigmas[neighbor] += sigmas[dequeued_node];
				if (search_depth < neighbor_dist) {
					search_depth = neighbor_dist;
				}
				frontier.push_back(neighbor);
			}
		}
	}
	printf("\n**********************\n");
	
	cpu_timer.Stop();
	float elapsed = cpu_timer.ElapsedMillis();
	search_depth++;

	if (g_verbose) {
		Histogram(src, source_path, csr_graph, search_depth);
	}

	if (test_iteration < 0) {
		printf("Warmup iteration: %.3f ms\n", elapsed);

	} else {
		if (with_value)
		{
		    DisplayStats<false, true, VertexId, Value, SizeT>(
			    stats,
			    src,
			    source_path,
			    NULL,						// No reference source path
			    node_values,
			    NULL,
                sigmas,
                NULL,
			    csr_graph,
			    elapsed,
			    search_depth,
			    0,							// No redundant queuing
			    0);							// No barrier duty
		}
		else
		{
            DisplayStats<false, false, VertexId, Value, SizeT>(
			    stats,
			    src,
			    source_path,
			    NULL,						// No reference source path
			    node_values,
			    NULL,
                sigmas,
                NULL,
			    csr_graph,
			    elapsed,
			    search_depth,
			    0,							// No redundant queuing
			    0);							// No barrier duty
		}
	}
}


/**
 * Runs tests
 */
template <
	typename VertexId,
	typename Value,
	typename SizeT,
	bool INSTRUMENT,
	bool MARK_PREDECESSORS,
	bool WITH_VALUE>
void RunTests(
	const CsrGraph<VertexId, Value, SizeT> &csr_graph,
	VertexId src,
	bool randomized_src,
	int test_iterations,
	int max_grid_size,
	int num_gpus,
	double max_queue_sizing,
	std::vector<int> strategies)
{
	typedef bc::CsrProblem<VertexId, SizeT, Value, MARK_PREDECESSORS, WITH_VALUE> CsrProblem;

	// Allocate host-side label array (for both reference and gpu-computed results)
	VertexId* reference_labels 			= (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
	VertexId* h_labels 					= (VertexId*) malloc(sizeof(VertexId) * csr_graph.nodes);
	VertexId* reference_check 			= (g_quick) ? NULL : reference_labels;

    //Allocate host-side node_value array (both ref and gpu-computed results)
    Value* ref_node_values              = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
    Value* h_node_values                = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
    Value* ref_node_value_check         = (g_quick) ? NULL : ref_node_values;

    //Allocate host-side sigma value array (both ref and gpu-computed results)
    Value* ref_sigmas                   = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
    Value* h_sigmas                     = (Value*) malloc(sizeof(Value) * csr_graph.nodes);
    Value* ref_sigmas_check             = (g_quick) ? NULL : ref_sigmas;
    Value* h_deltas                     = (Value*) malloc(sizeof(Value) * csr_graph.nodes);

	// Allocate BFS enactor map
    bc::EnactorVertexCentric<INSTRUMENT>    vertex_centric(g_verbose);

	// Allocate Stats map
	std::map<Strategy, Stats*> stats_map;
	stats_map[HOST] 				= new Stats("Simple CPU BC");
	stats_map[VERTEX_CENTRIC] 		= new Stats("Vertex-Centric BC");

	// Allocate problem on GPU
	CsrProblem csr_problem;
	if (csr_problem.FromHostProblem(
		g_stream_from_host,
		csr_graph.nodes,
		csr_graph.edges,
		csr_graph.column_indices,
		csr_graph.row_offsets,
		csr_graph.edge_values,
        csr_graph.node_values,
		num_gpus)) exit(1);

    printf("got here.\n");

	int test_iteration = 1;
	// Perform the specified number of test iterations
	while (test_iteration <= test_iterations) {
	
		// If randomized-src was specified, re-roll the src
		if (randomized_src) src = builder::RandomNode(csr_graph.nodes);
		
		printf("---------------------------------------------------------------\n");

		//
		// Compute reference CPU BFS solution for source-distance
		//


        /*printf("compute ref value\n");
        SimpleReferenceBfs(
            WITH_VALUE,
            test_iteration,
            csr_graph,
            reference_labels,
            ref_node_values,
            ref_sigmas,
            src,
            *stats_map[HOST]);
        printf("\n");*/

        if (g_verbose2) {
            //printf("Reference solution: ");
            //DisplaySolution(reference_labels, ref_node_values, ref_sigmas, csr_graph.nodes);
            //printf("\n");
        }
        fflush(stdout);

        if (test_iteration == stats_map[HOST]->rate.count)
        {
            // Didn't start within the main connected component
            continue;
        }

		//
		// Iterate over GPU strategies
		//

		for (typename std::vector<int>::iterator itr = strategies.begin();
			itr != strategies.end();
			++itr)
		{
			Strategy strategy = (Strategy) *itr;
			Stats *stats = stats_map[strategy];

			long long 		total_queued = 0;
			VertexId		search_depth = 0;
			double			avg_duty = 0.0;
			cudaError_t		retval = cudaSuccess;

			// Perform BFS
			GpuTimer gpu_timer;
		
			{

			switch (strategy) {

			    case VERTEX_CENTRIC:
				    gpu_timer.Start();
				    if (retval = vertex_centric.EnactIterativeSearch(csr_problem, src, max_grid_size, max_queue_sizing)) break;
				    gpu_timer.Stop();
				    vertex_centric.GetStatistics(total_queued, search_depth, avg_duty);
				    break;

			    }
			}

			if (retval && (retval != cudaErrorInvalidDeviceFunction)) {
				exit(1);
			}

			float elapsed = gpu_timer.ElapsedMillis();

			// Copy out results
			if (csr_problem.ExtractResults(h_labels, h_node_values, h_sigmas, h_deltas)) exit(1);

			if ((test_iterations > 0) && (test_iteration == 0))
			{
				printf("Warmup iteration: %.3f ms\n", elapsed);
			}
			else
			{
				DisplayStats<CsrProblem::ProblemType::MARK_PREDECESSORS, CsrProblem::ProblemType::WITH_VALUE>(
					*stats,
					src,
					h_labels,
					reference_check,
					h_node_values,
					ref_node_value_check,
                    h_sigmas,
                    ref_sigmas_check,
					csr_graph,
					elapsed,
					search_depth,
					total_queued,
					avg_duty);
			}

			printf("\n");
			if (g_verbose2) {
				printf("Computed solution (%s): ", (MARK_PREDECESSORS) ? "predecessor" : "source dist");
				DisplaySolution(h_labels, h_node_values, h_sigmas, csr_graph.nodes);
				printf("\n");
			}
			fflush(stdout);

		}

		if (!randomized_src) {
			test_iteration++;
		}
	}
	
	
	//
	// Cleanup
	//
	
	if (reference_labels) free(reference_labels);
	if (h_labels) free(h_labels);
    if (ref_node_values) free(ref_node_values);
	if (h_node_values) free(h_node_values);
	for (typename std::map<Strategy, Stats*>::iterator itr = stats_map.begin(); itr != stats_map.end(); ++itr) delete itr->second;

	cudaDeviceSynchronize();
}


template <
	typename VertexId,
	typename Value,
	typename SizeT>
void RunTests(
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	CommandLineArgs &args)
{
	VertexId 	src 				= -1;			// Use whatever the specified graph-type's default is
	std::string	src_str;
	bool 		randomized_src		= false;		// Whether or not to select a new random src for each test iteration
	bool 		instrumented		= false;		// Whether or not to collect instrumentation from kernels
	bool 		mark_pred			= false;		// Whether or not to mark src-distance vs. parent vertices
	bool        with_value          = false;        // Whether or not to include edge/node computation
	int 		test_iterations 	= 1;
	int 		max_grid_size 		= 0;			// Maximum grid size (0: leave it up to the enactor)
	int 		num_gpus			= 1;			// Number of GPUs for multi-gpu enactor to use
	double 		max_queue_sizing	= 1.3;			// Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).
	std::vector<int> strategies(1, VERTEX_CENTRIC);

	instrumented = args.CheckCmdLineFlag("instrumented");
	args.GetCmdLineArgument("src", src_str);
	if (src_str.empty()) {
		// Random source
		src = builder::RandomNode(csr_graph.nodes);
	} else if (src_str.compare("randomize") == 0) {
		randomized_src = true;
	} else {
		args.GetCmdLineArgument("src", src);
	}

	g_quick = args.CheckCmdLineFlag("quick");
	mark_pred = args.CheckCmdLineFlag("mark-pred");
	with_value = args.CheckCmdLineFlag("with-value");
	args.GetCmdLineArgument("i", test_iterations);
	args.GetCmdLineArgument("max-ctas", max_grid_size);
	args.GetCmdLineArgument("num-gpus", num_gpus);
	args.GetCmdLineArgument("queue-sizing", max_queue_sizing);
	if (g_verbose2 = args.CheckCmdLineFlag("v2")) {
		g_verbose = true;
	} else {
		g_verbose = args.CheckCmdLineFlag("v");
	}
	args.GetCmdLineArguments("strategy", strategies);

	if (num_gpus > 1) {
		if (__B40C_LP64__ == 0) {
			printf("Must be compiled in 64-bit to run multiple GPUs\n");
			exit(1);
		}
		strategies.clear();
		strategies.push_back(MULTI_GPU);
	}

	// Enable symmetric peer access between gpus
	for (int gpu = 0; gpu < num_gpus; gpu++) {
		for (int other_gpu = (gpu + 1) % num_gpus;
			other_gpu != gpu;
			other_gpu = (other_gpu + 1) % num_gpus)
		{
			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			printf("Enabling peer access to GPU %d from GPU %d\n", other_gpu, gpu);

			cudaError_t error = cudaDeviceEnablePeerAccess(other_gpu, 0);
			if ((error != cudaSuccess) && (error != cudaErrorPeerAccessAlreadyEnabled)) {
				util::B40CPerror(error, "MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__);
				exit(1);
			}
		}
	}

	// Optionally display graph
	/*if (g_verbose2) {
		printf("\n");
		csr_graph.DisplayGraph();
		printf("\n");
	}*/
	csr_graph.PrintHistogram();

	//
	// Run tests
	//

 	if (instrumented) {
/* Commented out for compilation speed
		// Run instrumented kernel for per-CTA clock cycle timings
		if (mark_pred) {
			// label predecessor
			RunTests<VertexId, Value, SizeT, true, true>(
				csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, max_queue_sizing, strategies);
		} else {
			// label distance
			RunTests<VertexId, Value, SizeT, true, false>(
				csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, max_queue_sizing, strategies);
		}
*/
	} else {

		// Run regular kernel
		if (mark_pred) {
			if (with_value)
			    // label predecessor
			    RunTests<VertexId, Value, SizeT, false, true, true>(
				    csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, max_queue_sizing, strategies);
			else
                RunTests<VertexId, Value, SizeT, false, true, false>(
				    csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, max_queue_sizing, strategies);
		} else {
			if (with_value)
			    // label distance
			    RunTests<VertexId, Value, SizeT, false, false, true>(
				    csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, max_queue_sizing, strategies);
			else
                RunTests<VertexId, Value, SizeT, false, false, false>(
				    csr_graph, src, randomized_src, test_iterations, max_grid_size, num_gpus, max_queue_sizing, strategies);
		}
	}
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
		Usage();
		return 1;
	}

	DeviceInit(args);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	srand(0);									// Presently deterministic
	//srand(time(NULL));

	// Parse graph-contruction params
	g_stream_from_host = args.CheckCmdLineFlag("stream-from-host");
	g_undirected = args.CheckCmdLineFlag("undirected");
	g_with_value = args.CheckCmdLineFlag("with_value");

	std::string graph_type = argv[1];
	int flags = args.ParsedArgc();
	int graph_args = argc - flags - 1;

	if (graph_args < 1) {
		Usage();
		return 1;
	}
	
	//
	// Construct graph and perform search(es)
	//

	if (graph_type == "grid2d") {

		// Two-dimensional regular lattice grid (degree 4)
		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 2) { Usage(); return 1; }
		VertexId width = atoi(argv[2]);
		if (builder::BuildGrid2dGraph<false>(
			width,
			csr_graph) != 0)
		{
			return 1;
		}

		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "grid3d") {

		// Three-dimensional regular lattice grid (degree 6)

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 2) { Usage(); return 1; }
		VertexId width = atoi(argv[2]);
		if (builder::BuildGrid3dGraph<false>(
			width,
			csr_graph) != 0)
		{
			return 1;
		}

		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "dimacs") {

		// DIMACS-formatted graph file

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 1) { Usage(); return 1; }
		char *dimacs_filename = (graph_args == 2) ? argv[2] : NULL;
		int splice = 0;
		args.GetCmdLineArgument("splice", splice);
		if (builder::BuildDimacsGraph<false>(
			dimacs_filename,
			csr_graph,
			g_undirected,
			splice) != 0)
		{
			return 1;
		}
		
		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "metis") {

		// METIS-formatted graph file

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 1) { Usage(); return 1; }
		char *metis_filename = (graph_args == 2) ? argv[2] : NULL;
		if (builder::BuildMetisGraph<false>(metis_filename, csr_graph) != 0) {
			return 1;
		}
		
		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "market") {

		// Matrix-market coordinate-formatted graph file

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 1) { Usage(); return 1; }
		char *market_filename = (graph_args == 2) ? argv[2] : NULL;
		if (builder::BuildMarketGraph<false>(
			market_filename, 
			csr_graph, 
			g_undirected) != 0) 
		{
			return 1;
		}

		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "rmat") {

		// GTGraph R-MAT graph of n nodes and m edges (

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 3) { Usage(); return 1; }
		SizeT nodes = atol(argv[2]);
		SizeT edges = atol(argv[3]);
		if (builder::BuildRmatGraph<false>(
			nodes,
			edges,
			csr_graph,
			g_undirected,
			0.45,
			0.15,
			0.15) != 0)
		{
			return 1;
		}

		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "g500") {

		// Graph500 R-MAT graph of n nodes (8-byte vertex identifiers)

		typedef int VertexId;						// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 2) { Usage(); return 1; }
		SizeT nodes = atol(argv[2]);
		SizeT edges = nodes * 16;						// Undirected edge factor is 16, i.e., half the average degree of a vertex in the graph
		if (builder::BuildRmatGraph<false>(
			nodes,
			edges,
			csr_graph,
			true,										// Edges are undirected (i.e., add a back-edge for every directed edge sampled)
			0.57,
			0.19,
			0.19) != 0)
		{
			return 1;
		}

		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "random") {

		// Random graph of n nodes and m edges

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 3) { Usage(); return 1; }
		SizeT nodes = atol(argv[2]);
		SizeT edges = atol(argv[3]);
		if (builder::BuildRandomGraph<false>(
				nodes, 
				edges, 
				csr_graph, 
				g_undirected) != 0) 
		{
			return 1;
		}

		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "rr") {

		// Random-regular(ish) graph of n nodes, each with degree d (allows loops and cycles)

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

		if (graph_args < 3) { Usage(); return 1; }
		SizeT nodes = atol(argv[2]);
		int degree = atol(argv[3]);
		if (builder::BuildRandomRegularishGraph<false>(nodes, degree, csr_graph) != 0) {
			return 1;
        }

		// Run tests
		RunTests(csr_graph, args);

	} else if (graph_type == "fromfile") {

        // Directed graph built from CherityNet data

        typedef int VertexId;
        typedef int SizeT;
        typedef float Value;

        CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);

       if (graph_args < 2) { Usage(); return 1; }
       char* filename = argv[2];
       if (g_with_value)
       {
           if (builder::BuildGraphFromFile<true>(filename, csr_graph) != 0) {
               return 1;
           }
       }
       else
       {
       	   if (builder::BuildGraphFromFile<false>(filename, csr_graph) != 0) {
               return 1;
           }
       }

       // Run tests
       RunTests(csr_graph, args);
    } else {

		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;

	}
	

	return 0;
}
