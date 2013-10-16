/****************************************************************
 * Simple test driver program for CC labeling
 *
 ****************************************************************/

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <fstream>

// Utilities and correctness-checking
#include <b40c_test_util.h>

// Graph construction utils
#include <b40c/graph/builder/fromfile.cuh>
#include <b40c/graph/builder/market.cuh>

// CC includes
#include <b40c/graph/cc/coo_problem.cuh>
#include <b40c/graph/cc/enactor.cuh>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

using namespace b40c;
using namespace graph;

/**
 * Defines, constants, globals
 */
bool g_verbose;

bool g_undirected;



/**
 * Display the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_cc <graph_type> <graph_type_args> [--device=<device_index>]  "
	        "[--v] [--instrumented] [--i=<num-iterations>]"
	        "\n"
	        "All graph types use 8-byte vertex-identifiers.\n"
	        "\n"
	        "Graph types and args:\n"
	        "  fromfile <filename>\n"
	        "    Reads an undirected graph from file.\n"
	        "\n"
	        "--v Verbose launch and statistical output is displayed to the console.\n"
	        "\n"
	        "--instrumented Kernels keep track of iteration and average barrier duty.\n"
	        "\n"
	        "--i Performs <num-iterations> test-iterations of CC algorithms.\n"
	        "     Default = 1\n"
	        "--gpu_index Index of device intend to use.\n");
}

template <typename VertexId>
struct CcList {
    VertexId root;
    unsigned int histogram;

    CcList(VertexId root, unsigned int histogram) : root(root), histogram(histogram) {}
    };

template<typename CcList>
bool CCCompare (
    CcList elem1,
    CcList elem2)
    {
        if (elem1.histogram > elem2.histogram) {
            return true;
        }
        return false;
    }


/**
 * Displays the CC result
 */
template<typename VertexId>
void DisplaySolution(VertexId* node_parent, VertexId* edge_from, VertexId* edge_to, bool *masks, int num_nodes, int num_edges, unsigned int num_component, VertexId* roots, unsigned int *histogram)
{
    typedef CcList<VertexId> CcListType;
	printf("Number of components: %d\n", num_component);

    if (num_nodes < 50)
    {
	    for (int i = 0; i < num_nodes; ++i)
	    {
		    printf("node:%d component number:%ld\n",i, node_parent[i]);
	    }
	}

    //sort the componnets by size
    CcListType *cclist = (CcListType*) malloc(sizeof(CcListType) * num_component);
    for (int i = 0; i < num_component; ++i)
    {
        cclist[i].root = roots[i];
        cclist[i].histogram = histogram[i];
    }
    std::stable_sort(cclist, cclist + num_component, CCCompare<CcListType>);

    int top_10 = (num_component < 10) ? num_component : 10;

    for (int i = 0; i < top_10; ++i)
    {
        printf("cc id: %d, cc root: %d, cc size: %d\n", i, cclist[i].root, cclist[i].histogram);
    }

    //output node_parents to a file
    std::ofstream of("cc_labels.txt");
    for (int i = 0; i < num_nodes; ++i)
    {
        if (node_parent[i] != cclist[0].root)
            of << i << " " << node_parent[i] << " " << std::endl;
    }

    std::ofstream of1("largest_cc.txt");
    long largest_cc_edge_num = 0;
    for (int i = 0; i < num_edges; ++i)
    {
        edge_from[i] = edge_from[i] >> 32;
        if (node_parent[edge_from[i]] == cclist[0].root && node_parent[edge_to[i]] == cclist[0].root)
        {
            of1 << edge_from[i] << "\t" << edge_to[i] <<"\t" << "1"<< std::endl;
            largest_cc_edge_num++;
        }
    }
    printf("largest_cc_edge_num: %ld\n", largest_cc_edge_num);


    if (cclist) free(cclist);
}

/**
 * Transform csr graph to coo edge tuple
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void PrepareData(
    VertexId *node_parents,
    VertexId *from,
    VertexId *to,
    const CsrGraph<VertexId, Value, SizeT> &csr_graph)
{
	for (int i = 0; i < csr_graph.nodes; ++i)
	{
		node_parents[i] = i;
	}

	for (int i = 0; i < csr_graph.edges; ++i)
	{
		from[i] = csr_graph.from_nodes[i];
		to[i] = csr_graph.to_nodes[i];
	}
}


/**
 * CPU-based reference CC algorithm using Boost Graph Library
 */
//TODO: implement this
template<typename VertexId>
unsigned int RefCPUCC(VertexId *from_node, VertexId *to_node, int num_edges, int *labels)
{
    using namespace boost;
    typedef adjacency_list <vecS, vecS, undirectedS> Graph;
    Graph G;
    for (int i = 0; i < num_edges; ++i)
    {
        add_edge(from_node[i], to_node[i], G);
    }
    int num_nodes = num_vertices(G);
    int num_components = connected_components(G, &labels[0]);
    return num_components;
}

template <typename VertexId>
void VerifyResults(VertexId *h_node_parents, int *h_ref_node_parents, unsigned int num_components, unsigned int ref_num_components, unsigned int num_nodes)
{
    bool flag = true;
    if (num_components != ref_num_components)
        flag = false;
    
    if (flag)
        printf("\n PASSED TEST.\n");
    else
    {
        printf("\n %d \n", ref_num_components);
        printf("\n FAILED TEST.\n");
        exit(1);
    }
}



/**
 * Run tests
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT>
void RunTests(
    const CsrGraph<VertexId, Value, SizeT> &csr_graph,
    int test_iterations,
    int max_grid_size,
    int gpu_index)
{
	typedef cc::CooProblem<VertexId> CooProblem;

	VertexId *h_node_parents            = (VertexId*)malloc(sizeof(VertexId) * csr_graph.nodes);
	VertexId *h_from = (VertexId*)malloc(sizeof(VertexId) * csr_graph.edges);
	VertexId *h_to = (VertexId*)malloc(sizeof(VertexId) * csr_graph.edges);
	bool     *h_masks = (bool*)malloc(sizeof(bool) * csr_graph.nodes);
    VertexId *h_cc_bins = (VertexId*)malloc(sizeof(VertexId) * csr_graph.nodes);
    int *h_ref_node_parents        = (int*)malloc(sizeof(int) * csr_graph.nodes);
	
	cc::Enactor<INSTRUMENT>             cc_enactor(g_verbose);

	//Allocate problem on GPU
	CooProblem  coo_problem;
	PrepareData<VertexId, Value, SizeT> (h_node_parents, h_from, h_to, csr_graph);
	if (coo_problem.FromHostProblem(
		csr_graph.nodes,
		csr_graph.edges,
		h_from,
		h_to,
		h_node_parents,
		gpu_index)) exit(1);
    
    //Run CPU reference Connected Component Labeling algorithm
    unsigned int ref_num_components = RefCPUCC(h_from, h_to, coo_problem.edges, h_ref_node_parents);
	
	int test_iteration = 1;
	// Perform the specified number of test iterations
	while (test_iteration <=  test_iterations) {

		cudaError_t retval = cudaSuccess;
		
		// Perform CC
		GpuTimer gpu_timer;

        gpu_timer.Start();
		if (retval = cc_enactor.EnactCC(coo_problem, max_grid_size)) break;
		gpu_timer.Stop();

		if (retval && (retval != cudaErrorInvalidDeviceFunction)) {
			exit(1);
		}

		float elapsed = gpu_timer.ElapsedMillis();

		// Copy out results
		if (coo_problem.ExtractResults(h_node_parents, h_from, h_masks, h_cc_bins)) exit(1);

        VerifyResults(h_node_parents, h_ref_node_parents, coo_problem.num_components, ref_num_components, coo_problem.nodes);

        unsigned int *h_cc_histograms = (unsigned int*)malloc(sizeof(unsigned int) * coo_problem.num_components);
        coo_problem.CountComponentSize(h_node_parents, coo_problem.num_components,
                           h_cc_histograms, h_cc_bins);

        printf("Connected Component Labeling algorithm finishes in %lfms\n", elapsed);
		DisplaySolution(h_node_parents, h_from, h_to, h_masks, coo_problem.nodes, coo_problem.edges, coo_problem.num_components, h_cc_bins, h_cc_histograms);

		fflush(stdout);
		test_iteration++;
        if (h_cc_histograms) free(h_cc_histograms);

	}
    
    if (h_cc_bins) free(h_cc_bins);
    if (h_masks) free(h_masks);
	if (h_node_parents) free(h_node_parents);
	if (h_from) free(h_from);
	if (h_to) free(h_to);

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
	bool    instrumented        = false;
	int     test_iterations     = 1;
	int     max_grid_size       = 0;
	int     gpu_index           = 0;

	instrumented = args.CheckCmdLineFlag("instrumented");
	args.GetCmdLineArgument("i", test_iterations);
	args.GetCmdLineArgument("max-ctas", max_grid_size);
	args.GetCmdLineArgument("gpu-index", gpu_index);
	g_verbose = args.CheckCmdLineFlag("v");

    csr_graph.PrintHistogram();

	if (instrumented) {
		RunTests<VertexId, Value, SizeT, true>(
		    csr_graph, test_iterations, max_grid_size, gpu_index);
	}
	else {
		RunTests<VertexId, Value, SizeT, false>(
		    csr_graph, test_iterations, max_grid_size, gpu_index);
	}
}

/**
 * Main
 */

int main( int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
		Usage();
		return 1;
	}

    g_undirected = args.CheckCmdLineFlag("undirected");

	DeviceInit(args);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	std::string graph_type = argv[1];
	int flags = args.ParsedArgc();
	int graph_args = argc - flags - 1;

	if (graph_args < 1) {
		Usage();
		return 1;
	}

	if (graph_type == "fromfile") {
		//Undirected graph built from file
		typedef long long int  VertexId;
		typedef int             SizeT;
		typedef float           Value;

		CsrGraph<VertexId, Value, SizeT> csr_graph(false);

		if (graph_args < 2) { Usage(); return 1; }
		char* filename = argv[2];
        if (builder::BuildGraphFromFile<false>(filename, csr_graph) != 0) {
            return 1;
        }

		// Run Tests
		RunTests(csr_graph, args);
	} else if (graph_type == "market") {

		// Matrix-market coordinate-formatted graph file

		typedef long long int VertexId;						
		typedef int           SizeT;	
		typedef float         Value;								
		CsrGraph<VertexId, Value, SizeT> csr_graph(false);

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

	} else {
		//Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;
	}

	return 0;

}

		    



