#include <config.h>

//register the parameters from command line or configuration file
inline void registerParameters() {
//	Config::registerParameter<std::string>("source_file_name","the file of starting vertices", std::string(""));
  Config::registerParameter<int>("src","Starting vertex (default 1)", 1); // starting vertex
  Config::registerParameter<int>("verbose","Print out frontier size in each iteration (default 0)", 0);//print more infor
  Config::registerParameter<int>("origin","The origin (0 or 1) for the starting vertices (default 1)", 1); //vertex indices origin
  Config::registerParameter<int>("directed","The graph is directed (default 1)", 1); //whether the graph is directed or not
  Config::registerParameter<int>("device","The device to use (default 0)", 0); // the device number
  Config::registerParameter<int>("iter_num","The number of iterations to perform (default 1)", 1);
  Config::registerParameter<int>("num_src","The number of starting vertices when random sources is specified (default 1)", 1); // the device number
  Config::registerParameter<int>("with_value","Whether to load edge values from market file (default 0)", 0); // the device number
  Config::registerParameter<double>("max_queue_sizing","The frontier queue size is this value times the number of vertices in the graph (default 3.0)", 3.0); //frontier queue size
}
