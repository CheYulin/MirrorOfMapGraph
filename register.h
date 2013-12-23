#include <config.h>

//register the parameters from command line or configuration file
inline void registerParameters() {
//	Config::registerParameter<std::string>("source_file_name","the file of starting vertices", std::string(""));
  Config::registerParameter<int>("src","starting vertex", 1); // starting vertex
  Config::registerParameter<int>("verbose","print out frontier size in each iteration", 0);//print more infor
  Config::registerParameter<int>("origin","zero based indices or one based", 1); //vertex indices origin
  Config::registerParameter<int>("directed","the graph is directed", 1); //whether the graph is directed or not
  Config::registerParameter<int>("device","the device to use", 0); // the device number
  Config::registerParameter<double>("max_queue_sizing","frontier queue size", 3.0); //frontier queue size
}
