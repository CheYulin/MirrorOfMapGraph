#include <config.h>

//#include <amg_level.h>
//#include <norm.h>
//#include <convergence.h>
//#include <smoothers/smoother.h>
//#include <cycles/cycle.h>
//#include <smoothedMG/aggregators/aggregator.h>
inline void registerParameters() {
//	AMG_Config::registerParameter<std::string>("source_file_name","the file of starting vertices", std::string(""));
  Config::registerParameter<int>("src","starting vertex", 1);
  Config::registerParameter<int>("verbose","print out frontier size in each iteration", 0);
  Config::registerParameter<int>("origin","zero based indices or one based", 1);
  Config::registerParameter<int>("directed","the graph is directed", 1);
  Config::registerParameter<int>("device","the device to use", 0);
  Config::registerParameter<double>("max_queue_sizing","frontier queue size", 3.0);
}
