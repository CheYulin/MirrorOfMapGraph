#pragma once
#include <stdlib.h>

template <class T> 
inline T getValue(const char* name);

template <> 
inline int getValue<int>(const char* name) {
  return atoi(name);
}

template <> 
inline bool getValue<bool>(const char* name) {
  if(strcmp(name, "true"))
		  return true;
  else if(strcmp(name, "false"))
	  return false;
  else
  {
	  std::cout << "Bool value error, use default value false" << std::endl;
	  return false;
  }
}

template <>
inline float getValue<float>(const char* name) {
  return atof(name);
}

template <> 
inline double getValue<double>(const char* name) {
  return atof(name);
}
