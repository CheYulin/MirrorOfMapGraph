/**
Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This work was (partially) funded by the DARPA XDATA program under
AFRL Contract #FA8750-13-C-0002.

This material is based upon work supported by the Defense Advanced
Research Projects Agency (DARPA) under Contract No. D14PC00029.
*/

#ifndef STATISTICS_H_
#define STATISTICS_H_

using namespace std;
#include <iostream>
#include <mpi.h>

struct Statistics
{
  int rank_id;
  double GPU_time;
  double propagate_time;
	double wave_setup_time;
  double broadcast_time;
  double wave_time;
  double allreduce_time;
  double update_time; // update visited bitmap and label time
  double total_time;

  Statistics(int rank_id) :
      GPU_time(0.0), propagate_time(0.0),wave_setup_time(0.0), broadcast_time(0.0), wave_time(
          0.0), update_time(0.0), allreduce_time(0.0), total_time(
          0.0), rank_id(rank_id)
  {
  }
  void print_stats(void)
  {
	//get all stats to rank 0. 
	//Getting All max times for now

  double l_GPU_time;
  double l_propagate_time;
	double l_wave_setup_time;
  double l_broadcast_time;
  double l_wave_time;
  double l_allreduce_time;
  double l_update_time; // update visited bitmap and label time
  double l_total_time;

	MPI_Reduce(&GPU_time,&l_GPU_time,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&propagate_time,&l_propagate_time,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&wave_setup_time,&l_wave_setup_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	MPI_Reduce(&broadcast_time,&l_broadcast_time,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&wave_time,&l_wave_time,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&allreduce_time,&l_allreduce_time,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&update_time,&l_update_time,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&total_time,&l_total_time,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if (rank_id == 0)
      {
        cout << "GPU_time: " << l_GPU_time;
        cout << ", propagate_time: " << l_propagate_time;
	cout << ", wave_setup_time: " << l_wave_setup_time;
        cout << ", broadcast_time: " << l_broadcast_time;
        cout << ", wave_time: " << l_wave_time;
        cout << ", update_time: " << l_update_time;
        cout << ", allreduce_time: " << l_allreduce_time;
        cout << ", total_time: " << l_total_time << endl;
      }
  }
};

#endif /* STATISTICS_H_ */
