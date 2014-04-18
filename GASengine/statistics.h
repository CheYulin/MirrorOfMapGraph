/*
 * statistics.h
 *
 *  Created on: Apr 17, 2014
 *      Author: zhisong
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
