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
  double broadcast_time;
  double wave_time;
  double allreduce_time;
  double update_time; // update visited bitmap and label time
  double total_time;

  Statistics(int rank_id) :
      GPU_time(0.0), propagate_time(0.0), broadcast_time(0.0), wave_time(
          0.0), update_time(0.0), allreduce_time(0.0), total_time(
          0.0), rank_id(rank_id)
  {
  }
  void print_stats(void)
  {
    int rank = 0;
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    while (rank < numprocs)
    {
      if (rank_id == rank)
      {
        cout << "Rank_id: " << rank_id;
        cout << ", GPU_time: " << GPU_time;
        cout << ", propagate_time: " << propagate_time;
        cout << ", broadcast_time: " << broadcast_time;
        cout << ", wave_time: " << wave_time;
        cout << ", update_time: " << update_time;
        cout << ", allreduce_time: " << allreduce_time;
        cout << ", total_time: " << total_time << endl;
      }
      rank++;
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
};

#endif /* STATISTICS_H_ */
