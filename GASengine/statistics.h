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
#include <vector>

struct Statistics
{
  int rank_id;
  int num_procs;
  double total_GPU_time;
  double wave_setup_time;

  struct stats_per_iter
  {
    long long frontier_size;
    double GPU_time;
    double propagate_time;
    double broadcast_time;
    double wave_time;
    double allreduce_time;
    double update_time; // update visited bitmap and label time
  };

  vector<stats_per_iter> iter_stats;
  double total_propagate_time;
  double total_broadcast_time;
  double total_wave_time;
  double total_allreduce_time;
  double total_update_time; // update visited bitmap and label time
  double total_time;
  double total_iter;

  Statistics(int rank_id, int num_procs) :
      total_GPU_time(0.0), total_propagate_time(0.0), wave_setup_time(0.0), total_broadcast_time(0.0),
          total_wave_time(0.0), total_update_time(0.0), total_allreduce_time(0.0), total_time(0.0), rank_id(rank_id), num_procs(num_procs)
  {
    iter_stats.reserve(8000);
  }

  void print_stats(char* csv_file_name)
  {

    //get all stats to rank 0.
    //Getting All max times for now

    double l_GPU_time;
    double l_propagate_time = 0.0;
    double l_wave_setup_time;
    double l_broadcast_time = 0.0;
    double l_wave_time;
    double l_allreduce_time;
    double l_update_time; // update visited bitmap and label time
    double l_total_time;

//    for (int i = 0; i < iter_stats.size(); i++)
//    {
//      total_propagate_time += iter_stats[i].propagate_time;
//      total_broadcast_time += iter_stats[i].broadcast_time;
//    }

    MPI_Reduce(&total_GPU_time, &l_GPU_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_propagate_time, &l_propagate_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wave_setup_time, &l_wave_setup_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_broadcast_time, &l_broadcast_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_wave_time, &l_wave_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_allreduce_time, &l_allreduce_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_update_time, &l_update_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &l_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    l_GPU_time /= num_procs;
    l_wave_time /= num_procs;
    l_allreduce_time /= num_procs;
    l_propagate_time /= num_procs;
    l_broadcast_time /= num_procs;
    l_update_time /= num_procs;
    l_total_time /= num_procs;

    if (rank_id == 0)
    {
      cout << "AVERAGE: ";
      cout << "total_iter: " << total_iter;
      cout << ", total_GPU_time: " << l_GPU_time;
      cout << ", total_propagate_time: " << l_propagate_time;
      cout << ", wave_setup_time: " << l_wave_setup_time;
      cout << ", total_broadcast_time: " << l_broadcast_time;
      cout << ", total_wave_time: " << l_wave_time;
      cout << ", total_update_time: " << l_update_time;
      cout << ", total_allreduce_time: " << l_allreduce_time;
      cout << ", total_time: " << l_total_time << endl;

      //      fclose(f);
    }

    MPI_Reduce(&total_GPU_time, &l_GPU_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_propagate_time, &l_propagate_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wave_setup_time, &l_wave_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_broadcast_time, &l_broadcast_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_wave_time, &l_wave_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_allreduce_time, &l_allreduce_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_update_time, &l_update_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &l_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank_id == 0)
    {
      cout << "MAX: ";
      cout << "total_iter: " << total_iter;
      cout << ", total_GPU_time: " << l_GPU_time;
      cout << ", total_propagate_time: " << l_propagate_time;
      cout << ", wave_setup_time: " << l_wave_setup_time;
      cout << ", total_broadcast_time: " << l_broadcast_time;
      cout << ", total_wave_time: " << l_wave_time;
      cout << ", total_update_time: " << l_update_time;
      cout << ", total_allreduce_time: " << l_allreduce_time;
      cout << ", total_time: " << l_total_time << endl;

      //      fclose(f);
    }

//		l_propagate_time /= 49;
//		l_broadcast_time /= 49;

//    FILE* f;

//    if (rank_id == 0)
//    {
//      f = fopen(csv_file_name, "w+");
//    }
//    for (int i = 0; i < iter_stats.size(); i++)
//    {
//      double iter_GPU_time;
//      double iter_wave_time;
//      double iter_allreduce_time;
//      double iter_update_time; // update visited bitmap and label time
//      double iter_propagate_time;
//      double iter_broadcast_time;
//
//
//      MPI_Reduce(&iter_stats[i].GPU_time, &iter_GPU_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//      MPI_Reduce(&iter_stats[i].wave_time, &iter_wave_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//      MPI_Reduce(&iter_stats[i].propagate_time, &iter_propagate_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//      MPI_Reduce(&iter_stats[i].broadcast_time, &iter_broadcast_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//      MPI_Reduce(&iter_stats[i].allreduce_time, &iter_allreduce_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//      MPI_Reduce(&iter_stats[i].update_time, &iter_update_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//
//      //			cout << "iter: " << i << " rank_id: " << rank_id << " propagate_time: " << iter_stats[i].propagate_time << endl;
//      //			MPI_Barrier(MPI_COMM_WORLD);
//      //			cout << "iter: " << i << " rank_id: " << rank_id << " broadcast_time: " << iter_stats[i].broadcast_time << endl;
//      //			MPI_Barrier(MPI_COMM_WORLD);
//      //			cout << "iter: " << i << " rank_id: " << rank_id << " wave_time: " << iter_stats[i].wave_time << endl;
//      //			MPI_Barrier(MPI_COMM_WORLD);
//
//      if (rank_id == 0)
//      {
//        cout << "iter: " << i << " rank_id: " << rank_id << " propagate_time: " << iter_stats[i].propagate_time << endl;
//
//        cout << "iter: " << i << " rank_id: " << rank_id << " broadcast_time: " << iter_stats[i].broadcast_time << endl;
//
//        cout << "iter: " << i << " rank_id: " << rank_id << " wave_time: " << iter_stats[i].wave_time << endl;
//
//
//        fprintf(f,
//                "iter,%lld\nfrontier_size,%lld\nGPU_time,%lf\npropagate_time,%lf\nbroadcast_time,%lf\nwave_time,%lf\nallreduce_time,%lf\nupdate_time,%lf\n",
//                i, iter_stats[i].frontier_size, iter_GPU_time, iter_propagate_time, iter_broadcast_time, iter_wave_time, iter_allreduce_time, iter_update_time);
//
//
//        cout << "iter " << i << ": "
//                << "frontier_size: " << iter_stats[i].frontier_size
//                << ", GPU_time: " << iter_GPU_time
//                << ", propagate_time: " << iter_propagate_time
//                << ", broadcast_time: " << iter_broadcast_time
//                << ", wave_time: " << iter_wave_time
//                << ", allreduce_time: " << iter_allreduce_time
//                << ", update_time: " << iter_update_time << endl;
//      }
//    }
  }
};

#endif /* STATISTICS_H_ */
